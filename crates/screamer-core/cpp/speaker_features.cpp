/*
 * Speaker feature extraction — MFCC-based voice fingerprinting.
 *
 * On macOS uses Apple Accelerate (vDSP) for FFT, which is natively
 * optimised for Apple Silicon NEON/AMX units. On other platforms falls
 * back to a small portable radix-2 FFT.
 *
 * All processing assumes 16 kHz mono float32 input.
 */

#include "speaker_features.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

static constexpr int   SAMPLE_RATE     = 16000;
static constexpr int   FRAME_SIZE      = 400;   /* 25 ms */
static constexpr int   HOP_SIZE        = 160;   /* 10 ms */
static constexpr int   FFT_SIZE        = 512;   /* next pow2 >= FRAME_SIZE */
static constexpr int   FFT_HALF        = FFT_SIZE / 2 + 1;
static constexpr int   NUM_MEL_FILTERS = 26;
static constexpr int   NUM_MFCC        = SCREAMER_NUM_MFCC;
static constexpr float MEL_LOW_HZ      = 80.0f;
static constexpr float MEL_HIGH_HZ     = 7600.0f;
static constexpr float PI_F            = 3.14159265358979323846f;

/* ------------------------------------------------------------------ */
/*  Platform FFT                                                       */
/* ------------------------------------------------------------------ */

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>

struct PlatformFFT {
    FFTSetup setup;
    int log2n;

    PlatformFFT() {
        log2n = 0;
        int n = FFT_SIZE;
        while (n > 1) { n >>= 1; log2n++; }
        setup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    }

    ~PlatformFFT() {
        vDSP_destroy_fftsetup(setup);
    }

    /*
     * In-place forward FFT.  `re` and `im` are length FFT_SIZE.
     * After return, bins 0..FFT_HALF-1 hold the one-sided spectrum.
     */
    void forward(float *re, float *im) const {
        DSPSplitComplex sc{re, im};
        vDSP_fft_zip(setup, &sc, 1, log2n, kFFTDirection_Forward);
    }
};

#else
/* Portable radix-2 Cooley–Tukey FFT for non-Apple platforms. */

struct PlatformFFT {
    PlatformFFT() = default;

    void forward(float *re, float *im) const {
        /* Bit-reversal permutation */
        for (int i = 1, j = 0; i < FFT_SIZE; i++) {
            int bit = FFT_SIZE >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }

        /* Butterfly stages */
        for (int len = 2; len <= FFT_SIZE; len <<= 1) {
            float ang = -2.0f * PI_F / (float)len;
            float wre = cosf(ang), wim = sinf(ang);
            for (int i = 0; i < FFT_SIZE; i += len) {
                float cur_re = 1.0f, cur_im = 0.0f;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    float tre = re[v] * cur_re - im[v] * cur_im;
                    float tim = re[v] * cur_im + im[v] * cur_re;
                    re[v] = re[u] - tre;
                    im[v] = im[u] - tim;
                    re[u] += tre;
                    im[u] += tim;
                    float new_re = cur_re * wre - cur_im * wim;
                    cur_im = cur_re * wim + cur_im * wre;
                    cur_re = new_re;
                }
            }
        }
    }
};
#endif /* __APPLE__ */

/* ------------------------------------------------------------------ */
/*  Pre-computed tables (lazy-initialised, thread-safe via static)      */
/* ------------------------------------------------------------------ */

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

struct Tables {
    float hamming[FRAME_SIZE];
    float mel_bank[NUM_MEL_FILTERS][FFT_HALF];
    float dct[NUM_MFCC][NUM_MEL_FILTERS];

    Tables() {
        /* Hamming window */
        for (int n = 0; n < FRAME_SIZE; n++) {
            hamming[n] = 0.54f - 0.46f * cosf(2.0f * PI_F * (float)n / (float)(FRAME_SIZE - 1));
        }

        /* Mel filterbank */
        float mel_low  = hz_to_mel(MEL_LOW_HZ);
        float mel_high = hz_to_mel(MEL_HIGH_HZ);
        int   num_pts  = NUM_MEL_FILTERS + 2;
        std::vector<float> hz_pts(num_pts);
        std::vector<int>   bin_pts(num_pts);

        for (int i = 0; i < num_pts; i++) {
            float mel = mel_low + (mel_high - mel_low) * (float)i / (float)(num_pts - 1);
            hz_pts[i]  = mel_to_hz(mel);
            bin_pts[i] = (int)roundf(hz_pts[i] / (float)SAMPLE_RATE * (float)FFT_SIZE);
        }

        memset(mel_bank, 0, sizeof(mel_bank));
        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            int left   = bin_pts[m];
            int center = bin_pts[m + 1];
            int right  = bin_pts[m + 2];
            for (int k = left; k < center && k < FFT_HALF; k++) {
                if (center > left)
                    mel_bank[m][k] = (float)(k - left) / (float)(center - left);
            }
            for (int k = center; k < right && k < FFT_HALF; k++) {
                if (right > center)
                    mel_bank[m][k] = (float)(right - k) / (float)(right - center);
            }
        }

        /* DCT-II matrix */
        float scale = sqrtf(2.0f / (float)NUM_MEL_FILTERS);
        for (int i = 0; i < NUM_MFCC; i++) {
            for (int j = 0; j < NUM_MEL_FILTERS; j++) {
                dct[i][j] = scale * cosf(PI_F * (float)i * ((float)j + 0.5f) / (float)NUM_MEL_FILTERS);
            }
        }
    }
};

static const Tables &tables() {
    static Tables t;
    return t;
}

/* ------------------------------------------------------------------ */
/*  Core MFCC extraction                                               */
/* ------------------------------------------------------------------ */

extern "C"
int screamer_extract_speaker_embedding(
    const float *samples,
    size_t       num_samples,
    float       *out_mean,
    float       *out_std)
{
    if (num_samples < (size_t)FRAME_SIZE) return -1;

    const auto &t   = tables();
    PlatformFFT fft;

    size_t num_frames = (num_samples - FRAME_SIZE) / HOP_SIZE + 1;
    if (num_frames == 0) return -1;

    double mfcc_sum[NUM_MFCC]    = {};
    double mfcc_sq_sum[NUM_MFCC] = {};

    float re[FFT_SIZE];
    float im[FFT_SIZE];
    float power[FFT_HALF];
    float mel_energy[NUM_MEL_FILTERS];
    float mfcc[NUM_MFCC];

    for (size_t f = 0; f < num_frames; f++) {
        size_t start = f * HOP_SIZE;

        /* Window + zero-pad */
        for (int i = 0; i < FFT_SIZE; i++) {
            if (i < FRAME_SIZE)
                re[i] = samples[start + i] * t.hamming[i];
            else
                re[i] = 0.0f;
            im[i] = 0.0f;
        }

        fft.forward(re, im);

        /* Power spectrum */
        float inv = 1.0f / (float)FFT_SIZE;
        for (int k = 0; k < FFT_HALF; k++) {
            power[k] = (re[k] * re[k] + im[k] * im[k]) * inv;
        }

        /* Mel filterbank */
        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            float sum = 0.0f;
            for (int k = 0; k < FFT_HALF; k++) {
                sum += t.mel_bank[m][k] * power[k];
            }
            mel_energy[m] = logf(sum + 1e-10f);
        }

        /* DCT -> MFCCs */
        for (int i = 0; i < NUM_MFCC; i++) {
            float sum = 0.0f;
            for (int j = 0; j < NUM_MEL_FILTERS; j++) {
                sum += t.dct[i][j] * mel_energy[j];
            }
            mfcc[i] = sum;
        }

        for (int i = 0; i < NUM_MFCC; i++) {
            mfcc_sum[i]    += (double)mfcc[i];
            mfcc_sq_sum[i] += (double)mfcc[i] * (double)mfcc[i];
        }
    }

    double n = (double)num_frames;
    for (int i = 0; i < NUM_MFCC; i++) {
        out_mean[i] = (float)(mfcc_sum[i] / n);
        double var  = (mfcc_sq_sum[i] / n) - (mfcc_sum[i] / n) * (mfcc_sum[i] / n);
        out_std[i]  = (float)sqrt(var > 0.0 ? var : 0.0);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Cosine similarity                                                  */
/* ------------------------------------------------------------------ */

extern "C"
float screamer_embedding_similarity(
    const float *a_mean, const float *a_std,
    const float *b_mean, const float *b_std)
{
    double dot    = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    /* Skip c0 (index 0) — it encodes energy / volume, not voice identity. */
    for (int i = 1; i < NUM_MFCC; i++) {
        double va = (double)a_mean[i];
        double vb = (double)b_mean[i];
        dot    += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    for (int i = 1; i < NUM_MFCC; i++) {
        double va = (double)a_std[i];
        double vb = (double)b_std[i];
        dot    += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    double denom = sqrt(norm_a) * sqrt(norm_b);
    if (denom < 1e-12) return 0.0f;
    return (float)(dot / denom);
}
