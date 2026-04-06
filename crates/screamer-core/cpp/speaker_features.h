#ifndef SCREAMER_SPEAKER_FEATURES_H
#define SCREAMER_SPEAKER_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of MFCC coefficients per frame. */
#define SCREAMER_NUM_MFCC 13

/*
 * Extract a speaker embedding from raw 16 kHz mono f32 audio.
 *
 * Writes NUM_MFCC floats into `out_mean` (mean MFCCs across frames)
 * and NUM_MFCC floats into `out_std` (std-dev MFCCs across frames).
 *
 * Returns 0 on success, -1 if audio is too short.
 */
int screamer_extract_speaker_embedding(
    const float *samples,
    size_t num_samples,
    float *out_mean,
    float *out_std);

/*
 * Cosine similarity between two embedding vectors (mean + std concatenated).
 * Each vector is 2 * NUM_MFCC floats long.
 * Returns a value in [-1, 1].
 */
float screamer_embedding_similarity(
    const float *a_mean, const float *a_std,
    const float *b_mean, const float *b_std);

#ifdef __cplusplus
}
#endif

#endif /* SCREAMER_SPEAKER_FEATURES_H */
