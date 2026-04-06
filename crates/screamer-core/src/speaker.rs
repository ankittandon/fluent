//! Voice-based speaker identification using MFCC embeddings.
//!
//! The heavy lifting (FFT, mel filterbank, DCT) is done in C++ via
//! Apple Accelerate on macOS and a portable fallback on other platforms.

use crate::ambient::SpeakerLabel;

/// Number of MFCC coefficients — must match `SCREAMER_NUM_MFCC` in C++.
pub const NUM_MFCC: usize = 13;

/// Similarity threshold: above this, two embeddings are considered the same speaker.
const SAME_SPEAKER_THRESHOLD: f32 = 0.88;

/// Minimum audio length in samples (at 16 kHz) required for a usable embedding.
/// ~200 ms = 3200 samples. Shorter clips produce unreliable fingerprints.
const MIN_SAMPLES_FOR_EMBEDDING: usize = 3200;

extern "C" {
    fn screamer_extract_speaker_embedding(
        samples: *const f32,
        num_samples: usize,
        out_mean: *mut f32,
        out_std: *mut f32,
    ) -> i32;

    fn screamer_embedding_similarity(
        a_mean: *const f32,
        a_std: *const f32,
        b_mean: *const f32,
        b_std: *const f32,
    ) -> f32;
}

/// A compact voice fingerprint derived from MFCC statistics.
#[derive(Clone, Debug)]
pub struct SpeakerEmbedding {
    pub mfcc_mean: [f32; NUM_MFCC],
    pub mfcc_std: [f32; NUM_MFCC],
}

impl SpeakerEmbedding {
    /// Extract a speaker embedding from raw 16 kHz mono audio.
    /// Returns `None` if the audio is too short for a reliable fingerprint.
    pub fn from_samples(samples: &[f32]) -> Option<Self> {
        if samples.len() < MIN_SAMPLES_FOR_EMBEDDING {
            return None;
        }

        let mut mfcc_mean = [0.0f32; NUM_MFCC];
        let mut mfcc_std = [0.0f32; NUM_MFCC];

        let rc = unsafe {
            screamer_extract_speaker_embedding(
                samples.as_ptr(),
                samples.len(),
                mfcc_mean.as_mut_ptr(),
                mfcc_std.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return None;
        }

        Some(SpeakerEmbedding {
            mfcc_mean,
            mfcc_std,
        })
    }

    /// Cosine similarity to another embedding. Returns a value in [-1, 1].
    pub fn similarity(&self, other: &SpeakerEmbedding) -> f32 {
        unsafe {
            screamer_embedding_similarity(
                self.mfcc_mean.as_ptr(),
                self.mfcc_std.as_ptr(),
                other.mfcc_mean.as_ptr(),
                other.mfcc_std.as_ptr(),
            )
        }
    }
}

/// Running centroid for a known speaker, updated as we hear more of their voice.
#[derive(Clone, Debug)]
pub struct SpeakerProfile {
    pub label: SpeakerLabel,
    mfcc_sum: [f64; NUM_MFCC],
    mfcc_std_sum: [f64; NUM_MFCC],
    count: usize,
}

impl SpeakerProfile {
    pub fn new(label: SpeakerLabel, embedding: &SpeakerEmbedding) -> Self {
        let mut mfcc_sum = [0.0f64; NUM_MFCC];
        let mut mfcc_std_sum = [0.0f64; NUM_MFCC];
        for i in 0..NUM_MFCC {
            mfcc_sum[i] = embedding.mfcc_mean[i] as f64;
            mfcc_std_sum[i] = embedding.mfcc_std[i] as f64;
        }
        Self {
            label,
            mfcc_sum,
            mfcc_std_sum,
            count: 1,
        }
    }

    pub fn update(&mut self, embedding: &SpeakerEmbedding) {
        for i in 0..NUM_MFCC {
            self.mfcc_sum[i] += embedding.mfcc_mean[i] as f64;
            self.mfcc_std_sum[i] += embedding.mfcc_std[i] as f64;
        }
        self.count += 1;
    }

    pub fn centroid(&self) -> SpeakerEmbedding {
        let n = self.count as f64;
        let mut mfcc_mean = [0.0f32; NUM_MFCC];
        let mut mfcc_std = [0.0f32; NUM_MFCC];
        for i in 0..NUM_MFCC {
            mfcc_mean[i] = (self.mfcc_sum[i] / n) as f32;
            mfcc_std[i] = (self.mfcc_std_sum[i] / n) as f32;
        }
        SpeakerEmbedding {
            mfcc_mean,
            mfcc_std,
        }
    }
}

/// Identifies speakers by comparing voice embeddings against known profiles.
#[derive(Clone, Debug, Default)]
pub struct SpeakerIdentifier {
    profiles: Vec<SpeakerProfile>,
}

/// The ordered labels assigned to new speakers.
const SPEAKER_ORDER: [SpeakerLabel; 6] = [
    SpeakerLabel::S1,
    SpeakerLabel::S2,
    SpeakerLabel::S3,
    SpeakerLabel::S4,
    SpeakerLabel::S5,
    SpeakerLabel::S6,
];

impl SpeakerIdentifier {
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Identify the speaker for a chunk of audio.
    /// Compares the voice embedding against all known profiles.
    /// If it matches an existing speaker, updates that profile and returns the label.
    /// If it doesn't match anyone, creates a new speaker profile.
    pub fn identify(&mut self, samples: &[f32]) -> Option<SpeakerLabel> {
        let embedding = SpeakerEmbedding::from_samples(samples)?;

        // Find best matching profile
        let mut best_idx: Option<usize> = None;
        let mut best_sim: f32 = -1.0;

        for (idx, profile) in self.profiles.iter().enumerate() {
            let centroid = profile.centroid();
            let sim = embedding.similarity(&centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        if best_sim >= SAME_SPEAKER_THRESHOLD {
            let idx = best_idx.unwrap();
            self.profiles[idx].update(&embedding);
            return Some(self.profiles[idx].label);
        }

        // New speaker
        if self.profiles.len() >= SPEAKER_ORDER.len() {
            // At capacity — assign to closest match
            if let Some(idx) = best_idx {
                self.profiles[idx].update(&embedding);
                return Some(self.profiles[idx].label);
            }
        }

        let label = SPEAKER_ORDER[self.profiles.len()];
        self.profiles.push(SpeakerProfile::new(label, &embedding));
        Some(label)
    }

    /// Get the current number of distinct speakers detected.
    pub fn speaker_count(&self) -> usize {
        self.profiles.len()
    }

    /// Reset all speaker profiles (e.g. at session start).
    pub fn reset(&mut self) {
        self.profiles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: usize = 16_000;

    fn sine_wave(freq: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (SR as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn short_audio_returns_none() {
        assert!(SpeakerEmbedding::from_samples(&[0.0; 100]).is_none());
    }

    #[test]
    fn valid_audio_returns_embedding() {
        let samples = sine_wave(440.0, 0.5, 0.5);
        assert!(SpeakerEmbedding::from_samples(&samples).is_some());
    }

    #[test]
    fn identical_audio_similarity_near_one() {
        let samples = sine_wave(440.0, 0.5, 0.5);
        let a = SpeakerEmbedding::from_samples(&samples).unwrap();
        let b = SpeakerEmbedding::from_samples(&samples).unwrap();
        assert!((a.similarity(&b) - 1.0).abs() < 0.01);
    }

    #[test]
    fn different_tones_lower_similarity() {
        let low = sine_wave(200.0, 0.5, 0.5);
        let high = sine_wave(3000.0, 0.5, 0.5);
        let a = SpeakerEmbedding::from_samples(&low).unwrap();
        let b = SpeakerEmbedding::from_samples(&high).unwrap();
        assert!(a.similarity(&b) < 0.85);
    }

    #[test]
    fn identifier_assigns_consistent_labels() {
        let voice_a = sine_wave(300.0, 0.5, 0.5);
        let voice_b = sine_wave(2500.0, 0.5, 0.5);

        let mut ident = SpeakerIdentifier::new();
        let label_a1 = ident.identify(&voice_a).unwrap();
        let label_b1 = ident.identify(&voice_b).unwrap();
        let label_a2 = ident.identify(&voice_a).unwrap();

        assert_eq!(label_a1, SpeakerLabel::S1);
        assert_eq!(label_a2, label_a1, "same voice should get same label");
        assert_ne!(label_a1, label_b1, "different voices should differ");
    }
}
