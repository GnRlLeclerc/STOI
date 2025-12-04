//! STOI constants

pub const DYNAMIC_RANGE: f64 = 40.0;

// Audio frame length
pub const FRAME_LENGTH: usize = 256; // For stft and energy as well

// Audio frame hop length (half overlap)
pub const HOP_LENGTH: usize = FRAME_LENGTH / 2;

// Internal sampling frequency for STOI computation
pub const FS: u32 = 10_000;

pub const FFT_LENGTH: usize = 512;
pub const FFT_BINS: usize = FFT_LENGTH / 2 + 1;

pub const NUM_BANDS: usize = 15; // Amount of 13 octave band

pub const SEGMENT_LENGTH: usize = 30;
