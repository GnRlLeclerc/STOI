//! Rust STOI implementation

mod constants;
mod errors;
mod extended;
mod frames;
mod octave;
mod resample;
mod standard;
mod stft;
mod upfirdn;

use crate::{
    constants::{FS, SEGMENT_LENGTH},
    errors::{NotEnoughFramesError, Result},
};

/// Do the full computation post resampling to 10kHz
fn compute(x: &[f32], y: &[f32], extended: bool) -> Result<f32> {
    // Compute frames
    let (x_frames, y_frames, mask, count) = frames::process_frames(x, y);

    if count < SEGMENT_LENGTH {
        return Err(NotEnoughFramesError);
    }

    // Compute spectrograms
    let x_spec = stft::compute_frame_rffts(x_frames.as_ref(), mask.as_ref(), count);
    let y_spec = stft::compute_frame_rffts(y_frames.as_ref(), mask.as_ref(), count);

    // Accumulate into octave bands
    let x_bands = octave::compute_octave_bands(x_spec.as_ref());
    let y_bands = octave::compute_octave_bands(y_spec.as_ref());

    // Slice into segments
    let mut x_segments = frames::segments(x_bands.transpose());
    let mut y_segments = frames::segments(y_bands.transpose());

    if extended {
        Ok(extended::from_segments(
            x_segments.as_mut(),
            y_segments.as_mut(),
        ))
    } else {
        Ok(standard::from_segments(
            x_segments.as_mut(),
            y_segments.as_mut(),
        ))
    }
}

/// Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
///
/// Args:
/// * `x` - Clean speech signal
/// * `y` - Processed speech signal
/// * `fs_sig` - Sampling frequency of the signals
/// * `extended` - Whether to use the extended STOI measure
pub fn stoi(x: &[f32], y: &[f32], fs_sig: usize, extended: bool) -> Result<f32> {
    assert!(
        x.len() == y.len(),
        "Input signals must have the same length"
    );

    if fs_sig != FS {
        let x = resample::resample(x, fs_sig, FS);
        let y = resample::resample(y, fs_sig, FS);

        compute(&x, &y, extended)
    } else {
        compute(x, y, extended)
    }
}
