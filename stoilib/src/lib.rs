//! Rust STOI implementation

mod constants;
mod frames;
mod octave;
mod resample;
mod standard;
mod stft;

use ndarray::prelude::*;

use crate::constants::FS;

/// Do the full computation post resampling to 10kHz
fn compute(x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) -> f64 {
    // Compute frames
    let (x_frames, y_frames, mask, count) = frames::process_frames(x.view(), y.view());

    // Compute spectrograms
    let x_spec = stft::compute_frame_rffts(x_frames.view(), mask.view(), count);
    let y_spec = stft::compute_frame_rffts(y_frames.view(), mask.view(), count);

    // Accumulate into octave bands
    let x_bands = octave::compute_octave_bands(x_spec.view());
    let y_bands = octave::compute_octave_bands(y_spec.view());
    let x_bands_t = x_bands.t();
    let y_bands_t = y_bands.t();

    // Slice into segments
    let mut x_segments = frames::segments(x_bands_t.view());
    let mut y_segments = frames::segments(y_bands_t.view());

    standard::from_segments(x_segments.view_mut(), y_segments.view_mut())
}

/// Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
/// # Arguments
/// * `x` - Clean speech signal
/// * `y` - Processed speech signal
/// * `fs_sig` - Sampling frequency of the signals
/// * `extended` - Whether to use the extended STOI measure
pub fn stoi(x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>, fs_sig: u32, extended: bool) -> f64 {
    assert!(
        x.shape() == y.shape(),
        "Input signals must have the same shape"
    );

    if fs_sig != FS {
        let x = resample::resample(x, fs_sig, FS);
        let y = resample::resample(y, fs_sig, FS);

        compute(x.view(), y.view())
    } else {
        compute(x, y)
    }
}
