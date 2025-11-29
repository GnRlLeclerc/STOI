//! Rust STOI implementation

use std::f32::consts::PI;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

/// Compute the Hann window of length `n`.
fn hann_window(n: usize) -> Array1<f32> {
    let mut w = Array1::zeros(n);
    for i in 0..n {
        w[i] = 0.5 - 0.5 * ((2.0 * PI * i as f32) / (n as f32 - 1.0)).cos();
    }
    w
}

/// Compute the L2 norm of a frame.
fn norm_l2(frame: ArrayView1<'_, f32>) -> f32 {
    frame.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn remove_silent_frames(
    x: ArrayView1<'_, f32>,
    y: ArrayView1<'_, f32>,
    dynamic_range: f32,
    frame_length: usize,
    hop_length: usize,
) -> (Array1<f32>, Array1<f32>) {
    // 1. Prepare Hann window
    let hann = hann_window(frame_length);

    // 2. Compute frames and energies
    let n = x.len() / hop_length;
    let mut x_frames = Array2::<f32>::zeros((n, frame_length));
    let mut y_frames = Array2::<f32>::zeros((n, frame_length));
    let mut energies = Array1::<f32>::zeros(n);

    for (i, start) in (0..x.len() - frame_length).step_by(hop_length).enumerate() {
        // Compute the energy for the current x frame
        let end = start + frame_length;
        let x_frame = x.slice(s![start..end]).to_owned() * &hann;
        let y_frame = y.slice(s![start..end]).to_owned() * &hann;
        x_frames.row_mut(i).assign(&x_frame);
        y_frames.row_mut(i).assign(&y_frame);

        // Compute frame energy
        energies[i] = 20.0 * (norm_l2(x_frame.view()) + f32::EPSILON).log10();
    }

    // 3. Filter frames by energies
    let threshold = energies.max_skipnan() - dynamic_range;
    let mask = energies.mapv(|e| e >= threshold);
    let n = mask.iter().filter(|&&m| m).count();

    // 4. Collect non-silent frames directly into new arrays
    let mut x = Array1::<f32>::zeros(n * hop_length + frame_length);
    let mut y = Array1::<f32>::zeros(n * hop_length + frame_length);

    let mut idx = 0;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            x.slice_mut(s![idx..idx + frame_length])
                .assign(&x_frames.row(i));
            y.slice_mut(s![idx..idx + frame_length])
                .assign(&y_frames.row(i));
            idx += hop_length;
        }
    }

    (x, y)
}

/// Compute the Short-Time Objective Intelligibility (STOI) measure between two signals.
/// # Arguments
/// * `x` - Clean speech signal
/// * `y` - Processed speech signal
/// * `fs_sig` - Sampling frequency of the signals
/// * `extended` - Whether to use the extended STOI measure
pub fn stoi(x: ArrayView1<'_, f32>, y: ArrayView1<'_, f32>, fs_sig: u32, extended: bool) -> f32 {
    unimplemented!("stoi function is not yet implemented");
}
