//! Rust STOI implementation

use std::f64::consts::PI;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

/// Compute the Hann window of length `n`.
fn hann_window(n: usize) -> Array1<f64> {
    let mut w = Array1::zeros(n);
    for i in 0..n {
        w[i] = 0.5 - 0.5 * ((2.0 * PI * i as f64) / (n as f64 - 1.0)).cos();
    }
    w
}

/// Compute the L2 norm of a frame.
fn norm_l2(frame: ArrayView1<'_, f64>) -> f64 {
    frame.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

pub fn remove_silent_frames(
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    dynamic_range: f64,
    frame_length: usize,
    hop_length: usize,
) -> (Array1<f64>, Array1<f64>) {
    // 1. Prepare Hann window
    let hann = hann_window(frame_length + 2)
        .slice(s![1..frame_length + 1])
        .to_owned();

    // 2. Compute frames and energies
    let n = 1 + (x.len() - frame_length - 1) / hop_length;
    let mut x_frames = Array2::<f64>::zeros((n, frame_length));
    let mut y_frames = Array2::<f64>::zeros((n, frame_length));
    let mut energies = Array1::<f64>::zeros(n);

    for (i, start) in (0..x.len() - frame_length).step_by(hop_length).enumerate() {
        // Compute the energy for the current x frame
        let end = start + frame_length;
        let x_frame = x.slice(s![start..end]).to_owned() * &hann;
        let y_frame = y.slice(s![start..end]).to_owned() * &hann;
        x_frames.row_mut(i).assign(&x_frame);
        y_frames.row_mut(i).assign(&y_frame);

        // Compute frame energy
        energies[i] = 20.0 * (norm_l2(x_frame.view()) + f64::EPSILON).log10();
    }

    // 3. Filter frames by energies
    let threshold = energies.max_skipnan() - dynamic_range;
    let mask = energies.mapv(|e| e >= threshold);
    let n = mask.iter().filter(|&&m| m).count();

    // 4. Collect non-silent frames directly into new arrays
    let mut x = Array1::<f64>::zeros((n - 1) * hop_length + frame_length);
    let mut y = Array1::<f64>::zeros((n - 1) * hop_length + frame_length);

    let mut idx = 0;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            let mut x_slice = x.slice_mut(s![idx..idx + frame_length]);
            let mut y_slice = y.slice_mut(s![idx..idx + frame_length]);
            x_slice += &x_frames.row(i);
            y_slice += &y_frames.row(i);

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
    assert!(
        x.shape() == y.shape(),
        "Input signals must have the same shape"
    );
}
