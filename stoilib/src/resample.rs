//! Sinc poly resampling

use std::f64::consts::PI;

use ndarray::prelude::*;
use num::integer;
use windowfunctions::{Symmetry, WindowFunction, window};

const REJECTION_DB: f64 = 60.0;

/// Generate an ideal sinc low-pass filter with normalized cutoff frequency f.
/// Returns an iterator over the filter coefficients to avoid allocation.
fn ideal_sinc(f: f64, half_length: usize) -> impl Iterator<Item = f64> {
    (-(half_length as isize)..half_length as isize + 1).map(move |n| {
        if n == 0 {
            2.0 * f
        } else {
            (2.0 * PI * f * n as f64).sin() / (PI * n as f64)
        }
    })
}

/// Generates a Kaiser window with given beta and length.
/// Returns an iterator over the window coefficients to avoid allocation.
fn kaiser(beta: f32, half_length: usize) -> impl Iterator<Item = f64> {
    window(
        2 * half_length + 1,
        WindowFunction::Kaiser { beta },
        Symmetry::Symmetric,
    )
}

/// Generates an apodized Kaiser window collected into an Array1.
/// The original STOI implementation scales the window by the upsampling factor.
fn apodized_kaiser_window(f: f64, beta: f64, half_length: usize, factor: f64) -> Array1<f64> {
    let sinc_iter = ideal_sinc(f, half_length);
    let kaiser_iter = kaiser(beta as f32, half_length);

    Array1::from_iter(
        sinc_iter
            .zip(kaiser_iter)
            .map(|(sinc, kaiser)| factor * sinc * kaiser),
    )
}

/// Polyphase resampling.
///
/// Some information for this doc (reformulate and clean this up later):
/// - zero-phase => the window is symmetric (does not introduce any shift)
/// - FIR filter => finite impulse response. Basically, the window is of finite length.
/// - low-pass => when upsampling by inserting zeros, if we upsample *n, we create
///   high frequency signals. The window must smooth this out and remove these high frequencies
pub fn resample(x: ArrayView1<'_, f64>, from: u32, to: u32) -> Array1<f64> {
    // Compute upsampling and dowsampling ratios
    let gcd = integer::gcd(from, to);
    let up = to / gcd;
    let down = from / gcd;

    let stopband_cutoff_freq = 1.0 / (2.0 * up.max(down) as f64);
    let roll_off_width = stopband_cutoff_freq / 10.0;

    // Compute the filter
    let filter_half_length = ((REJECTION_DB - 8.0) / (28.714 * roll_off_width)).ceil() as u32;
    let beta = 0.1102 * (REJECTION_DB - 8.7);
    let mut filter = apodized_kaiser_window(
        stopband_cutoff_freq,
        beta,
        filter_half_length as usize,
        up as f64,
    );
    filter /= filter.sum();

    // Create target array
    let target_len = x.len() as u32 * up / down;
    let mut target = Array1::<f64>::zeros(target_len as usize);

    // Compute polyphase components
    let polyphases: Vec<_> = (0..up)
        .map(|offset| filter.slice(s![offset as isize..;up as isize]))
        .collect();

    for i in 0..target_len {
        // Compute the indices of the target value and filter boundaries
        // in the virtually upsampled (by `up`) signal
        let virt_idx = i * down;
        let virt_start = virt_idx.saturating_sub(filter_half_length);
        let virt_end = virt_idx.saturating_add(filter_half_length); // inclusive

        // Compute indices in the original signal
        let x_start = (virt_start as f64 / up as f64).ceil() as usize;
        let x_end = (virt_end as f64 / up as f64)
            .floor()
            .min(x.len() as f64 - 1.0) as usize; // inclusive

        // Compute indices in the filter
        let filter_start = x_start * up as usize - virt_start as usize;
        let filter_end = x_end * up as usize - virt_start as usize; // inclusive

        let filter_slice = filter.slice(s![filter_start..filter_end+1;up]);
        let x_slice = x.slice(s![x_start..x_end + 1]);

        target[i as usize] = filter_slice.dot(&x_slice);
    }

    target
}
