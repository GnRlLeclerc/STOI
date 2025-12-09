//! Sinc poly resampling

use std::f64::consts::PI;

use dashmap::DashMap;
use lazy_static::lazy_static;
use num::integer;
use windowfunctions::{Symmetry, WindowFunction, window};

use crate::upfirdn::upfirdn;

lazy_static! {
    /// Cache filters for different from rates
    static ref WINDOWS: DashMap<usize, Vec<f64>> = DashMap::new();
}

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

/// Generates an apodized Kaiser window collected into a Row.
fn apodized_kaiser_window(f: f64, beta: f64, half_length: usize) -> Vec<f64> {
    let sinc_iter = ideal_sinc(f, half_length);
    let kaiser_iter = kaiser(beta as f32, half_length);

    sinc_iter
        .zip(kaiser_iter)
        .map(|(sinc, kaiser)| sinc * kaiser)
        .collect()
}

/// Generates the different contiguous filter phases for efficient
/// computation. Also returns the filter half-length.
fn generate_filter_phases(up: usize, down: usize) -> Vec<f64> {
    let stopband_cutoff_freq = 1.0 / (2.0 * up.max(down) as f64);
    let roll_off_width = stopband_cutoff_freq / 10.0;

    // Compute the filter
    let filter_half_length = ((REJECTION_DB - 8.0) / (28.714 * roll_off_width)).ceil() as u32;
    let beta = 0.1102 * (REJECTION_DB - 8.7);
    let mut filter =
        apodized_kaiser_window(stopband_cutoff_freq, beta, filter_half_length as usize);
    let sum: f64 = filter.iter().sum();
    filter.iter_mut().for_each(|v| *v /= sum);

    filter
}

/// Polyphase resampling.
///
/// Some information for this doc (reformulate and clean this up later):
/// - zero-phase => the window is symmetric (does not introduce any shift)
/// - FIR filter => finite impulse response. Basically, the window is of finite length.
/// - low-pass => when upsampling by inserting zeros, if we upsample *n, we create
///   high frequency signals. The window must smooth this out and remove these high frequencies
pub fn resample(x: &[f64], from: usize, to: usize) -> Vec<f64> {
    // Compute upsampling and dowsampling ratios
    let gcd = integer::gcd(from, to);
    let up = to / gcd;
    let down = from / gcd;

    // Get the filters
    // If filters are missing, they are inserted and fetched
    // again to drop the exclusive mutable ref held by entry
    WINDOWS
        .entry(from)
        .or_insert_with(|| generate_filter_phases(up, down));
    let filter = WINDOWS.get(&from).unwrap();

    upfirdn(&filter, x, up, down)
}
