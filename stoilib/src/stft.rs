//! STFT computation on frames

use std::sync::Arc;

use faer::prelude::*;
use lazy_static::lazy_static;
use num::complex::ComplexFloat;
use realfft::{RealFftPlanner, RealToComplex};

use crate::constants::{FFT_BINS, FFT_LENGTH};

lazy_static! {
    static ref R2C: Arc<dyn RealToComplex<f64>> =
        RealFftPlanner::<f64>::new().plan_fft_forward(FFT_LENGTH);
}

/// Compute the RFFT of each valid frame as indicated by the mask.
/// Input frames have shape (frame_length, n_frames).
/// Returns a real valued squared magnitude spectrogram
/// of shape (FFT_BINS, frames).
pub fn compute_frame_rffts(frames: MatRef<f64>, mask: ColRef<bool>, count: usize) -> Mat<f64> {
    // Create buffers
    let mut scratch_buffer = R2C.make_scratch_vec();
    let mut input_buffer = R2C.make_input_vec();
    let mut output_buffer = R2C.make_output_vec();

    // Create output array as row-major for faster writes
    let mut spectrogram = Mat::<f64>::zeros(FFT_BINS, count);
    let mut index = 0; // destination row index (skips invalid frames)

    // Iterate over valid frames and compute their RFFT
    frames
        .col_iter()
        .zip(mask.iter())
        .for_each(|(frame, &valid)| {
            if !valid {
                return;
            }

            // Copy frame into input buffer with zero padding
            input_buffer[..frame.nrows()]
                .copy_from_slice(frame.try_as_col_major().unwrap().as_slice());

            // Perform RFFT
            R2C.process_with_scratch(&mut input_buffer, &mut output_buffer, &mut scratch_buffer)
                .unwrap();

            // Copy magnitude spectrum to output spectrogram
            spectrogram
                .col_mut(index)
                .iter_mut()
                .zip(&output_buffer)
                .for_each(|(real, complex)| {
                    *real = complex.re().powi(2) + complex.im().powi(2);
                });

            index += 1;
        });

    spectrogram
}
