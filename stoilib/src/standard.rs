//! Standard STOI computation from octave segment spectrograms

use std::f64::EPSILON;

use faer::prelude::*;

use crate::constants::{BETA, SEGMENT_LENGTH};

/// Compute the standard STOI from octave segment spectrograms of the clean and processed signals.
/// The segments have shapes (segment_length, num_segments * num_bands).
pub fn from_segments(x_segments: MatMut<f64>, y_segments: MatMut<f64>) -> f64 {
    let clip_value = 10.0_f64.powf(-BETA / 20.0);
    let n = x_segments.ncols();

    let mut similarity = 0.0;

    // Perform the per-segment processing
    x_segments
        .col_iter_mut()
        .zip(y_segments.col_iter_mut())
        .for_each(|(mut x_segment, mut y_segment)| {
            // Normalize y so that it has the same norm as x
            // and then clip y
            let ratio = x_segment.norm_l2() / (y_segment.norm_l2() + EPSILON);
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            zip!(&x_segment, &mut y_segment).for_each(|unzip!(x, y)| {
                *y = (*y * ratio).min(x * (1.0 + clip_value));
                x_sum += x;
                y_sum += *y;
            });

            // Substract means
            let x_mean = x_sum / SEGMENT_LENGTH as f64;
            let y_mean = y_sum / SEGMENT_LENGTH as f64;

            // Subtract mean and start computing resulting norm
            // at the same time
            let mut x_sq_sum = 0.0;
            let mut y_sq_sum = 0.0;
            zip!(&mut x_segment, &mut y_segment).for_each(|unzip!(x, y)| {
                *x -= x_mean;
                *y -= y_mean;
                x_sq_sum += x.powi(2);
                y_sq_sum += y.powi(2);
            });

            let x_norm = x_sq_sum.sqrt() + EPSILON;
            let y_norm = y_sq_sum.sqrt() + EPSILON;

            // Compute pre-normalization similarity
            let mut s = 0.0;
            zip!(&x_segment, &y_segment).for_each(|unzip!(x, y)| {
                s += x * y;
            });

            // Aggregate similarity and apply normalization
            similarity += s / (x_norm * y_norm);
        });

    similarity / n as f64
}
