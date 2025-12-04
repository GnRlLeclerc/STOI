//! Standard STOI computation from octave segment spectrograms

use ndarray::{Zip, prelude::*};

use crate::constants::{BETA, NUM_BANDS};

/// Compute the standard STOI from octave segment spectrograms of the clean and processed signals.
/// The segments have shapes (num_segments, num_bands, segment_length).
pub fn from_segments(
    mut x_segments: ArrayViewMut3<f64>,
    mut y_segments: ArrayViewMut3<f64>,
) -> f64 {
    let clip_value = 10.0_f64.powf(-BETA / 20.0);
    let num_segments = x_segments.shape()[0];

    // Perform the per-segment processing
    let similarity = Zip::from(x_segments.outer_iter_mut())
        .and(y_segments.outer_iter_mut())
        .fold(0.0, |acc, mut x_bands, mut y_bands| {
            acc + Zip::from(x_bands.outer_iter_mut())
                .and(y_bands.outer_iter_mut())
                .fold(0.0, |acc, mut x_segment, mut y_segment| {
                    // Normalize y so that it has the same norm as x
                    let x_norm = x_segment.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let y_norm = y_segment.iter().map(|v| v * v).sum::<f64>().sqrt();

                    y_segment *= x_norm / (y_norm + f64::EPSILON);

                    // Clip y
                    x_segment
                        .iter()
                        .zip(y_segment.iter_mut())
                        .for_each(|(x, y)| {
                            *y = y.min(x * (1.0 + clip_value));
                        });

                    // Substract means
                    x_segment -= x_segment.mean().unwrap();
                    y_segment -= y_segment.mean().unwrap();

                    // Divide by norms
                    let x_norm = x_segment.iter().map(|v| v * v).sum::<f64>().sqrt() + f64::EPSILON;
                    let y_norm = y_segment.iter().map(|v| v * v).sum::<f64>().sqrt() + f64::EPSILON;
                    x_segment /= x_norm;
                    y_segment /= y_norm;

                    // Compute similarity
                    acc + &x_segment.dot(&y_segment)
                })
        });

    similarity / (num_segments * NUM_BANDS) as f64
}
