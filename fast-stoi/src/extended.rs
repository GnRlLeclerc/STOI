//! Extended STOI computation from octave segment spectrograms

use std::f32::EPSILON;

use faer::prelude::*;

use crate::constants::{NUM_BANDS, SEGMENT_LENGTH};

/// Compute the extended STOI from octave segment spectrograms of the clean and processed signals.
/// The segments have shapes (segment_length, num_segments * num_bands).
pub fn from_segments(mut x_segments: MatMut<f32>, mut y_segments: MatMut<f32>) -> f32 {
    row_col_normalize(x_segments.as_mut());
    row_col_normalize(y_segments.as_mut());

    let n_segments = x_segments.ncols();

    let dotted: f32 = x_segments
        .col_iter()
        .zip(y_segments.col_iter())
        .map(|(x_col, y_col)| x_col.transpose() * y_col)
        .sum();

    dotted / (n_segments as f32) * (NUM_BANDS as f32 / SEGMENT_LENGTH as f32)
}

/// Normalize segments both along columns, and along rows by band subgroups.
fn row_col_normalize(mut mat: MatMut<f32>) {
    normalize_cols(mat.as_mut());

    // Group segments by bands
    let band_segments = mat.ncols() / NUM_BANDS;
    for i in 0..band_segments {
        let mut submat = mat.as_mut().subcols_mut(i * NUM_BANDS, NUM_BANDS);
        normalize_rows(submat.as_mut());
    }
}

/// Normalize a 2D matrix along columns.
fn normalize_cols(mat: MatMut<f32>) {
    // Subtract mean and divide by norm l2
    mat.col_iter_mut().for_each(|mut col| {
        let mean = col.sum() / (col.nrows() as f32);
        col.as_mut().iter_mut().for_each(|x| {
            *x -= mean;
        });

        let norm2 = col.norm_l2() + EPSILON;
        col.iter_mut().for_each(|x| {
            *x /= norm2;
        });
    });
}

/// Normalize a 2D matrix along rows.
fn normalize_rows(mat: MatMut<f32>) {
    // Subtract mean and divide by norm l2
    mat.row_iter_mut().for_each(|mut row| {
        let mean = row.sum() / (row.ncols() as f32);
        row.as_mut().iter_mut().for_each(|x| {
            *x -= mean;
        });
        let norm2 = row.norm_l2() + EPSILON;
        row.iter_mut().for_each(|x| {
            *x /= norm2;
        });
    });
}
