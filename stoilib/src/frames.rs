//! Slice, filter and preprocess audio frames.

use faer::prelude::*;
use lazy_static::lazy_static;
use windowfunctions::{Symmetry, WindowFunction, window};

use crate::constants::{DYNAMIC_RANGE, FRAME_LENGTH, HALF_FRAME, HOP_LENGTH, SEGMENT_LENGTH};

struct FrameWindows {
    /// Trimmed hann window
    pub hann: Col<f64>,
    /// Hann window with half overlap with another hann window at the end
    pub hann_start: Col<f64>,
    /// Hann window with overlapping hann windows added at both ends
    pub hann_center: Col<f64>,
    // NOTE: we don't need hann end, that frame is discarded
}

impl FrameWindows {
    fn new() -> Self {
        let hann = window(FRAME_LENGTH + 2, WindowFunction::Hann, Symmetry::Symmetric)
            .skip(1)
            .take(FRAME_LENGTH)
            .collect::<Col<f64>>();

        // 1. Combine hann windows to mimic slicing + overlap-adding
        let mut hann_start = hann.clone();
        let mut slice = hann_start.subrows_mut(HALF_FRAME, HALF_FRAME);
        slice += &hann.subrows(0, HALF_FRAME);
        // 2. Apply hann again to account for the reslicing just before rfft
        zip!(&mut hann_start, &hann).for_each(|unzip!(w1, &w2)| *w1 *= w2);

        // 1. Combine hann windows to mimic slicing + overlap-adding
        let mut hann_center = hann.clone();
        let mut slice = hann_center.subrows_mut(0, HALF_FRAME);
        slice += &hann.subrows(HALF_FRAME, HALF_FRAME);
        let mut slice = hann_center.subrows_mut(HALF_FRAME, HALF_FRAME);
        slice += &hann.subrows(0, HALF_FRAME);
        // 2. Apply hann again to account for the reslicing just before rfft
        zip!(&mut hann_center, &hann).for_each(|unzip!(w1, &w2)| *w1 *= w2);

        Self {
            hann,
            hann_start,
            hann_center,
        }
    }
}

lazy_static! {
    static ref FRAME_WINDOWS: FrameWindows = FrameWindows::new();
}

/// Slice 2 input signals into overlapping frames and
/// applies a hann window to each frame.
/// The frames are then filtered based on their energy.
///
/// Returns 2D arrays containing the frames with shape
/// (frame_length, n_frames) along with a boolean mask
/// and the total amount of valid frames.
///
/// Performance notes:
/// Energy-based filtering is performed once all energies have been computed.
/// For this reason, we cannot know beforehand which frames are to be discarded,
/// hence why we store all frames in an intermediate 2D array.
/// In order to avoid reallocations, we return the unfiltered 2D array along
/// with a boolean mask indicating which frames to keep.
pub fn process_frames(x: &[f64], y: &[f64]) -> (Mat<f64>, Mat<f64>, Col<bool>, usize) {
    // 1. Compute frames and energies
    let n = 1 + (x.len() - FRAME_LENGTH - 1) / HOP_LENGTH;
    let mut x_frames = Mat::<f64>::zeros(FRAME_LENGTH, n);
    let mut y_frames = Mat::<f64>::zeros(FRAME_LENGTH, n);
    let mut energies = Col::<f64>::zeros(n);

    for (i, start) in (0..x.len() - FRAME_LENGTH).step_by(HOP_LENGTH).enumerate() {
        // Compute the energy for the current x frame
        let end = start + FRAME_LENGTH;

        let mut x_frame = x_frames.col_mut(i);
        let mut y_frame = y_frames.col_mut(i);

        // Copy frames
        x_frame.copy_from(ColRef::from_slice(&x[start..end]));
        y_frame.copy_from(ColRef::from_slice(&y[start..end]));

        // Compute the frame norm after applying hann window
        // Note that we do not apply hann window to the frame in place,
        // because due to the original stoi implementation
        // 1. applying hann
        // 2. rebuilding the signal by overlap-adding the frames
        // 3. slicing and applying hann again
        // the resulting window that is effectively applied to each frame
        // is a little different.
        let frame_norm = zip!(&x_frame, &FRAME_WINDOWS.hann)
            .map(|unzip!(x, w)| x * w)
            .sum()
            .sqrt();

        // Compute frame energy
        energies[i] = 20.0 * (frame_norm + f64::EPSILON).log10();
    }

    // 2. Compute frame mask based on energies
    let threshold = energies.max().unwrap() - DYNAMIC_RANGE;
    let mut count = 0;
    let mut mask = energies
        .iter()
        .map(|&e| {
            let valid = e >= threshold;
            count += valid as usize;
            valid
        })
        .collect::<Col<_>>();

    // 3. Discard the last valid frame as the original implementation does (bad slicing)
    // and then apply the combined hann window to each valid frame to mimic the result
    // from slicing, overlap-adding and slicing again.
    let mut index = 0;
    x_frames
        .col_iter_mut()
        .zip(y_frames.col_iter_mut())
        .zip(mask.iter_mut())
        .for_each(|((mut x_frame, mut y_frame), valid)| {
            if !*valid {
                return;
            }

            // First valid frame: apply hann_start
            if index == 0 {
                zip!(&mut x_frame, &FRAME_WINDOWS.hann_start).for_each(|unzip!(w1, &w2)| *w1 *= w2);
                zip!(&mut y_frame, &FRAME_WINDOWS.hann_start).for_each(|unzip!(w1, &w2)| *w1 *= w2);
            }
            // Last valid frame: discard it
            else if index == count - 1 {
                *valid = false;
            } else {
                // Center frames: apply hann center
                zip!(&mut x_frame, &FRAME_WINDOWS.hann_center)
                    .for_each(|unzip!(w1, &w2)| *w1 *= w2);
                zip!(&mut y_frame, &FRAME_WINDOWS.hann_center)
                    .for_each(|unzip!(w1, &w2)| *w1 *= w2);
            }

            index += 1;
        });

    count -= 1; // account for the discarded last frame

    (x_frames, y_frames, mask, count)
}

/// Slice octave band spectrogram into overlapping segments
/// Shapes: (frames, bands) -> (N, n_segments * bands)
///
/// We copy the segments into a new array because we need to perform per-segment
/// mutating operations later.
/// Because x and y will be compared in a per-segment basis, we merge the
/// n_segments and bands dimensions for efficient storage and iteration.
pub fn segments(x_bands: MatRef<f64>) -> Mat<f64> {
    let n_bands = x_bands.ncols();
    let n_frames = x_bands.nrows();
    let n_segments = n_frames.saturating_sub(SEGMENT_LENGTH) + 1;

    let mut segments = Mat::<f64>::zeros(SEGMENT_LENGTH, n_segments * n_bands);

    for i in 0..n_segments {
        let mut segments_slice = segments.subcols_mut(i * n_bands, n_bands);
        let bands_slice = x_bands.submatrix(i, 0, SEGMENT_LENGTH, n_bands);
        segments_slice.copy_from(bands_slice);
    }

    segments
}
