//! Slice, filter and preprocess audio frames.

use lazy_static::lazy_static;
use ndarray::{Zip, prelude::*};
use ndarray_stats::QuantileExt;
use windowfunctions::{Symmetry, WindowFunction, window};

use crate::constants::{DYNAMIC_RANGE, FRAME_LENGTH, HOP_LENGTH, SEGMENT_LENGTH};

struct FrameWindows {
    /// Trimmed hann window
    pub hann: Array1<f64>,
    /// Hann window with half overlap with another hann window at the end
    pub hann_start: Array1<f64>,
    /// Hann window with overlapping hann windows added at both ends
    pub hann_center: Array1<f64>,
    // NOTE: we don't need hann end, that frame is discarded
}

impl FrameWindows {
    fn new() -> Self {
        let hann = window(FRAME_LENGTH + 2, WindowFunction::Hann, Symmetry::Symmetric)
            .skip(1)
            .take(FRAME_LENGTH)
            .collect::<Array1<f64>>();

        // 1. Combine hann windows to mimic slicing + overlap-adding
        let mut hann_start = hann.clone();
        let mut slice = hann_start.slice_mut(s![(FRAME_LENGTH / 2)..]);
        slice += &hann.slice(s![..(FRAME_LENGTH / 2)]);
        // 2. Apply hann again to account for the reslicing just before rfft
        hann_start *= &hann;

        // 1. Combine hann windows to mimic slicing + overlap-adding
        let mut hann_center = hann.clone();
        let mut slice = hann_center.slice_mut(s![..(FRAME_LENGTH / 2)]);
        slice += &hann.slice(s![(FRAME_LENGTH / 2)..]);
        let mut slice = hann_center.slice_mut(s![(FRAME_LENGTH / 2)..]);
        slice += &hann.slice(s![..(FRAME_LENGTH / 2)]);
        // 2. Apply hann again to account for the reslicing just before rfft
        hann_center *= &hann;

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
/// Returns 2D arrays containing the frames along with
/// a boolean mask and the total amount of valid frames.
///
/// Performance notes:
/// Energy-based filtering is performed once all energies have been computed.
/// For this reason, we cannot know beforehand which frames are to be discarded,
/// hence why we store all frames in an intermediate 2D array.
/// In order to avoid reallocations, we return the unfiltered 2D array along
/// with a boolean mask indicating which frames to keep.
pub fn process_frames(
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) -> (Array2<f64>, Array2<f64>, Array1<bool>, usize) {
    // 1. Compute frames and energies
    let n = 1 + (x.len() - FRAME_LENGTH - 1) / HOP_LENGTH;
    let mut x_frames = Array2::<f64>::zeros((n, FRAME_LENGTH));
    let mut y_frames = Array2::<f64>::zeros((n, FRAME_LENGTH));
    let mut energies = Array1::<f64>::zeros(n);

    for (i, start) in (0..x.len() - FRAME_LENGTH).step_by(HOP_LENGTH).enumerate() {
        // Compute the energy for the current x frame
        let end = start + FRAME_LENGTH;

        let mut x_frame = x_frames.row_mut(i);
        let mut y_frame = y_frames.row_mut(i);

        // Copy frames
        x_frame.assign(&x.slice(s![start..end]));
        y_frame.assign(&y.slice(s![start..end]));

        // Compute the frame norm after applying hann window
        // Note that we do not apply hann window to the frame in place,
        // because due to the original stoi implementation
        // 1. applying hann
        // 2. rebuilding the signal by overlap-adding the frames
        // 3. slicing and applying hann again
        // the resulting window that is effectively applied to each frame
        // is a little different.
        let frame_norm = Zip::from(x_frame.view())
            .and(&FRAME_WINDOWS.hann)
            .fold(0.0, |acc, &x, &w| acc + (x * w).powi(2))
            .sqrt();

        // Compute frame energy
        energies[i] = 20.0 * (frame_norm + f64::EPSILON).log10();
    }

    // 2. Compute frame mask based on energies
    let threshold = energies.max_skipnan() - DYNAMIC_RANGE;
    let mut count = 0;
    let mut mask = energies.mapv(|e| {
        let valid = e >= threshold;
        count += valid as usize;
        valid
    });

    // 3. Discard the last valid frame as the original implementation does (bad slicing)
    // and then apply the combined hann window to each valid frame to mimic the result
    // from slicing, overlap-adding and slicing again.
    let mut index = 0;
    Zip::from(x_frames.rows_mut())
        .and(y_frames.rows_mut())
        .and(mask.view_mut())
        .for_each(|mut x_frame, mut y_frame, valid| {
            if !*valid {
                return;
            }

            // First valid frame: apply hann_start
            if index == 0 {
                x_frame *= &FRAME_WINDOWS.hann_start;
                y_frame *= &FRAME_WINDOWS.hann_start;
            }
            // Last valid frame: discard it
            else if index == count - 1 {
                *valid = false;
            } else {
                // Center frames: apply hann_center
                x_frame *= &FRAME_WINDOWS.hann_center;
                y_frame *= &FRAME_WINDOWS.hann_center;
            }

            index += 1;
        });

    count -= 1; // account for the discarded last frame

    (x_frames, y_frames, mask, count)
}

/// Slice octave band spectrogram into overlapping segments
/// Shapes: (bands, frames) -> (segments, bands, N)
///
/// We copy the segments into a new array because we need to perform per-segment
/// mutating operations later.
pub fn segments(x_bands: ArrayView2<'_, f64>) -> Array3<f64> {
    let n_bands = x_bands.shape()[0];
    let n_frames = x_bands.shape()[1];
    let n_segments = n_frames.saturating_sub(SEGMENT_LENGTH) + 1;

    let mut segments = Array3::<f64>::zeros((n_segments, n_bands, SEGMENT_LENGTH));

    for i in 0..n_segments {
        let segment = x_bands.slice(s![.., i..(i + SEGMENT_LENGTH)]);
        let mut target = segments.slice_mut(s![i, .., ..]);
        target.assign(&segment);
    }

    segments
}
