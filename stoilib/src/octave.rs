//! Third octave bands

use faer::prelude::*;

use crate::constants::NUM_BANDS;

/// Octave band indices in FFT spectrums of length 512
/// (precomputed from the original STOI implementation)
const OCTAVE_BANDS: [(usize, usize); NUM_BANDS] = [
    (7, 9),
    (9, 11),
    (11, 14),
    (14, 17),
    (17, 22),
    (22, 27),
    (27, 34),
    (34, 43),
    (43, 55),
    (55, 69),
    (69, 87),
    (87, 109),
    (109, 138),
    (138, 174),
    (174, 219),
];

/// Merge FFT spectrogram into octave bands specified by the index ranges in `OCTAVE_BANDS`.
/// Input spectrograms have shape (FFT_BINS, num_frames).
/// The merged output has shape (NUM_BANDS, num_frames).
pub fn compute_octave_bands(spectrogram: MatRef<f64>) -> Mat<f64> {
    let num_frames = spectrogram.ncols();
    let mut band_spectrogram = Mat::<f64>::zeros(NUM_BANDS, num_frames);

    // Iterate over each frame
    spectrogram
        .col_iter()
        .zip(band_spectrogram.col_iter_mut())
        .for_each(|(rfft, bands)| {
            bands
                .iter_mut()
                .zip(OCTAVE_BANDS.iter())
                .for_each(|(band, &(start, end))| {
                    // The spectrogram contains squared magnitudes,
                    // so we just need to sum and sqrt instead of norm_l2
                    *band = rfft.subrows(start, end - start).sum().sqrt();
                });
        });

    band_spectrogram
}
