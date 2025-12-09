//! Upfirdn implementation

use faer::prelude::*;

/// Upfirdn implementation
/// h: window
/// x: input signal
/// up: upsampling factor
/// down: downsampling factor
/// Normalization by up is applied to conserve signal energy
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    // Compute contiguous filter phases
    let phase_length = (h.len() as f32 / up as f32).ceil() as usize;
    let mut phases = vec![0.0; phase_length * up];
    for phase in 0..up {
        for n in 0..phase_length {
            phases[phase * phase_length + n] = h[n * up + phase];
        }
    }

    // Pad the input signal with zeros to avoid bound checks during filtering
    let padding = h.len() / (2 * up); // Padding at both ends
    let mut padded_x = vec![0.0; x.len() + 2 * padding];
    padded_x[padding..padding + x.len()].copy_from_slice(x);

    // Create output vector
    let mut target = vec![0.0; x.len() * up / down];

    // Prepare iteration indices
    let mut phase: usize = (h.len() / 2) % up;
    let phase_step = down % up; // Phase step within 0..up
    let x_step = down / up; // Base input step
    let mut x_start: usize = 0; // Padding ensures it starts at 0

    // Iterate over target samples
    for y in target.iter_mut() {
        let p = phase * phase_length;

        *y = RowRef::<f64>::from_slice(&phases[p..p + phase_length])
            * ColRef::<f64>::from_slice(&padded_x[x_start..x_start + phase_length])
            * up as f64;

        // Update phase and input start index
        x_start += x_step;
        if phase >= phase_step {
            phase -= phase_step;
        } else {
            phase += up - phase_step;
            x_start += 1; // Carry over
        }
    }

    target
}
