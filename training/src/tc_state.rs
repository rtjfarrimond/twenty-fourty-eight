//! TC (Temporal Coherence) learning state.
//!
//! Holds per-weight error accumulators that drive the adaptive learning rate
//! in TC learning (Jaśkowski 2016). Separate from NTupleNetwork because TC
//! state is training-only — it is not needed for inference and is not saved
//! with the model.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::memory::hint_huge_pages;

/// Per-weight error accumulators for TC learning.
///
/// Each weight in the n-tuple network has a corresponding pair:
/// - `error_sums[i]` (E_i): signed running sum of TD errors
/// - `abs_error_sums[i]` (A_i): running sum of |TD errors|
///
/// The adaptive learning rate for weight i is α_i = |E_i| / A_i.
/// When updates consistently agree in sign, α_i stays near 1.0 (keep learning).
/// When updates oscillate, α_i drops toward 0.0 (weight has converged).
///
/// Uses `AtomicU32` with `Relaxed` ordering for Hogwild compatibility —
/// same racy load-modify-store pattern as the weight table itself.
pub struct TcState {
    error_sums: Vec<AtomicU32>,
    abs_error_sums: Vec<AtomicU32>,
}

impl TcState {
    /// Creates zero-initialized TC state for the given number of weights.
    pub fn new(num_weights: usize) -> Self {
        let zero_bits = 0.0f32.to_bits();
        let error_sums: Vec<AtomicU32> = (0..num_weights)
            .map(|_| AtomicU32::new(zero_bits))
            .collect();
        let abs_error_sums: Vec<AtomicU32> = (0..num_weights)
            .map(|_| AtomicU32::new(zero_bits))
            .collect();

        hint_huge_pages(
            error_sums.as_ptr() as *const u8,
            std::mem::size_of_val(&error_sums[..]),
        );
        hint_huge_pages(
            abs_error_sums.as_ptr() as *const u8,
            std::mem::size_of_val(&abs_error_sums[..]),
        );

        Self {
            error_sums,
            abs_error_sums,
        }
    }

    /// Returns the adaptive learning rate for weight at `offset`.
    ///
    /// α_i = |E_i| / A_i, or 1.0 if the weight has never been updated (A_i = 0).
    /// By the triangle inequality, |E_i| ≤ A_i, so α_i ∈ [0.0, 1.0].
    #[inline(always)]
    pub fn adaptive_rate(&self, offset: usize) -> f32 {
        let abs_error = f32::from_bits(
            self.abs_error_sums[offset].load(Ordering::Relaxed),
        );
        if abs_error == 0.0 {
            return 1.0;
        }
        let error = f32::from_bits(
            self.error_sums[offset].load(Ordering::Relaxed),
        );
        error.abs() / abs_error
    }

    /// Accumulates a TD error into the signed and absolute accumulators.
    ///
    /// Skips non-finite td_errors to prevent NaN poisoning the accumulators
    /// under hogwild (where racy weight divergence can produce Inf TD errors).
    #[inline(always)]
    pub fn accumulate(&self, offset: usize, td_error: f32) {
        if !td_error.is_finite() {
            return;
        }
        let error_slot = &self.error_sums[offset];
        let old_error = f32::from_bits(error_slot.load(Ordering::Relaxed));
        let new_error = old_error + td_error;
        if new_error.is_finite() {
            error_slot.store(new_error.to_bits(), Ordering::Relaxed);
        }

        let abs_slot = &self.abs_error_sums[offset];
        let old_abs = f32::from_bits(abs_slot.load(Ordering::Relaxed));
        let new_abs = old_abs + td_error.abs();
        if new_abs.is_finite() {
            abs_slot.store(new_abs.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn num_weights(&self) -> usize {
        self.error_sums.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_correct_length() {
        let state = TcState::new(100);
        assert_eq!(state.num_weights(), 100);
    }

    #[test]
    fn adaptive_rate_defaults_to_one_for_unseen_weight() {
        let state = TcState::new(10);
        assert_eq!(state.adaptive_rate(0), 1.0);
        assert_eq!(state.adaptive_rate(9), 1.0);
    }

    #[test]
    fn consistent_positive_errors_keep_rate_near_one() {
        let state = TcState::new(1);
        state.accumulate(0, 5.0);
        state.accumulate(0, 3.0);
        state.accumulate(0, 7.0);
        // E = 15.0, A = 15.0 → α = 1.0
        assert!((state.adaptive_rate(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cancelling_errors_drive_rate_toward_zero() {
        let state = TcState::new(1);
        state.accumulate(0, 10.0);
        state.accumulate(0, -10.0);
        // E = 0.0, A = 20.0 → α = 0.0
        assert!((state.adaptive_rate(0)).abs() < 1e-6);
    }

    #[test]
    fn partially_cancelling_errors_produce_intermediate_rate() {
        let state = TcState::new(1);
        state.accumulate(0, 10.0);
        state.accumulate(0, -4.0);
        // E = 6.0, A = 14.0 → α = 6/14 ≈ 0.4286
        let rate = state.adaptive_rate(0);
        assert!((rate - 6.0 / 14.0).abs() < 1e-5);
    }

    #[test]
    fn accumulate_handles_negative_errors() {
        let state = TcState::new(1);
        state.accumulate(0, -3.0);
        state.accumulate(0, -7.0);
        // E = -10.0, A = 10.0 → α = |-10| / 10 = 1.0
        assert!((state.adaptive_rate(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn different_offsets_are_independent() {
        let state = TcState::new(3);
        state.accumulate(0, 10.0);
        state.accumulate(1, 10.0);
        state.accumulate(1, -10.0);
        // Offset 0: E=10, A=10 → α=1.0
        // Offset 1: E=0, A=20 → α=0.0
        // Offset 2: unseen → α=1.0
        assert!((state.adaptive_rate(0) - 1.0).abs() < 1e-6);
        assert!(state.adaptive_rate(1).abs() < 1e-6);
        assert_eq!(state.adaptive_rate(2), 1.0);
    }
}
