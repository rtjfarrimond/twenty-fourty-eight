use std::sync::atomic::{AtomicU32, Ordering};

use game_engine::Board;

use crate::memory::hint_huge_pages;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_pext_u64;

/// The number of possible tile values (0=empty, 1=2, ..., 15=32768).
const TILE_VALUES: usize = 16;

/// Extracts scattered bits from `source` selected by `mask` into a
/// contiguous result. Uses BMI2 PEXT hardware instruction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn pext_hw(source: u64, mask: u64) -> u64 {
    unsafe { _pext_u64(source, mask) }
}

/// Software fallback for PEXT — extracts nibbles selected by the mask.
/// The mask must select only complete nibbles (each set bit group is 4 bits
/// aligned to a 4-bit boundary).
#[inline(always)]
fn pext_sw(source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    let mut output_shift = 0;
    let mut remaining_mask = mask;
    while remaining_mask != 0 {
        let bit = remaining_mask.trailing_zeros();
        result |= ((source >> bit) & 0xF) << output_shift;
        output_shift += 4;
        remaining_mask &= !(0xF << bit);
    }
    result
}

/// Extracts scattered nibbles from `source` selected by `mask`.
/// Uses hardware PEXT on x86_64, software fallback otherwise.
///
/// Note on AMD Zen 1/2: hardware PEXT is microcoded (~18 cycles) but we
/// still prefer it. A software fallback of 6 shift/mask/or ops with a
/// data-dependent `trailing_zeros` loop measured ~equivalent in practice,
/// and hardware PEXT's latency is hidden behind the much larger (~60+
/// cycle) DRAM loads that follow.
#[inline(always)]
fn pext(source: u64, mask: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        pext_hw(source, mask)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pext_sw(source, mask)
    }
}

/// Builds the PEXT mask for a set of board positions.
/// Each position (flat index 0-15) contributes 4 bits (one nibble) to the mask.
fn build_pext_mask(positions: &[usize]) -> u64 {
    let mut mask: u64 = 0;
    for &position in positions {
        mask |= 0xF_u64 << (position * 4);
    }
    mask
}

/// An n-tuple network using the isomorphic evaluation approach with BMI2 PEXT:
/// - Stores only the base patterns (no symmetry expansion in weights)
/// - At evaluation time, transforms the board 8 ways using fast bitwise ops
/// - Extracts tuple indices with a single PEXT instruction per pattern
///
/// Weights are stored as `AtomicU32` (reinterpreting f32 bits) so that both
/// single-threaded and Hogwild-style parallel training can share the same
/// storage. `Relaxed` ordering compiles to plain MOV on x86 — no overhead
/// versus a plain `Vec<f32>`. Lock-free racy updates are the intended model
/// for Hogwild (see DESIGN_DECISIONS §17, FUTURE.md parallel-training entry).
pub struct NTupleNetwork {
    /// One PEXT mask per base pattern.
    masks: Vec<u64>,
    /// Flat weight storage: masks.len() contiguous weight tables.
    weights: Vec<AtomicU32>,
}

impl NTupleNetwork {
    /// Size of each pattern's weight table: 16 tile values ^ 6 positions.
    /// Hoisted to a constant so `pattern_index * TABLE_SIZE` compiles to a
    /// single `shl rax, 24` rather than a dependent load + `imul`. This
    /// isn't a massive win on its own but it removes one L1 load from the
    /// inner loop that runs 32-64× per move.
    pub const TABLE_SIZE: usize = TILE_VALUES.pow(6);

    pub fn with_symmetry_expansion(
        base_patterns: &[Vec<(usize, usize)>],
        initial_weight: f32,
    ) -> Self {
        let (masks, total_weights) = Self::build_layout(base_patterns);
        let initial_bits = initial_weight.to_bits();
        let weights: Vec<AtomicU32> = (0..total_weights)
            .map(|_| AtomicU32::new(initial_bits))
            .collect();

        hint_huge_pages(
            weights.as_ptr() as *const u8,
            std::mem::size_of_val(&weights[..]),
        );

        Self { masks, weights }
    }

    /// Constructs a network whose weights are independently sampled uniformly
    /// from `[-amplitude, +amplitude]`. Deterministic given `seed`.
    ///
    /// Random init breaks symmetry between weights but does not bias the
    /// agent toward exploration the way optimistic init does — values start
    /// near zero, not near the natural game-score scale. Useful for ensembling
    /// experiments and seed-variance studies.
    pub fn with_random_init(
        base_patterns: &[Vec<(usize, usize)>],
        amplitude: f32,
        seed: u64,
    ) -> Self {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let (masks, total_weights) = Self::build_layout(base_patterns);
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights: Vec<AtomicU32> = (0..total_weights)
            .map(|_| {
                let value: f32 = if amplitude > 0.0 {
                    rng.random_range(-amplitude..=amplitude)
                } else {
                    0.0
                };
                AtomicU32::new(value.to_bits())
            })
            .collect();

        hint_huge_pages(
            weights.as_ptr() as *const u8,
            std::mem::size_of_val(&weights[..]),
        );

        Self { masks, weights }
    }

    /// Computes per-pattern PEXT masks and the total weight count.
    /// Shared by every constructor so they agree on layout.
    fn build_layout(base_patterns: &[Vec<(usize, usize)>]) -> (Vec<u64>, usize) {
        assert!(
            base_patterns.iter().all(|p| p.len() == 6),
            "Only 6-tuple patterns are supported"
        );
        let masks: Vec<u64> = base_patterns
            .iter()
            .map(|pattern| {
                let flat: Vec<usize> = pattern.iter().map(|&(row, col)| row * 4 + col).collect();
                build_pext_mask(&flat)
            })
            .collect();
        let total_weights = base_patterns.len() * Self::TABLE_SIZE;
        (masks, total_weights)
    }

    /// Fast bitwise flip (reverse rows): swap rows 0<->3, 1<->2.
    #[inline(always)]
    fn flip(raw: u64) -> u64 {
        let buf = (raw ^ raw.rotate_left(16)) & 0x0000ffff0000ffff;
        raw ^ (buf | buf.rotate_right(16))
    }

    /// Fast bitwise transpose: swap (row,col) with (col,row).
    #[inline(always)]
    fn transpose(raw: u64) -> u64 {
        let a = raw;
        let t = (a ^ (a >> 12)) & 0x0000f0f00000f0f0;
        let a = a ^ t ^ (t << 12);
        let t = (a ^ (a >> 24)) & 0x00000000ff00ff00;
        a ^ t ^ (t << 24)
    }

    /// Evaluate base patterns against a single board orientation.
    #[inline(always)]
    fn evaluate_orientation(&self, raw: u64) -> f32 {
        let mut total = 0.0f32;
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * Self::TABLE_SIZE + index;
            // Bounds: offset is pattern_index * table_size + (pext result < table_size)
            // so it is always < masks.len() * table_size == weights.len().
            let bits = unsafe { self.weights.get_unchecked(offset) }
                .load(Ordering::Relaxed);
            total += f32::from_bits(bits);
        }
        total
    }

    /// Update base patterns for a single board orientation.
    ///
    /// Uses racy load-modify-store with Relaxed ordering — this is the
    /// Hogwild semantics. In single-threaded use it is a plain RMW; under
    /// Hogwild parallelism, concurrent writers may occasionally clobber
    /// each other's updates, which TD(0) on sparse weight tables tolerates.
    #[inline(always)]
    fn update_orientation(&self, raw: u64, delta: f32) {
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * Self::TABLE_SIZE + index;
            let slot = unsafe { self.weights.get_unchecked(offset) };
            let old_bits = slot.load(Ordering::Relaxed);
            let new_bits = (f32::from_bits(old_bits) + delta).to_bits();
            slot.store(new_bits, Ordering::Relaxed);
        }
    }

    /// Computes the 8 board orientations (D4 symmetry group).
    #[inline(always)]
    fn orientations(raw: u64) -> [u64; 8] {
        let r0 = raw;
        let r1 = Self::flip(r0);
        let r2 = Self::transpose(r1);
        let r3 = Self::flip(r2);
        let r4 = Self::transpose(r3);
        let r5 = Self::flip(r4);
        let r6 = Self::transpose(r5);
        let r7 = Self::flip(r6);
        [r0, r1, r2, r3, r4, r5, r6, r7]
    }

    /// Returns the total value estimate by evaluating all 8 board symmetries.
    #[inline]
    pub fn evaluate(&self, board: &Board) -> f32 {
        let orientations = Self::orientations(board.raw());
        let mut total = 0.0f32;
        for &oriented in &orientations {
            total += self.evaluate_orientation(oriented);
        }
        total
    }

    /// Updates all patterns across all 8 symmetries.
    ///
    /// Takes `&self` (not `&mut self`) because weights are interior-mutable
    /// via atomics. This is what enables multi-threaded Hogwild training:
    /// an `Arc<NTupleNetwork>` or a scope-shared `&NTupleNetwork` can be
    /// passed to many worker threads simultaneously.
    #[inline]
    pub fn update(&self, board: &Board, delta: f32) {
        let orientations = Self::orientations(board.raw());
        let per_symmetry_delta = delta / (self.masks.len() * 8) as f32;
        for &oriented in &orientations {
            self.update_orientation(oriented, per_symmetry_delta);
        }
    }

    pub fn num_patterns(&self) -> usize {
        self.masks.len() * 8
    }

    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// TC learning weight update: adaptive per-weight learning rate.
    ///
    /// Instead of a fixed learning rate, each weight's rate is determined by
    /// its coherence ratio α_i = |E_i| / A_i from the TC accumulators.
    /// `beta` is the meta-learning rate (typically 1.0), `td_error` is the
    /// raw TD error (not pre-scaled).
    #[inline]
    pub fn tc_update(
        &self,
        board: &Board,
        tc_state: &crate::tc_state::TcState,
        td_error: f32,
        beta: f32,
    ) {
        let orientations = Self::orientations(board.raw());
        let num_features = (self.masks.len() * 8) as f32;
        for &oriented in &orientations {
            self.tc_update_orientation(tc_state, oriented, td_error, beta, num_features);
        }
    }

    /// TC update for a single board orientation.
    ///
    /// For each pattern weight: read adaptive rate, apply scaled update,
    /// then accumulate the TD error into TC state. The accumulation happens
    /// after the rate read so α_i reflects errors *prior to* this update.
    #[inline(always)]
    fn tc_update_orientation(
        &self,
        tc_state: &crate::tc_state::TcState,
        raw: u64,
        td_error: f32,
        beta: f32,
        num_features: f32,
    ) {
        // Under hogwild, racy weight updates can cascade into Inf evaluations,
        // producing NaN TD errors (Inf - Inf). Skip the entire orientation if
        // the td_error is already poisoned.
        if !td_error.is_finite() {
            return;
        }
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * Self::TABLE_SIZE + index;
            let adaptive_rate = tc_state.adaptive_rate(offset);
            let delta = beta * (adaptive_rate / num_features) * td_error;
            let slot = unsafe { self.weights.get_unchecked(offset) };
            let old_bits = slot.load(Ordering::Relaxed);
            let new_value = f32::from_bits(old_bits) + delta;
            if new_value.is_finite() {
                slot.store(new_value.to_bits(), Ordering::Relaxed);
            }
            tc_state.accumulate(offset, td_error);
        }
    }

    /// Saves the network to a binary file.
    /// Format: [num_masks: u32] [masks: u64...] [weights: f32...]
    /// Weights are written as f32 bytes (reinterpreted from the underlying
    /// AtomicU32 storage) so the on-disk format is unchanged.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let num_masks = self.masks.len() as u32;
        file.write_all(&num_masks.to_le_bytes())?;
        for &mask in &self.masks {
            file.write_all(&mask.to_le_bytes())?;
        }
        for weight in &self.weights {
            let bits = weight.load(Ordering::Relaxed);
            file.write_all(&bits.to_le_bytes())?;
        }
        file.flush()
    }

    /// Loads a network from a binary file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        file.read_exact(&mut buf4)?;
        let num_masks = u32::from_le_bytes(buf4) as usize;

        let mut masks = Vec::with_capacity(num_masks);
        for _ in 0..num_masks {
            file.read_exact(&mut buf8)?;
            masks.push(u64::from_le_bytes(buf8));
        }

        let total_weights = num_masks * Self::TABLE_SIZE;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            file.read_exact(&mut buf4)?;
            weights.push(AtomicU32::new(u32::from_le_bytes(buf4)));
        }

        hint_huge_pages(
            weights.as_ptr() as *const u8,
            std::mem::size_of_val(&weights[..]),
        );

        Ok(Self { masks, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_weights_equals_masks_times_table_size() {
        let base_patterns = vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        ];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        assert_eq!(network.num_weights(), 2 * NTupleNetwork::TABLE_SIZE);
    }

    #[test]
    fn pext_extracts_correct_bits() {
        // Board with tile exponent 5 at position (0,0) and 3 at position (0,1)
        let mut board = Board::new();
        board.set_tile(0, 0, 5);
        board.set_tile(0, 1, 3);
        let raw = board.raw();

        // Mask selecting positions 0 and 1 (bits 0-7)
        let mask = build_pext_mask(&[0, 1]);
        let result = pext(raw, mask);
        // Should extract nibbles: position 0 = 5, position 1 = 3
        // PEXT extracts low bits first: result = 0x35
        assert_eq!(result, 0x35);
    }

    #[test]
    fn pext_mask_for_scattered_positions() {
        // Positions 0, 2, 5 → nibbles at bits 0-3, 8-11, 20-23
        let mask = build_pext_mask(&[0, 2, 5]);
        assert_eq!(mask, 0x00F00F0F);
    }

    #[test]
    fn network_evaluate_sums_all_patterns() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 10.0);
        let board = Board::new();
        assert_eq!(network.evaluate(&board), 80.0);
    }

    #[test]
    fn network_with_symmetry_expansion_reports_correct_count() {
        let base_patterns = vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        ];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        assert_eq!(network.num_patterns(), 16);
    }

    #[test]
    fn network_update_changes_evaluation() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(1, 1, 2);

        let before = network.evaluate(&board);
        network.update(&board, 100.0);
        let after = network.evaluate(&board);

        assert!(after > before, "Update should increase evaluation");
    }

    #[test]
    fn update_is_callable_via_shared_reference() {
        // This test would fail to compile if update still took &mut self.
        // It's the contract that makes Hogwild possible.
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let shared_ref: &NTupleNetwork = &network;
        let board = Board::new();
        shared_ref.update(&board, 10.0);
        assert!(shared_ref.evaluate(&board) != 0.0);
    }

    #[test]
    fn network_differentiates_board_states() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let mut board_a = Board::new();
        board_a.set_tile(0, 0, 1);

        let mut board_b = Board::new();
        board_b.set_tile(0, 0, 2);

        network.update(&board_a, 80.0);
        assert!(network.evaluate(&board_a) > 0.0);
        assert_ne!(network.evaluate(&board_a), network.evaluate(&board_b));
    }

    #[test]
    fn random_init_is_deterministic_given_seed() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let net1 = NTupleNetwork::with_random_init(&base_patterns, 0.01, 42);
        let net2 = NTupleNetwork::with_random_init(&base_patterns, 0.01, 42);
        let board = Board::new();
        assert_eq!(net1.evaluate(&board), net2.evaluate(&board));
    }

    #[test]
    fn random_init_differs_across_seeds() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let net1 = NTupleNetwork::with_random_init(&base_patterns, 0.01, 42);
        let net2 = NTupleNetwork::with_random_init(&base_patterns, 0.01, 43);
        let board = Board::new();
        assert_ne!(net1.evaluate(&board), net2.evaluate(&board));
    }

    #[test]
    fn random_init_zero_amplitude_equals_zero_init() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_random_init(&base_patterns, 0.0, 42);
        let mut board = Board::new();
        board.set_tile(2, 3, 7);
        assert_eq!(network.evaluate(&board), 0.0);
    }

    #[test]
    fn random_init_distinguishes_distinct_boards() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_random_init(&base_patterns, 0.01, 7);
        let mut board_a = Board::new();
        board_a.set_tile(0, 0, 1);
        let mut board_b = Board::new();
        board_b.set_tile(0, 0, 2);
        assert_ne!(network.evaluate(&board_a), network.evaluate(&board_b));
    }

    #[test]
    fn flip_reverses_rows() {
        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(3, 0, 2);
        let flipped = NTupleNetwork::flip(board.raw());
        let result = Board::from_raw(flipped);
        assert_eq!(result.get_tile(3, 0), 1);
        assert_eq!(result.get_tile(0, 0), 2);
    }

    #[test]
    fn transpose_swaps_rows_and_cols() {
        let mut board = Board::new();
        board.set_tile(0, 1, 5);
        board.set_tile(1, 0, 7);
        let transposed = NTupleNetwork::transpose(board.raw());
        let result = Board::from_raw(transposed);
        assert_eq!(result.get_tile(1, 0), 5);
        assert_eq!(result.get_tile(0, 1), 7);
    }

    #[test]
    fn training_improves_over_many_games() {
        use crate::training::train_one_game;
        use game_engine::MoveTables;
        use rand::SeedableRng;

        let tables = MoveTables::new();
        let base_patterns = vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        ];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut early_scores = Vec::new();
        let mut late_scores = Vec::new();

        for game in 0..2000 {
            let result = train_one_game(&network, &tables, 0.0025, &mut rng);
            if game < 100 {
                early_scores.push(result.score);
            } else if game >= 1900 {
                late_scores.push(result.score);
            }
        }

        let early_avg: f64 = early_scores.iter().sum::<u32>() as f64 / early_scores.len() as f64;
        let late_avg: f64 = late_scores.iter().sum::<u32>() as f64 / late_scores.len() as f64;

        assert!(
            late_avg > early_avg,
            "Expected improvement: early avg {early_avg:.0}, late avg {late_avg:.0}"
        );
    }

    #[test]
    fn tc_update_with_fresh_state_modifies_weights() {
        use crate::tc_state::TcState;

        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let tc_state = TcState::new(network.num_weights());

        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(1, 1, 2);

        let before = network.evaluate(&board);
        network.tc_update(&board, &tc_state, 100.0, 1.0);
        let after = network.evaluate(&board);

        assert!(after > before, "TC update should increase evaluation");
    }

    #[test]
    fn tc_update_coherent_weights_learn_faster_than_oscillating() {
        use crate::tc_state::TcState;

        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let tc_state = TcState::new(network.num_weights());

        let mut board = Board::new();
        board.set_tile(0, 0, 1);

        // Apply several consistent positive updates
        for _ in 0..10 {
            network.tc_update(&board, &tc_state, 50.0, 1.0);
        }
        let coherent_value = network.evaluate(&board);

        // Reset network, apply alternating updates (same magnitude)
        let network2 = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let tc_state2 = TcState::new(network2.num_weights());
        for step in 0..10 {
            let error = if step % 2 == 0 { 50.0 } else { -50.0 };
            network2.tc_update(&board, &tc_state2, error, 1.0);
        }
        let oscillating_value = network2.evaluate(&board).abs();

        assert!(
            coherent_value.abs() > oscillating_value,
            "Coherent updates should produce larger weight magnitude \
             ({coherent_value}) than oscillating ({oscillating_value})"
        );
    }

    #[test]
    fn tc_update_is_callable_via_shared_reference() {
        use crate::tc_state::TcState;

        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let tc_state = TcState::new(network.num_weights());

        let shared_ref: &NTupleNetwork = &network;
        let board = Board::new();
        shared_ref.tc_update(&board, &tc_state, 10.0, 1.0);
        assert!(shared_ref.evaluate(&board) != 0.0);
    }
}
