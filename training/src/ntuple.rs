use game_engine::Board;

/// The number of possible tile values (0=empty, 1=2, ..., 15=32768).
const TILE_VALUES: usize = 16;

/// Describes how to extract a 6-tuple index from a raw board u64 using
/// at most two mask+shift operations. This works for patterns where tiles
/// come from at most two contiguous groups within the u64.
struct IndexExtractor {
    /// Mask for the first group of bits.
    mask_a: u64,
    /// Mask for the second group (0 if single group).
    mask_b: u64,
    /// Right-shift to bring group A down to bit 0.
    shift_a: u32,
    /// Additional right-shift to close the gap between groups A and B.
    shift_b: u32,
}

impl IndexExtractor {
    /// Builds an extractor for the given cell positions (flat indices 0-15).
    /// The positions are expected to already be sorted.
    fn new(positions: &[usize; 6]) -> Self {
        // Compute the bitmask for each position's nibble
        // and figure out how to pack them into a contiguous index.
        //
        // For positions that are consecutive in the u64 (e.g. 0,1,2 = bits 0-11),
        // we can extract them with a single mask.
        // For positions that span a gap, we need two masks and a shift.

        // Build a mapping: for each position, what nibble offset is it in the u64?
        let bit_positions: Vec<u32> = positions.iter().map(|&p| (p as u32) * 4).collect();

        // Find groups of consecutive nibbles
        // Group boundary: when bit_positions[i] != bit_positions[i-1] + 4
        let mut groups: Vec<(u32, u32, usize)> = Vec::new(); // (start_bit, end_bit, count)
        let mut group_start = bit_positions[0];
        let mut group_count: usize = 1;

        for window in bit_positions.windows(2) {
            if window[1] == window[0] + 4 {
                group_count += 1;
            } else {
                groups.push((group_start, group_start + (group_count as u32) * 4, group_count));
                group_start = window[1];
                group_count = 1;
            }
        }
        groups.push((group_start, group_start + (group_count as u32) * 4, group_count));

        match groups.len() {
            1 => {
                // All positions are consecutive — single mask
                let bits = groups[0].2 * 4;
                let mask = ((1u64 << bits) - 1) << groups[0].0;
                Self {
                    mask_a: mask,
                    mask_b: 0,
                    shift_a: groups[0].0,
                    shift_b: 0,
                }
            }
            2 => {
                // Two groups — extract both and pack contiguously from bit 0.
                // Group A starts at groups[0].0, group B starts at groups[1].0.
                // We shift A down to bit 0, and B down so it sits above A.
                let bits_a = groups[0].2 * 4;
                let mask_a = ((1u64 << bits_a) - 1) << groups[0].0;
                let mask_b = ((1u64 << (groups[1].2 * 4)) - 1) << groups[1].0;
                Self {
                    mask_a,
                    mask_b,
                    shift_a: groups[0].0,
                    shift_b: groups[1].0 - groups[0].0 - bits_a as u32,
                }
            }
            _ => {
                // Fallback for complex patterns — use a general approach
                // For now, assert we only handle 1-2 groups
                panic!(
                    "Pattern positions form {} groups; only 1-2 supported. \
                     Positions: {:?}",
                    groups.len(),
                    positions
                );
            }
        }
    }

    /// Extracts the tuple index from a raw board state.
    #[inline(always)]
    fn extract(&self, raw: u64) -> usize {
        if self.mask_b == 0 {
            ((raw & self.mask_a) >> self.shift_a) as usize
        } else {
            let part_a = (raw & self.mask_a) >> self.shift_a;
            let part_b = (raw & self.mask_b) >> (self.shift_a + self.shift_b);
            (part_a | part_b) as usize
        }
    }
}

/// An n-tuple network using the isomorphic evaluation approach:
/// - Stores only the base patterns (no symmetry expansion in weights)
/// - At evaluation time, transforms the board 8 ways using fast bitwise ops
/// - Evaluates each base pattern against each transformed board
pub struct NTupleNetwork {
    /// One extractor per base pattern.
    extractors: Vec<IndexExtractor>,
    /// Flat weight storage: extractors.len() contiguous weight tables.
    weights: Vec<f32>,
    /// Size of each weight table (16^6 for 6-tuples).
    table_size: usize,
}

impl NTupleNetwork {
    pub fn with_symmetry_expansion(
        base_patterns: &[Vec<(usize, usize)>],
        initial_weight: f32,
    ) -> Self {
        let table_size = TILE_VALUES.pow(6);
        let mut extractors = Vec::with_capacity(base_patterns.len());

        for pattern in base_patterns {
            assert_eq!(pattern.len(), 6, "Only 6-tuple patterns are supported");
            let mut flat: [usize; 6] = [0; 6];
            for (index, &(row, col)) in pattern.iter().enumerate() {
                flat[index] = row * 4 + col;
            }
            flat.sort();
            extractors.push(IndexExtractor::new(&flat));
        }

        let total_weights = base_patterns.len() * table_size;
        let weights = vec![initial_weight; total_weights];

        Self {
            extractors,
            weights,
            table_size,
        }
    }

    /// Fast bitwise mirror (reverse columns): swap cols 0<->3, 1<->2.
    #[inline(always)]
    fn mirror(raw: u64) -> u64 {
        ((raw & 0x000f000f000f000f) << 12)
            | ((raw & 0x00f000f000f000f0) << 4)
            | ((raw & 0x0f000f000f000f00) >> 4)
            | ((raw & 0xf000f000f000f000) >> 12)
    }

    /// Fast bitwise flip (reverse rows): swap rows 0<->3, 1<->2.
    #[inline(always)]
    fn flip(raw: u64) -> u64 {
        let buf = (raw ^ raw.rotate_left(16)) & 0x0000ffff0000ffff;
        raw ^ (buf | buf.rotate_right(16))
    }

    /// Fast bitwise transpose: swap (row,col) with (col,row).
    /// Uses the delta-swap approach from the reference implementation.
    #[inline(always)]
    fn transpose(raw: u64) -> u64 {
        let a = raw;
        let t = (a ^ (a >> 12)) & 0x0000f0f00000f0f0;
        let a = a ^ t ^ (t << 12);
        let t = (a ^ (a >> 24)) & 0x00000000ff00ff00;
        let a = a ^ t ^ (t << 24);
        a
    }

    /// Evaluate base patterns against a single board orientation.
    #[inline(always)]
    fn evaluate_orientation(&self, raw: u64) -> f32 {
        let weights_ptr = self.weights.as_ptr();
        let mut total = 0.0f32;
        for (pattern_index, extractor) in self.extractors.iter().enumerate() {
            let index = extractor.extract(raw);
            debug_assert!(
                index < self.table_size,
                "Index {} out of bounds (table_size {}), pattern {}",
                index, self.table_size, pattern_index
            );
            let offset = pattern_index * self.table_size + index;
            debug_assert!(offset < self.weights.len());
            unsafe {
                total += *weights_ptr.add(offset);
            }
        }
        total
    }

    /// Update base patterns for a single board orientation.
    #[inline(always)]
    fn update_orientation(&mut self, raw: u64, delta: f32) {
        let weights_ptr = self.weights.as_mut_ptr();
        for (pattern_index, extractor) in self.extractors.iter().enumerate() {
            let index = extractor.extract(raw);
            let offset = pattern_index * self.table_size + index;
            unsafe {
                *weights_ptr.add(offset) += delta;
            }
        }
    }

    /// Returns the total value estimate by evaluating all 8 board symmetries.
    #[inline]
    pub fn evaluate(&self, board: &Board) -> f32 {
        let raw = board.raw();
        let mut total = 0.0f32;

        // 8 isomorphisms via flip and transpose
        let r0 = raw;
        let r1 = Self::flip(r0);
        let r2 = Self::transpose(r1);
        let r3 = Self::flip(r2);
        let r4 = Self::transpose(r3);
        let r5 = Self::flip(r4);
        let r6 = Self::transpose(r5);
        let r7 = Self::flip(r6);

        total += self.evaluate_orientation(r0);
        total += self.evaluate_orientation(r1);
        total += self.evaluate_orientation(r2);
        total += self.evaluate_orientation(r3);
        total += self.evaluate_orientation(r4);
        total += self.evaluate_orientation(r5);
        total += self.evaluate_orientation(r6);
        total += self.evaluate_orientation(r7);

        total
    }

    /// Updates all patterns across all 8 symmetries.
    #[inline]
    pub fn update(&mut self, board: &Board, delta: f32) {
        let raw = board.raw();
        let per_symmetry_delta = delta / (self.extractors.len() * 8) as f32;

        let r0 = raw;
        let r1 = Self::flip(r0);
        let r2 = Self::transpose(r1);
        let r3 = Self::flip(r2);
        let r4 = Self::transpose(r3);
        let r5 = Self::flip(r4);
        let r6 = Self::transpose(r5);
        let r7 = Self::flip(r6);

        self.update_orientation(r0, per_symmetry_delta);
        self.update_orientation(r1, per_symmetry_delta);
        self.update_orientation(r2, per_symmetry_delta);
        self.update_orientation(r3, per_symmetry_delta);
        self.update_orientation(r4, per_symmetry_delta);
        self.update_orientation(r5, per_symmetry_delta);
        self.update_orientation(r6, per_symmetry_delta);
        self.update_orientation(r7, per_symmetry_delta);
    }

    pub fn num_patterns(&self) -> usize {
        self.extractors.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_evaluate_sums_all_patterns() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 10.0);
        let board = Board::new();
        // 1 base pattern x 8 symmetries x 10.0 = 80.0
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
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        // Use a non-symmetric board so different orientations hit different entries
        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(1, 1, 2);

        let before = network.evaluate(&board);
        network.update(&board, 100.0);
        let after = network.evaluate(&board);

        assert!(after > before, "Update should increase evaluation");
    }

    #[test]
    fn network_differentiates_board_states() {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let mut board_a = Board::new();
        board_a.set_tile(0, 0, 1);

        let mut board_b = Board::new();
        board_b.set_tile(0, 0, 2);

        network.update(&board_a, 80.0);
        assert!(network.evaluate(&board_a) > 0.0);
        assert_ne!(network.evaluate(&board_a), network.evaluate(&board_b));
    }

    #[test]
    fn mirror_reverses_columns() {
        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(0, 3, 2);
        let mirrored = NTupleNetwork::mirror(board.raw());
        let result = Board::from_raw(mirrored);
        assert_eq!(result.get_tile(0, 3), 1);
        assert_eq!(result.get_tile(0, 0), 2);
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
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut early_scores = Vec::new();
        let mut late_scores = Vec::new();

        for game in 0..2000 {
            let score = train_one_game(&mut network, &tables, 0.0025, &mut rng);
            if game < 100 {
                early_scores.push(score);
            } else if game >= 1900 {
                late_scores.push(score);
            }
        }

        let early_avg: f64 = early_scores.iter().sum::<u32>() as f64 / early_scores.len() as f64;
        let late_avg: f64 = late_scores.iter().sum::<u32>() as f64 / late_scores.len() as f64;

        assert!(
            late_avg > early_avg,
            "Expected improvement: early avg {early_avg:.0}, late avg {late_avg:.0}"
        );
    }
}
