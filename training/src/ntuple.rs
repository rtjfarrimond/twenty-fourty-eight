use game_engine::Board;

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
pub struct NTupleNetwork {
    /// One PEXT mask per base pattern.
    masks: Vec<u64>,
    /// Flat weight storage: masks.len() contiguous weight tables.
    weights: Vec<f32>,
    /// Size of each weight table (16^6 for 6-tuples).
    table_size: usize,
}

impl NTupleNetwork {
    pub fn with_symmetry_expansion(
        base_patterns: &[Vec<(usize, usize)>],
        initial_weight: f32,
    ) -> Self {
        assert!(
            base_patterns.iter().all(|p| p.len() == 6),
            "Only 6-tuple patterns are supported"
        );
        let table_size = TILE_VALUES.pow(6);
        let masks: Vec<u64> = base_patterns
            .iter()
            .map(|pattern| {
                let flat: Vec<usize> = pattern.iter().map(|&(row, col)| row * 4 + col).collect();
                build_pext_mask(&flat)
            })
            .collect();

        let total_weights = base_patterns.len() * table_size;
        let weights = vec![initial_weight; total_weights];

        Self {
            masks,
            weights,
            table_size,
        }
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
        let weights_ptr = self.weights.as_ptr();
        let mut total = 0.0f32;
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * self.table_size + index;
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
        for (pattern_index, &mask) in self.masks.iter().enumerate() {
            let index = pext(raw, mask) as usize;
            let offset = pattern_index * self.table_size + index;
            unsafe {
                *weights_ptr.add(offset) += delta;
            }
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
    #[inline]
    pub fn update(&mut self, board: &Board, delta: f32) {
        let orientations = Self::orientations(board.raw());
        let per_symmetry_delta = delta / (self.masks.len() * 8) as f32;
        for &oriented in &orientations {
            self.update_orientation(oriented, per_symmetry_delta);
        }
    }

    pub fn num_patterns(&self) -> usize {
        self.masks.len() * 8
    }

    /// Saves the network to a binary file.
    /// Format: [num_masks: u32] [masks: u64...] [weights: f32...]
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let num_masks = self.masks.len() as u32;
        file.write_all(&num_masks.to_le_bytes())?;
        for &mask in &self.masks {
            file.write_all(&mask.to_le_bytes())?;
        }
        for &weight in &self.weights {
            file.write_all(&weight.to_le_bytes())?;
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

        let table_size = TILE_VALUES.pow(6);
        let total_weights = num_masks * table_size;
        let mut weights = Vec::with_capacity(total_weights);
        for _ in 0..total_weights {
            file.read_exact(&mut buf4)?;
            weights.push(f32::from_le_bytes(buf4));
        }

        Ok(Self {
            masks,
            weights,
            table_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

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
