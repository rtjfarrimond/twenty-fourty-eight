use game_engine::Board;

/// The number of possible tile values (0=empty, 1=2, ..., 15=32768).
const TILE_VALUES: usize = 16;

/// Maximum tuple size supported.
const MAX_TUPLE_SIZE: usize = 8;

/// A single n-tuple pattern: stores precomputed bit-shift offsets in a
/// fixed-size array for fast index extraction directly from the board's raw u64.
pub struct TuplePattern {
    /// Bit-shift amounts for each position in the tuple.
    shifts: [u32; MAX_TUPLE_SIZE],
    /// How many positions are actually used.
    tuple_size: usize,
    weights: Vec<f32>,
}

impl TuplePattern {
    /// Creates a new tuple pattern from (row, col) positions.
    pub fn new(positions: Vec<(usize, usize)>, initial_weight: f32) -> Self {
        assert!(positions.len() <= MAX_TUPLE_SIZE);
        let mut shifts = [0u32; MAX_TUPLE_SIZE];
        for (index, &(row, col)) in positions.iter().enumerate() {
            shifts[index] = ((row * 4 + col) * 4) as u32;
        }
        let table_size = TILE_VALUES.pow(positions.len() as u32);
        Self {
            shifts,
            tuple_size: positions.len(),
            weights: vec![initial_weight; table_size],
        }
    }

    /// Computes the weight table index directly from the raw board state.
    /// Manually unrolled for the common case (6-tuples) to allow full
    /// compiler optimization.
    #[inline(always)]
    fn index_for_raw(&self, raw: u64) -> usize {
        match self.tuple_size {
            6 => {
                ((raw >> self.shifts[0]) as usize & 0xF) << 20
                    | ((raw >> self.shifts[1]) as usize & 0xF) << 16
                    | ((raw >> self.shifts[2]) as usize & 0xF) << 12
                    | ((raw >> self.shifts[3]) as usize & 0xF) << 8
                    | ((raw >> self.shifts[4]) as usize & 0xF) << 4
                    | ((raw >> self.shifts[5]) as usize & 0xF)
            }
            _ => {
                let mut index = 0usize;
                for &shift in &self.shifts[..self.tuple_size] {
                    index = (index << 4) | ((raw >> shift) as usize & 0xF);
                }
                index
            }
        }
    }

    /// Returns the weight for the given raw board state.
    #[inline(always)]
    pub fn evaluate_raw(&self, raw: u64) -> f32 {
        unsafe { *self.weights.get_unchecked(self.index_for_raw(raw)) }
    }

    /// Updates the weight for the given raw board state.
    #[inline(always)]
    pub fn update_raw(&mut self, raw: u64, delta: f32) {
        let index = self.index_for_raw(raw);
        unsafe { *self.weights.get_unchecked_mut(index) += delta; }
    }

    /// Returns the weight for the given board state.
    pub fn evaluate(&self, board: &Board) -> f32 {
        self.evaluate_raw(board.raw())
    }

    /// Updates the weight for the given board state.
    pub fn update(&mut self, board: &Board, delta: f32) {
        self.update_raw(board.raw(), delta);
    }

    /// Returns the number of entries in the weight table.
    pub fn table_size(&self) -> usize {
        self.weights.len()
    }
}

/// An n-tuple network: a collection of tuple patterns.
/// The board value is the sum of all patterns' evaluations.
pub struct NTupleNetwork {
    patterns: Vec<TuplePattern>,
}

impl NTupleNetwork {
    /// Creates a network from base patterns, expanding each by the 8
    /// symmetries of the board (4 rotations x 2 reflections).
    pub fn with_symmetry_expansion(
        base_patterns: &[Vec<(usize, usize)>],
        initial_weight: f32,
    ) -> Self {
        let mut patterns = Vec::new();
        for base in base_patterns {
            for symmetry in all_symmetries(base) {
                patterns.push(TuplePattern::new(symmetry, initial_weight));
            }
        }
        Self { patterns }
    }

    /// Returns the total value estimate for the given board state.
    #[inline]
    pub fn evaluate(&self, board: &Board) -> f32 {
        let raw = board.raw();
        self.patterns.iter().map(|p| p.evaluate_raw(raw)).sum()
    }

    /// Updates all patterns for the given board state by delta / num_patterns.
    #[inline]
    pub fn update(&mut self, board: &Board, delta: f32) {
        let raw = board.raw();
        let per_pattern_delta = delta / self.patterns.len() as f32;
        for pattern in &mut self.patterns {
            pattern.update_raw(raw, per_pattern_delta);
        }
    }

    /// Returns the total number of patterns (including symmetry expansions).
    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }
}

/// Generates all 8 symmetries (D4 group) of a set of board positions.
fn all_symmetries(positions: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let transforms: Vec<fn(usize, usize) -> (usize, usize)> = vec![
        |row, col| (row, col),             // identity
        |row, col| (col, 3 - row),         // 90° clockwise
        |row, col| (3 - row, 3 - col),     // 180°
        |row, col| (3 - col, row),         // 270° clockwise
        |row, col| (row, 3 - col),         // horizontal reflection
        |row, col| (3 - col, 3 - row),     // reflect + 90°
        |row, col| (3 - row, col),         // vertical reflection
        |row, col| (col, row),             // reflect + 270° (transpose)
    ];

    transforms
        .iter()
        .map(|transform| {
            positions
                .iter()
                .map(|&(row, col)| transform(row, col))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_pattern_evaluates_to_initial_weight() {
        let pattern = TuplePattern::new(vec![(0, 0), (0, 1)], 100.0);
        let board = Board::new();
        assert_eq!(pattern.evaluate(&board), 100.0);
    }

    #[test]
    fn tuple_pattern_index_varies_with_tile_values() {
        let mut board_a = Board::new();
        board_a.set_tile(0, 0, 1);

        let mut board_b = Board::new();
        board_b.set_tile(0, 0, 2);

        let mut pattern = TuplePattern::new(vec![(0, 0), (0, 1)], 0.0);
        pattern.update(&board_a, 10.0);

        assert_eq!(pattern.evaluate(&board_a), 10.0);
        assert_eq!(pattern.evaluate(&board_b), 0.0);
    }

    #[test]
    fn tuple_pattern_update_modifies_correct_weight() {
        let mut pattern = TuplePattern::new(vec![(0, 0)], 0.0);
        let mut board = Board::new();
        board.set_tile(0, 0, 3);

        pattern.update(&board, 5.0);
        assert_eq!(pattern.evaluate(&board), 5.0);
        assert_eq!(pattern.evaluate(&Board::new()), 0.0);
    }

    #[test]
    fn tuple_pattern_table_size_correct_for_6tuple() {
        let positions = vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)];
        let pattern = TuplePattern::new(positions, 0.0);
        assert_eq!(pattern.table_size(), 16_usize.pow(6));
    }

    #[test]
    fn all_symmetries_produces_8_variants() {
        let base = vec![(0, 0), (0, 1), (0, 2)];
        let symmetries = all_symmetries(&base);
        assert_eq!(symmetries.len(), 8);
    }

    #[test]
    fn all_symmetries_identity_preserves_positions() {
        let base = vec![(0, 0), (1, 2), (3, 3)];
        let symmetries = all_symmetries(&base);
        assert_eq!(symmetries[0], base);
    }

    #[test]
    fn network_evaluate_sums_all_patterns() {
        let base_patterns = vec![vec![(0, 0)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 10.0);
        let board = Board::new();
        assert_eq!(network.evaluate(&board), 80.0);
    }

    #[test]
    fn network_with_symmetry_expansion_creates_8x_patterns() {
        let base_patterns = vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        ];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        assert_eq!(network.num_patterns(), 16);
    }

    #[test]
    fn network_update_distributes_delta_across_patterns() {
        let base_patterns = vec![vec![(0, 0)]];
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let board = Board::new();

        network.update(&board, 80.0);
        assert_eq!(network.evaluate(&board), 80.0);
    }
}
