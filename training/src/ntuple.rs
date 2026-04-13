use game_engine::Board;

/// The number of possible tile values (0=empty, 1=2, ..., 15=32768).
const TILE_VALUES: usize = 16;

/// A single n-tuple pattern: a list of board positions (row, col)
/// and a weight table indexed by the combined tile values at those positions.
pub struct TuplePattern {
    positions: Vec<(usize, usize)>,
    weights: Vec<f32>,
}

impl TuplePattern {
    /// Creates a new tuple pattern with the given board positions.
    /// All weights are initialized to the given value.
    pub fn new(positions: Vec<(usize, usize)>, initial_weight: f32) -> Self {
        let table_size = TILE_VALUES.pow(positions.len() as u32);
        Self {
            positions,
            weights: vec![initial_weight; table_size],
        }
    }

    /// Computes the index into the weight table for the given board state.
    fn index_for(&self, board: &Board) -> usize {
        let mut index = 0;
        for &(row, col) in &self.positions {
            index = index * TILE_VALUES + board.get_tile(row, col) as usize;
        }
        index
    }

    /// Returns the weight (value estimate) for the given board state.
    pub fn evaluate(&self, board: &Board) -> f32 {
        self.weights[self.index_for(board)]
    }

    /// Updates the weight for the given board state by the given delta.
    pub fn update(&mut self, board: &Board, delta: f32) {
        let index = self.index_for(board);
        self.weights[index] += delta;
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
    pub fn evaluate(&self, board: &Board) -> f32 {
        self.patterns.iter().map(|p| p.evaluate(board)).sum()
    }

    /// Updates all patterns for the given board state by delta / num_patterns.
    pub fn update(&mut self, board: &Board, delta: f32) {
        let per_pattern_delta = delta / self.patterns.len() as f32;
        for pattern in &mut self.patterns {
            pattern.update(board, per_pattern_delta);
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
        |row, col| (row, col),         // identity
        |row, col| (col, 3 - row),     // 90° clockwise
        |row, col| (3 - row, 3 - col), // 180°
        |row, col| (3 - col, row),     // 270° clockwise
        |row, col| (row, 3 - col),     // horizontal reflection
        |row, col| (3 - col, 3 - row), // reflect + 90°
        |row, col| (3 - row, col),     // vertical reflection
        |row, col| (col, row),         // reflect + 270° (transpose)
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

        // Different tile values should yield different evaluations after update
        let mut pattern = TuplePattern::new(vec![(0, 0), (0, 1)], 0.0);
        pattern.update(&board_a, 10.0);

        assert_eq!(pattern.evaluate(&board_a), 10.0);
        assert_eq!(pattern.evaluate(&board_b), 0.0); // different index
    }

    #[test]
    fn tuple_pattern_update_modifies_correct_weight() {
        let mut pattern = TuplePattern::new(vec![(0, 0)], 0.0);
        let mut board = Board::new();
        board.set_tile(0, 0, 3);

        pattern.update(&board, 5.0);
        assert_eq!(pattern.evaluate(&board), 5.0);

        // Empty board uses different index
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

        // 8 symmetries of a single position, each evaluating to 10.0
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
        assert_eq!(network.num_patterns(), 16); // 2 base x 8 symmetries
    }

    #[test]
    fn network_update_distributes_delta_across_patterns() {
        let base_patterns = vec![vec![(0, 0)]];
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let board = Board::new();

        network.update(&board, 80.0);
        // 80.0 / 8 patterns = 10.0 per pattern
        assert_eq!(network.evaluate(&board), 80.0);
    }
}
