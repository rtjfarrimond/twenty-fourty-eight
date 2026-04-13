/// Represents a 2048 board as a packed u64 bitboard.
/// Each tile is stored as a 4-bit exponent (0 = empty, 1 = 2, 2 = 4, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Board {
    state: u64,
}

impl Board {
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Creates a board from a raw u64 state.
    pub fn from_raw(raw: u64) -> Self {
        Self { state: raw }
    }

    /// Returns the raw u64 state for efficient comparison and hashing.
    pub fn raw(&self) -> u64 {
        self.state
    }

    /// Returns the tile exponent at the given row and column.
    /// 0 means empty, 1 means tile value 2, 2 means tile value 4, etc.
    pub fn get_tile(&self, row: usize, col: usize) -> u8 {
        let shift = (row * 4 + col) * 4;
        ((self.state >> shift) & 0xF) as u8
    }

    /// Returns a row as a packed 16-bit value (4 nibbles, low = col 0).
    pub fn get_row(&self, row: usize) -> u16 {
        let shift = row * 16;
        ((self.state >> shift) & 0xFFFF) as u16
    }

    /// Replaces an entire row with a packed 16-bit value.
    pub fn set_row(&mut self, row: usize, packed_row: u16) {
        let shift = row * 16;
        let mask = !(0xFFFF_u64 << shift);
        self.state = (self.state & mask) | ((packed_row as u64) << shift);
    }

    /// Sets the tile exponent at the given row and column.
    pub fn set_tile(&mut self, row: usize, col: usize, exponent: u8) {
        let shift = (row * 4 + col) * 4;
        let mask = !(0xF_u64 << shift);
        self.state = (self.state & mask) | ((exponent as u64) << shift);
    }

    /// Returns a new board with rows and columns swapped.
    /// Used to apply row-based move logic to columns (for up/down moves).
    pub fn transpose(&self) -> Self {
        let mut result = Board::new();
        for row in 0..4 {
            for col in 0..4 {
                result.set_tile(col, row, self.get_tile(row, col));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_board_is_empty() {
        let board = Board::new();
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(board.get_tile(row, col), 0);
            }
        }
    }

    #[test]
    fn set_and_get_tile() {
        let mut board = Board::new();
        board.set_tile(0, 0, 1); // exponent 1 = tile value 2
        board.set_tile(3, 3, 11); // exponent 11 = tile value 2048

        assert_eq!(board.get_tile(0, 0), 1);
        assert_eq!(board.get_tile(3, 3), 11);
        // other tiles remain empty
        assert_eq!(board.get_tile(1, 1), 0);
    }

    #[test]
    fn get_row_extracts_correct_tiles() {
        let mut board = Board::new();
        // Row 0: exponents [1, 2, 3, 4] → tile values [2, 4, 8, 16]
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 2);
        board.set_tile(0, 2, 3);
        board.set_tile(0, 3, 4);

        // Packed as 16 bits: col0 in low nibble, col3 in high nibble
        let expected: u16 = 1 | (2 << 4) | (3 << 8) | (4 << 12);
        assert_eq!(board.get_row(0), expected);
    }

    #[test]
    fn get_row_on_empty_board_is_zero() {
        let board = Board::new();
        for row in 0..4 {
            assert_eq!(board.get_row(row), 0);
        }
    }

    #[test]
    fn set_row_replaces_entire_row() {
        let mut board = Board::new();
        let packed_row: u16 = 1 | (2 << 4) | (3 << 8) | (4 << 12);
        board.set_row(2, packed_row);

        assert_eq!(board.get_tile(2, 0), 1);
        assert_eq!(board.get_tile(2, 1), 2);
        assert_eq!(board.get_tile(2, 2), 3);
        assert_eq!(board.get_tile(2, 3), 4);
        // other rows unaffected
        assert_eq!(board.get_row(0), 0);
        assert_eq!(board.get_row(1), 0);
        assert_eq!(board.get_row(3), 0);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let mut board = Board::new();
        // Set up a board where row 0 = [1,2,3,4], rest empty
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 2);
        board.set_tile(0, 2, 3);
        board.set_tile(0, 3, 4);

        let transposed = board.transpose();

        // After transpose, column 0 becomes row 0, etc.
        // So tile at (0,0)=1 stays, (0,1)=2 goes to (1,0), etc.
        assert_eq!(transposed.get_tile(0, 0), 1);
        assert_eq!(transposed.get_tile(1, 0), 2);
        assert_eq!(transposed.get_tile(2, 0), 3);
        assert_eq!(transposed.get_tile(3, 0), 4);
        // original row 1-3 were empty, so transposed cols 1-3 are empty
        assert_eq!(transposed.get_tile(0, 1), 0);
        assert_eq!(transposed.get_tile(0, 2), 0);
        assert_eq!(transposed.get_tile(0, 3), 0);
    }
}
