use crate::board::Board;

const FOUR_TILE_THRESHOLD: f64 = 0.9;

/// Returns the positions (row, col) of all empty tiles on the board.
pub fn empty_tiles(board: &Board) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for row in 0..4 {
        for col in 0..4 {
            if board.get_tile(row, col) == 0 {
                positions.push((row, col));
            }
        }
    }
    positions
}

/// Spawns a tile on the board at the given position.
/// Exponent is 1 (tile value 2) with 90% probability, 2 (tile value 4) with 10%.
/// `random_value` should be a float in [0.0, 1.0) used to determine the tile.
pub fn spawn_tile(board: &Board, row: usize, col: usize, random_value: f64) -> Board {
    let exponent = if random_value < FOUR_TILE_THRESHOLD {
        1
    } else {
        2
    };
    let mut new_board = *board;
    new_board.set_tile(row, col, exponent);
    new_board
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tiles_on_new_board_returns_all_16() {
        let board = Board::new();
        let empties = empty_tiles(&board);
        assert_eq!(empties.len(), 16);
    }

    #[test]
    fn empty_tiles_excludes_occupied_positions() {
        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(2, 3, 5);

        let empties = empty_tiles(&board);
        assert_eq!(empties.len(), 14);
        assert!(!empties.contains(&(0, 0)));
        assert!(!empties.contains(&(2, 3)));
    }

    #[test]
    fn spawn_tile_places_two_when_random_below_threshold() {
        let board = Board::new();
        let result = spawn_tile(&board, 1, 2, 0.5); // 0.5 < 0.9 → exponent 1
        assert_eq!(result.get_tile(1, 2), 1);
    }

    #[test]
    fn spawn_tile_places_four_when_random_at_or_above_threshold() {
        let board = Board::new();
        let result = spawn_tile(&board, 1, 2, 0.95); // 0.95 >= 0.9 → exponent 2
        assert_eq!(result.get_tile(1, 2), 2);
    }

    #[test]
    fn spawn_tile_does_not_affect_other_tiles() {
        let mut board = Board::new();
        board.set_tile(0, 0, 3);

        let result = spawn_tile(&board, 1, 1, 0.0);
        assert_eq!(result.get_tile(0, 0), 3); // preserved
        assert_eq!(result.get_tile(1, 1), 1); // spawned
    }
}
