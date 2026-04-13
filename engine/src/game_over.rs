use crate::board::Board;
use crate::direction::Direction;
use crate::game::apply_move;
use crate::moves::MoveTables;

const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Left,
    Direction::Right,
    Direction::Up,
    Direction::Down,
];

/// Returns true if no legal move exists (the game is over).
pub fn is_game_over(board: &Board, tables: &MoveTables) -> bool {
    ALL_DIRECTIONS
        .iter()
        .all(|&direction| apply_move(board, direction, tables).is_none())
}

/// Returns the list of directions that produce a board change.
pub fn legal_moves(board: &Board, tables: &MoveTables) -> Vec<Direction> {
    ALL_DIRECTIONS
        .iter()
        .copied()
        .filter(|&direction| apply_move(board, direction, tables).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> MoveTables {
        MoveTables::new()
    }

    #[test]
    fn board_with_single_tile_is_not_game_over() {
        let tables = tables();
        let mut board = Board::new();
        board.set_tile(1, 1, 1);
        assert!(!is_game_over(&board, &tables));
    }

    #[test]
    fn full_board_with_no_merges_is_game_over() {
        let tables = tables();
        let mut board = Board::new();
        // Fill with distinct values so no merges are possible
        // 1,2,3,4 / 5,6,7,8 / 1,2,3,4 / 5,6,7,8
        let values = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]];
        for row in 0..4 {
            for col in 0..4 {
                board.set_tile(row, col, values[row][col]);
            }
        }

        assert!(is_game_over(&board, &tables));
    }

    #[test]
    fn full_board_with_horizontal_merge_is_not_game_over() {
        let tables = tables();
        let mut board = Board::new();
        let values = [
            [1, 1, 3, 4], // first two can merge
            [5, 6, 7, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ];
        for row in 0..4 {
            for col in 0..4 {
                board.set_tile(row, col, values[row][col]);
            }
        }

        assert!(!is_game_over(&board, &tables));
    }

    #[test]
    fn full_board_with_vertical_merge_is_not_game_over() {
        let tables = tables();
        let mut board = Board::new();
        let values = [
            [1, 2, 3, 4],
            [1, 6, 7, 8], // (0,0) and (1,0) can merge vertically
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ];
        for row in 0..4 {
            for col in 0..4 {
                board.set_tile(row, col, values[row][col]);
            }
        }

        assert!(!is_game_over(&board, &tables));
    }

    #[test]
    fn legal_moves_on_empty_board() {
        let board = Board::new();
        let moves = legal_moves(&board, &tables());
        // No tiles to move, so no direction changes the board
        assert!(moves.is_empty());
    }

    #[test]
    fn legal_moves_excludes_noop_directions() {
        let tables = tables();
        let mut board = Board::new();
        // Single tile at top-left: can move right and down, not left or up
        board.set_tile(0, 0, 1);

        let moves = legal_moves(&board, &tables);
        assert!(moves.contains(&Direction::Right));
        assert!(moves.contains(&Direction::Down));
        assert!(!moves.contains(&Direction::Left));
        assert!(!moves.contains(&Direction::Up));
    }
}
