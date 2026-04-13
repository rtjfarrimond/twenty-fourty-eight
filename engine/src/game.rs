use crate::board::Board;
use crate::direction::Direction;
use crate::moves::MoveTables;

/// Applies a move to a board using precomputed lookup tables.
/// Returns (new_board, score) or None if the move has no effect.
pub fn apply_move(
    board: &Board,
    direction: Direction,
    tables: &MoveTables,
) -> Option<(Board, u32)> {
    let (new_board, score) = match direction {
        Direction::Left => apply_horizontal(board, &tables.left_result, &tables.left_score),
        Direction::Right => apply_horizontal(board, &tables.right_result, &tables.right_score),
        Direction::Up => apply_vertical(board, &tables.left_result, &tables.left_score),
        Direction::Down => apply_vertical(board, &tables.right_result, &tables.right_score),
    };

    if new_board.raw() == board.raw() {
        None
    } else {
        Some((new_board, score))
    }
}

fn apply_horizontal(board: &Board, result_table: &[u16], score_table: &[u32]) -> (Board, u32) {
    let mut new_board = Board::new();
    let mut total_score = 0;

    for row in 0..4 {
        let packed_row = board.get_row(row) as usize;
        new_board.set_row(row, result_table[packed_row]);
        total_score += score_table[packed_row];
    }

    (new_board, total_score)
}

fn apply_vertical(board: &Board, result_table: &[u16], score_table: &[u32]) -> (Board, u32) {
    let transposed = board.transpose();
    let (result, score) = apply_horizontal(&transposed, result_table, score_table);
    (result.transpose(), score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> MoveTables {
        MoveTables::new()
    }

    #[test]
    fn move_left_slides_tiles_left() {
        let tables = tables();
        let mut board = Board::new();
        // Row 0: [0, 0, 1, 1] → [2, 0, 0, 0]
        board.set_tile(0, 2, 1);
        board.set_tile(0, 3, 1);

        let (result, score) = apply_move(&board, Direction::Left, &tables).unwrap();
        assert_eq!(result.get_tile(0, 0), 2);
        assert_eq!(result.get_tile(0, 1), 0);
        assert_eq!(result.get_tile(0, 2), 0);
        assert_eq!(result.get_tile(0, 3), 0);
        assert_eq!(score, 4);
    }

    #[test]
    fn move_right_slides_tiles_right() {
        let tables = tables();
        let mut board = Board::new();
        // Row 0: [1, 1, 0, 0] → [0, 0, 0, 2]
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 1);

        let (result, score) = apply_move(&board, Direction::Right, &tables).unwrap();
        assert_eq!(result.get_tile(0, 0), 0);
        assert_eq!(result.get_tile(0, 3), 2);
        assert_eq!(score, 4);
    }

    #[test]
    fn move_up_slides_tiles_up() {
        let tables = tables();
        let mut board = Board::new();
        // Column 0: rows [0,0,1,1] → [2,0,0,0]
        board.set_tile(2, 0, 1);
        board.set_tile(3, 0, 1);

        let (result, score) = apply_move(&board, Direction::Up, &tables).unwrap();
        assert_eq!(result.get_tile(0, 0), 2);
        assert_eq!(result.get_tile(1, 0), 0);
        assert_eq!(result.get_tile(2, 0), 0);
        assert_eq!(result.get_tile(3, 0), 0);
        assert_eq!(score, 4);
    }

    #[test]
    fn move_down_slides_tiles_down() {
        let tables = tables();
        let mut board = Board::new();
        // Column 0: rows [1,1,0,0] → [0,0,0,2]
        board.set_tile(0, 0, 1);
        board.set_tile(1, 0, 1);

        let (result, score) = apply_move(&board, Direction::Down, &tables).unwrap();
        assert_eq!(result.get_tile(0, 0), 0);
        assert_eq!(result.get_tile(3, 0), 2);
        assert_eq!(score, 4);
    }

    #[test]
    fn move_returns_none_when_board_unchanged() {
        let tables = tables();
        let mut board = Board::new();
        // [1, 2, 0, 0] — already packed left, can't slide further
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 2);

        assert!(apply_move(&board, Direction::Left, &tables).is_none());
    }

    #[test]
    fn move_sums_score_across_all_rows() {
        let tables = tables();
        let mut board = Board::new();
        // Two rows each with a merge
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 1); // merge → score 4
        board.set_tile(1, 0, 2);
        board.set_tile(1, 1, 2); // merge → score 8

        let (_, score) = apply_move(&board, Direction::Left, &tables).unwrap();
        assert_eq!(score, 12);
    }
}
