use game_engine::{Board, Direction, MoveTables, legal_moves};

use crate::Agent;

/// A trivial agent that picks moves in a fixed priority order:
/// Down, Right, Left, Up. This pushes tiles toward the bottom-right
/// corner, which is a crude but visible strategy.
pub struct DummyAgent {
    tables: std::sync::Arc<MoveTables>,
}

const PRIORITY_ORDER: [Direction; 4] = [
    Direction::Down,
    Direction::Right,
    Direction::Left,
    Direction::Up,
];

impl DummyAgent {
    pub fn new(tables: std::sync::Arc<MoveTables>) -> Self {
        Self { tables }
    }
}

impl Agent for DummyAgent {
    fn name(&self) -> &str {
        "heuristic-down-right"
    }

    fn description(&self) -> &str {
        "A simple heuristic agent that prefers Down and Right moves, \
         pushing tiles toward the bottom-right corner. No learning — \
         serves as a baseline for comparison."
    }

    fn best_move(&self, board: &Board) -> Direction {
        let available = legal_moves(board, &self.tables);
        PRIORITY_ORDER
            .iter()
            .find(|direction| available.contains(direction))
            .copied()
            .unwrap_or(Direction::Down)
    }

    fn evaluate(&self, _board: &Board) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tables() -> Arc<MoveTables> {
        Arc::new(MoveTables::new())
    }

    #[test]
    fn dummy_agent_prefers_down() {
        let agent = DummyAgent::new(tables());
        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        // Down and Right are both legal, should pick Down
        assert_eq!(agent.best_move(&board), Direction::Down);
    }

    #[test]
    fn dummy_agent_falls_back_when_preferred_blocked() {
        let agent = DummyAgent::new(tables());
        let mut board = Board::new();
        // Tile at bottom-right corner: can only go Left or Up
        board.set_tile(3, 3, 1);
        let chosen = agent.best_move(&board);
        assert_eq!(chosen, Direction::Left);
    }

    #[test]
    fn dummy_agent_evaluate_returns_zero() {
        let agent = DummyAgent::new(tables());
        assert_eq!(agent.evaluate(&Board::new()), 0.0);
    }
}
