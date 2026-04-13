pub mod dummy;
pub mod ntuple_agent;

use game_engine::{Board, Direction};

/// The stable interface between model implementations and consumers (server).
/// Any model — n-tuple network, neural net, or a simple heuristic — implements
/// this trait to provide move selection and board evaluation.
pub trait Agent {
    /// Returns the best move for the given board state.
    fn best_move(&self, board: &Board) -> Direction;

    /// Returns the estimated value of the given board state.
    fn evaluate(&self, board: &Board) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantAgent;

    impl Agent for ConstantAgent {
        fn best_move(&self, _board: &Board) -> Direction {
            Direction::Down
        }

        fn evaluate(&self, _board: &Board) -> f64 {
            42.0
        }
    }

    #[test]
    fn agent_trait_is_implementable() {
        let agent = ConstantAgent;
        let board = Board::new();
        assert_eq!(agent.best_move(&board), Direction::Down);
        assert_eq!(agent.evaluate(&board), 42.0);
    }
}
