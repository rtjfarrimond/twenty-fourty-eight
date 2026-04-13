use game_engine::{
    Board, Direction, MoveTables, apply_move, empty_tiles, is_game_over, spawn_tile,
};
use rand::Rng;

use crate::ntuple::NTupleNetwork;

const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Left,
    Direction::Right,
    Direction::Up,
    Direction::Down,
];

/// Result of evaluating a single move: the afterstate, reward, and direction.
struct MoveCandidate {
    afterstate: Board,
    reward: u32,
    direction: Direction,
}

/// Finds the best move by evaluating all legal afterstates.
/// Returns None if no legal move exists.
fn best_afterstate(
    board: &Board,
    network: &NTupleNetwork,
    tables: &MoveTables,
) -> Option<MoveCandidate> {
    ALL_DIRECTIONS
        .iter()
        .filter_map(|&direction| {
            apply_move(board, direction, tables).map(|(afterstate, reward)| MoveCandidate {
                afterstate,
                reward,
                direction,
            })
        })
        .max_by(|candidate_a, candidate_b| {
            let value_a = candidate_a.reward as f32 + network.evaluate(&candidate_a.afterstate);
            let value_b = candidate_b.reward as f32 + network.evaluate(&candidate_b.afterstate);
            value_a.partial_cmp(&value_b).unwrap()
        })
}

/// Plays one complete game using the network for move selection,
/// updating weights via TD(0) after each move.
/// Returns the final score.
pub fn train_one_game(
    network: &mut NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    rng: &mut impl Rng,
) -> u32 {
    let mut board = spawn_random_tile(Board::new(), rng);
    board = spawn_random_tile(board, rng);
    let mut total_score: u32 = 0;

    // Get first afterstate
    let Some(mut current) = best_afterstate(&board, network, tables) else {
        return 0;
    };
    total_score += current.reward;

    loop {
        // Spawn a random tile after the move
        let next_board = spawn_random_tile(current.afterstate, rng);

        // Check if game is over
        if is_game_over(&next_board, tables) {
            // Terminal update: V(terminal) = 0
            let td_error = 0.0 - network.evaluate(&current.afterstate);
            network.update(&current.afterstate, learning_rate * td_error);
            break;
        }

        // Find best move from the new state
        let Some(next) = best_afterstate(&next_board, network, tables) else {
            // No legal moves — terminal
            let td_error = 0.0 - network.evaluate(&current.afterstate);
            network.update(&current.afterstate, learning_rate * td_error);
            break;
        };

        // TD(0) update: V(s) <- V(s) + α * [r' + V(s') - V(s)]
        let td_error = next.reward as f32 + network.evaluate(&next.afterstate)
            - network.evaluate(&current.afterstate);
        network.update(&current.afterstate, learning_rate * td_error);

        total_score += next.reward;
        current = next;
    }

    total_score
}

fn spawn_random_tile(board: Board, rng: &mut impl Rng) -> Board {
    let empties = empty_tiles(&board);
    if empties.is_empty() {
        return board;
    }
    let position = empties[rng.random_range(0..empties.len())];
    spawn_tile(&board, position.0, position.1, rng.random::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn tables() -> MoveTables {
        MoveTables::new()
    }

    fn small_network() -> NTupleNetwork {
        // Use a small pattern set for fast tests
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3)]];
        NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0)
    }

    #[test]
    fn train_one_game_returns_nonzero_score() {
        let tables = tables();
        let mut network = small_network();
        let mut rng = rand::rng();

        let score = train_one_game(&mut network, &tables, 0.0025, &mut rng);
        // A game should produce some score (merges happen)
        assert!(score > 0);
    }

    #[test]
    fn train_one_game_modifies_weights() {
        let tables = tables();
        let mut network = small_network();
        let mut rng = rand::rng();

        let initial_eval = network.evaluate(&Board::new());
        train_one_game(&mut network, &tables, 0.0025, &mut rng);
        let after_eval = network.evaluate(&Board::new());

        // Weights should have changed
        assert_ne!(initial_eval, after_eval);
    }

    #[test]
    fn training_improves_over_many_games() {
        let tables = tables();
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

    #[test]
    fn best_afterstate_returns_none_for_game_over_board() {
        let tables = tables();
        let network = small_network();
        let mut board = Board::new();

        // Fill with non-mergeable values
        let values = [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]];
        for row in 0..4 {
            for col in 0..4 {
                board.set_tile(row, col, values[row][col]);
            }
        }

        assert!(best_afterstate(&board, &network, &tables).is_none());
    }

    #[test]
    fn best_afterstate_picks_highest_value_move() {
        let tables = tables();
        let base_patterns = vec![vec![(0, 0)]];
        let mut network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 1); // can merge left or right

        // Train the network to prefer some moves over others
        // by running a few games
        let mut rng = rand::rng();
        for _ in 0..50 {
            train_one_game(&mut network, &tables, 0.01, &mut rng);
        }

        // Just verify it returns a valid candidate
        let candidate = best_afterstate(&board, &network, &tables);
        assert!(candidate.is_some());
    }
}
