use game_engine::{
    Board, Direction, MoveTables, apply_move, empty_tiles, is_game_over, spawn_tile,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::ntuple::NTupleNetwork;

const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Left,
    Direction::Right,
    Direction::Up,
    Direction::Down,
];

/// A single evaluation log entry, written as one JSON line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub games_trained: u32,
    pub avg_score: f64,
    pub max_score: u32,
    pub tile_2048_pct: f64,
    pub tile_4096_pct: f64,
    pub tile_8192_pct: f64,
    pub tile_16384_pct: f64,
    pub tile_32768_pct: f64,
}

/// Finds the highest tile exponent on a board.
fn max_tile_exponent(board: &Board) -> u8 {
    let mut max_exp = 0;
    for row in 0..4 {
        for col in 0..4 {
            let exp = board.get_tile(row, col);
            if exp > max_exp {
                max_exp = exp;
            }
        }
    }
    max_exp
}

/// Plays one game greedily using the network (no learning).
/// Returns (score, max_tile_exponent).
fn play_eval_game(network: &NTupleNetwork, tables: &MoveTables, rng: &mut impl Rng) -> (u32, u8) {
    let mut board = Board::new();
    let empties = empty_tiles(&board);
    let pos = empties[rng.random_range(0..empties.len())];
    board = spawn_tile(&board, pos.0, pos.1, rng.random::<f64>());
    let empties = empty_tiles(&board);
    let pos = empties[rng.random_range(0..empties.len())];
    board = spawn_tile(&board, pos.0, pos.1, rng.random::<f64>());

    let mut score: u32 = 0;

    while !is_game_over(&board, tables) {
        let best = ALL_DIRECTIONS
            .iter()
            .filter_map(|&direction| {
                apply_move(&board, direction, tables)
                    .map(|(afterstate, reward)| (afterstate, reward))
            })
            .max_by(|(afterstate_a, reward_a), (afterstate_b, reward_b)| {
                let value_a = *reward_a as f32 + network.evaluate(afterstate_a);
                let value_b = *reward_b as f32 + network.evaluate(afterstate_b);
                value_a.partial_cmp(&value_b).unwrap()
            });

        if let Some((afterstate, reward)) = best {
            score += reward;
            let empties = empty_tiles(&afterstate);
            if empties.is_empty() {
                board = afterstate;
            } else {
                let pos = empties[rng.random_range(0..empties.len())];
                board = spawn_tile(&afterstate, pos.0, pos.1, rng.random::<f64>());
            }
        } else {
            break;
        }
    }

    (score, max_tile_exponent(&board))
}

/// Runs a batch of evaluation games and returns the results.
pub fn evaluate(
    network: &NTupleNetwork,
    tables: &MoveTables,
    num_games: u32,
    games_trained: u32,
) -> EvalResult {
    let mut rng = rand::rng();
    let mut scores = Vec::with_capacity(num_games as usize);
    let mut tile_counts = [0u32; 5]; // 2048, 4096, 8192, 16384, 32768

    for _ in 0..num_games {
        let (score, max_exp) = play_eval_game(network, tables, &mut rng);
        scores.push(score);

        // 2048 = exponent 11, 4096 = 12, etc.
        if max_exp >= 11 {
            tile_counts[0] += 1;
        }
        if max_exp >= 12 {
            tile_counts[1] += 1;
        }
        if max_exp >= 13 {
            tile_counts[2] += 1;
        }
        if max_exp >= 14 {
            tile_counts[3] += 1;
        }
        if max_exp >= 15 {
            tile_counts[4] += 1;
        }
    }

    let total: u64 = scores.iter().map(|&s| s as u64).sum();
    let avg_score = total as f64 / num_games as f64;
    let max_score = *scores.iter().max().unwrap_or(&0);
    let denominator = num_games as f64;

    EvalResult {
        games_trained,
        avg_score,
        max_score,
        tile_2048_pct: tile_counts[0] as f64 / denominator * 100.0,
        tile_4096_pct: tile_counts[1] as f64 / denominator * 100.0,
        tile_8192_pct: tile_counts[2] as f64 / denominator * 100.0,
        tile_16384_pct: tile_counts[3] as f64 / denominator * 100.0,
        tile_32768_pct: tile_counts[4] as f64 / denominator * 100.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> MoveTables {
        MoveTables::new()
    }

    #[test]
    fn max_tile_exponent_finds_highest() {
        let mut board = Board::new();
        board.set_tile(1, 2, 5);
        board.set_tile(3, 0, 11);
        assert_eq!(max_tile_exponent(&board), 11);
    }

    #[test]
    fn play_eval_game_completes_and_returns_score() {
        let tables = tables();
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let mut rng = rand::rng();

        let (score, max_exp) = play_eval_game(&network, &tables, &mut rng);
        assert!(score > 0);
        assert!(max_exp >= 1);
    }

    #[test]
    fn evaluate_returns_valid_percentages() {
        let tables = tables();
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3)]];
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);

        let result = evaluate(&network, &tables, 10, 0);
        assert!(result.avg_score > 0.0);
        assert!(result.tile_2048_pct >= 0.0 && result.tile_2048_pct <= 100.0);
    }

    #[test]
    fn eval_result_serializes_to_json() {
        let result = EvalResult {
            games_trained: 1000,
            avg_score: 5000.0,
            max_score: 12000,
            tile_2048_pct: 50.0,
            tile_4096_pct: 10.0,
            tile_8192_pct: 0.0,
            tile_16384_pct: 0.0,
            tile_32768_pct: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"games_trained\":1000"));
        assert!(json.contains("\"tile_2048_pct\":50.0"));
    }
}
