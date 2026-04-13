use game_engine::{
    Board, Direction, MoveTables, apply_move, empty_tiles, is_game_over, spawn_tile,
};
use rand::Rng;
use serde::Serialize;

const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Left,
    Direction::Right,
    Direction::Up,
    Direction::Down,
];

const PRIORITY_ORDER: [Direction; 4] = [
    Direction::Down,
    Direction::Right,
    Direction::Left,
    Direction::Up,
];

/// Eval result for a non-trained model (single data point, no training curve).
#[derive(Serialize)]
pub struct ModelEntry {
    pub model_name: String,
    pub model_type: String,
    pub description: String,
    pub eval: crate::eval::EvalResult,
}

fn dummy_best_move(board: &Board, tables: &MoveTables) -> Option<Direction> {
    let available: Vec<Direction> = ALL_DIRECTIONS
        .iter()
        .copied()
        .filter(|&dir| apply_move(board, dir, tables).is_some())
        .collect();

    PRIORITY_ORDER
        .iter()
        .find(|dir| available.contains(dir))
        .copied()
}

fn play_dummy_game(tables: &MoveTables, rng: &mut impl Rng) -> (u32, u8) {
    let mut board = Board::new();
    for _ in 0..2 {
        let empties = empty_tiles(&board);
        let pos = empties[rng.random_range(0..empties.len())];
        board = spawn_tile(&board, pos.0, pos.1, rng.random::<f64>());
    }

    let mut score: u32 = 0;

    while !is_game_over(&board, tables) {
        let Some(direction) = dummy_best_move(&board, tables) else {
            break;
        };

        if let Some((afterstate, reward)) = apply_move(&board, direction, tables) {
            score += reward;
            let empties = empty_tiles(&afterstate);
            if empties.is_empty() {
                board = afterstate;
            } else {
                let pos = empties[rng.random_range(0..empties.len())];
                board = spawn_tile(&afterstate, pos.0, pos.1, rng.random::<f64>());
            }
        }
    }

    let mut max_exp = 0u8;
    for row in 0..4 {
        for col in 0..4 {
            let exp = board.get_tile(row, col);
            if exp > max_exp {
                max_exp = exp;
            }
        }
    }

    (score, max_exp)
}

pub fn evaluate_dummy(tables: &MoveTables, num_games: u32) -> crate::eval::EvalResult {
    let mut rng = rand::rng();
    let mut scores = Vec::with_capacity(num_games as usize);
    let mut tile_counts = [0u32; 5];

    for _ in 0..num_games {
        let (score, max_exp) = play_dummy_game(tables, &mut rng);
        scores.push(score);

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
    let denominator = num_games as f64;

    crate::eval::EvalResult {
        games_trained: 0,
        avg_score: total as f64 / denominator,
        max_score: *scores.iter().max().unwrap_or(&0),
        tile_2048_pct: tile_counts[0] as f64 / denominator * 100.0,
        tile_4096_pct: tile_counts[1] as f64 / denominator * 100.0,
        tile_8192_pct: tile_counts[2] as f64 / denominator * 100.0,
        tile_16384_pct: tile_counts[3] as f64 / denominator * 100.0,
        tile_32768_pct: tile_counts[4] as f64 / denominator * 100.0,
    }
}
