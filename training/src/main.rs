mod eval;
mod ntuple;
mod training;

use std::fs::File;
use std::io::{BufWriter, Write};

use game_engine::MoveTables;
use ntuple::NTupleNetwork;
use training::train_one_game;

/// Standard 4-pattern 6-tuple configuration from the literature.
fn standard_6tuple_patterns() -> Vec<Vec<(usize, usize)>> {
    vec![
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    ]
}

fn main() {
    let num_games: u32 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);

    let eval_interval: u32 = std::env::args()
        .nth(2)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(10_000);

    let eval_games: u32 = 1000;
    let learning_rate: f32 = 0.0025;
    let log_path = "training_log.jsonl";

    println!("Training config:");
    println!("  Games: {num_games}");
    println!("  Eval interval: {eval_interval}");
    println!("  Eval games per checkpoint: {eval_games}");
    println!("  Learning rate: {learning_rate}");
    println!("  Patterns: 4 base x 8 symmetries = 32");
    println!("  Log file: {log_path}");
    println!();

    let tables = MoveTables::new();
    let patterns = standard_6tuple_patterns();
    let mut network = NTupleNetwork::with_symmetry_expansion(&patterns, 0.0);
    let mut rng = rand::rng();

    let log_file = File::create(log_path).expect("Failed to create log file");
    let mut log_writer = BufWriter::new(log_file);

    for game in 1..=num_games {
        train_one_game(&mut network, &tables, learning_rate, &mut rng);

        if game % eval_interval == 0 {
            let result = eval::evaluate(&network, &tables, eval_games, game);

            let json = serde_json::to_string(&result).unwrap();
            writeln!(log_writer, "{json}").expect("Failed to write log");
            log_writer.flush().expect("Failed to flush log");

            println!(
                "Games {game}: avg {:.0}, max {}, \
                 2048 {:.1}%, 4096 {:.1}%, 8192 {:.1}%",
                result.avg_score,
                result.max_score,
                result.tile_2048_pct,
                result.tile_4096_pct,
                result.tile_8192_pct,
            );
        }
    }

    println!("\nTraining complete. Log written to {log_path}");
}
