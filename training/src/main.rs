use std::fs::File;
use std::io::{BufWriter, Write};

use game_engine::MoveTables;
use serde::Serialize;
use training::eval;
use training::ntuple::NTupleNetwork;
use training::training::train_one_game;

/// Standard 4-pattern 6-tuple configuration from the literature.
fn standard_6tuple_patterns() -> Vec<Vec<(usize, usize)>> {
    vec![
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    ]
}

/// Written alongside the JSONL log so the dashboard knows the training config.
#[derive(Serialize)]
struct TrainingConfig {
    model_name: String,
    model_type: String,
    total_games: u32,
    eval_interval: u32,
    eval_games: u32,
    learning_rate: f32,
    num_base_patterns: usize,
    num_total_patterns: usize,
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

    let model_name = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "ntuple-4x6-td0".to_string());

    let eval_games: u32 = 1000;
    let learning_rate: f32 = 0.0025;
    let log_path = "training_log.jsonl";
    let config_path = "training_config.json";

    let patterns = standard_6tuple_patterns();

    let config = TrainingConfig {
        model_name: model_name.clone(),
        model_type: "n-tuple".to_string(),
        total_games: num_games,
        eval_interval,
        eval_games,
        learning_rate,
        num_base_patterns: patterns.len(),
        num_total_patterns: patterns.len() * 8,
    };

    let config_json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(config_path, &config_json).expect("Failed to write config");

    println!("Training config:");
    println!("  Model: {model_name}");
    println!("  Games: {num_games}");
    println!("  Eval interval: {eval_interval}");
    println!("  Eval games per checkpoint: {eval_games}");
    println!("  Learning rate: {learning_rate}");
    println!(
        "  Patterns: {} base x 8 symmetries = {}",
        patterns.len(),
        patterns.len() * 8
    );
    println!("  Log file: {log_path}");
    println!("  Config file: {config_path}");
    println!();

    let tables = MoveTables::new();
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
                "Games {game}/{num_games}: avg {:.0}, max {}, \
                 2048 {:.1}%, 4096 {:.1}%, 8192 {:.1}%",
                result.avg_score,
                result.max_score,
                result.tile_2048_pct,
                result.tile_4096_pct,
                result.tile_8192_pct,
            );
        }
    }

    let model_path = format!("{model_name}.bin");
    network.save(&model_path).expect("Failed to save model");
    println!("\nTraining complete.");
    println!("  Log: {log_path}");
    println!("  Model: {model_path}");
}
