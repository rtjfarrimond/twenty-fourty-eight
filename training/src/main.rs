use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use game_engine::MoveTables;
use serde::Serialize;
use training::eval;
use training::ntuple::NTupleNetwork;
use training::training::train_one_game;

/// Train an n-tuple network for 2048 using TD(0) with afterstate values.
#[derive(Parser)]
#[command(name = "training")]
struct Args {
    /// Number of training games to play
    #[arg(long, default_value_t = 100_000)]
    games: u32,

    /// Evaluate every N games
    #[arg(long, default_value_t = 10_000)]
    eval_interval: u32,

    /// Number of games per evaluation checkpoint
    #[arg(long, default_value_t = 1000)]
    eval_games: u32,

    /// Name for the model artefact (used in filenames)
    #[arg(long, default_value = "ntuple-4x6-td0")]
    model_name: String,

    /// Optimistic weight initialization value (0.0 = no optimistic init)
    #[arg(long, default_value_t = 0.0)]
    optimistic_init: f32,

    /// Pattern preset: "4x6" (4 base 6-tuples) or "8x6" (8 base 6-tuples)
    #[arg(long, default_value = "4x6")]
    patterns: String,

    /// Learning rate for TD(0) updates
    #[arg(long, default_value_t = 0.0025)]
    learning_rate: f32,

    /// Directory to deploy the trained model into. When set, the .bin and
    /// .meta.toml are copied here and models.json is regenerated.
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Human-readable description for the .meta.toml sidecar
    #[arg(long)]
    description: Option<String>,
}

/// Standard 4-pattern 6-tuple configuration from the literature.
fn patterns_4x6() -> Vec<Vec<(usize, usize)>> {
    vec![
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    ]
}

/// Stronger 8-pattern 6-tuple configuration (adds 4 more L/rectangle shapes).
/// From RESEARCH.md — flat indices converted to (row, col).
fn patterns_8x6() -> Vec<Vec<(usize, usize)>> {
    let mut patterns = patterns_4x6();
    patterns.extend(vec![
        vec![(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
        vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 1)],
        vec![(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 1)],
    ]);
    patterns
}

fn select_patterns(preset: &str) -> Vec<Vec<(usize, usize)>> {
    match preset {
        "4x6" => patterns_4x6(),
        "8x6" => patterns_8x6(),
        other => panic!("Unknown pattern preset: {other}. Expected 4x6 or 8x6."),
    }
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
    let args = Args::parse();

    let log_path = format!("{}.log.jsonl", args.model_name);
    let config_path = format!("{}.config.json", args.model_name);

    let patterns = select_patterns(&args.patterns);

    let config = TrainingConfig {
        model_name: args.model_name.clone(),
        model_type: "n-tuple".to_string(),
        total_games: args.games,
        eval_interval: args.eval_interval,
        eval_games: args.eval_games,
        learning_rate: args.learning_rate,
        num_base_patterns: patterns.len(),
        num_total_patterns: patterns.len() * 8,
    };

    let config_json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(&config_path, &config_json).expect("Failed to write config");

    println!("Training config:");
    println!("  Model: {}", args.model_name);
    println!("  Games: {}", args.games);
    println!("  Eval interval: {}", args.eval_interval);
    println!("  Eval games per checkpoint: {}", args.eval_games);
    println!("  Learning rate: {}", args.learning_rate);
    println!("  Optimistic init: {}", args.optimistic_init);
    println!(
        "  Patterns: {} ({} base x 8 symmetries = {})",
        args.patterns,
        patterns.len(),
        patterns.len() * 8
    );
    println!("  Log file: {log_path}");
    println!("  Config file: {config_path}");
    if let Some(ref dir) = args.models_dir {
        println!("  Deploy to: {}", dir.display());
    }
    println!();

    let tables = MoveTables::new();
    let mut network =
        NTupleNetwork::with_symmetry_expansion(&patterns, args.optimistic_init);
    let mut rng = rand::rng();

    let log_file = File::create(&log_path).expect("Failed to create log file");
    let mut log_writer = BufWriter::new(log_file);

    for game in 1..=args.games {
        train_one_game(&mut network, &tables, args.learning_rate, &mut rng);

        if game % args.eval_interval == 0 {
            let result = eval::evaluate(&network, &tables, args.eval_games, game);

            let json = serde_json::to_string(&result).unwrap();
            writeln!(log_writer, "{json}").expect("Failed to write log");
            log_writer.flush().expect("Failed to flush log");

            println!(
                "Games {game}/{}: avg {:.0}, max {}, \
                 2048 {:.1}%, 4096 {:.1}%, 8192 {:.1}%",
                args.games,
                result.avg_score,
                result.max_score,
                result.tile_2048_pct,
                result.tile_4096_pct,
                result.tile_8192_pct,
            );
        }
    }

    let model_path = format!("{}.bin", args.model_name);
    network.save(&model_path).expect("Failed to save model");
    println!("\nTraining complete.");
    println!("  Log: {log_path}");
    println!("  Model: {model_path}");

    if let Some(ref models_dir) = args.models_dir {
        deploy_model(&args, models_dir, &model_path, &log_path, &config_path);
    }
}

/// Copies a file, but skips if source and destination resolve to the same path
/// (which would wipe the file via truncate-before-read).
fn copy_if_different(source: &str, destination: &PathBuf) {
    let source_canonical = std::fs::canonicalize(source).ok();
    let dest_canonical = std::fs::canonicalize(destination).ok();

    if source_canonical.is_some() && source_canonical == dest_canonical {
        println!("  Skipping copy (already in place): {source}");
        return;
    }

    std::fs::copy(source, destination).expect("Failed to copy file");
    println!("  Copied {source} → {}", destination.display());
}

fn deploy_model(
    args: &Args,
    models_dir: &PathBuf,
    model_path: &str,
    log_path: &str,
    config_path: &str,
) {
    println!("\nDeploying to {}...", models_dir.display());

    let dest_bin = models_dir.join(format!("{}.bin", args.model_name));
    copy_if_different(model_path, &dest_bin);

    let description = args.description.clone().unwrap_or_else(|| {
        format!(
            "N-tuple network ({} preset) trained with TD(0). \
             {} training games, lr={}, optimistic_init={}.",
            args.patterns, args.games, args.learning_rate, args.optimistic_init
        )
    });

    let meta_path = models_dir.join(format!("{}.meta.toml", args.model_name));
    let meta_content = format!(
        "name = \"{}\"\ndescription = \"{}\"\n",
        args.model_name, description
    );
    std::fs::write(&meta_path, &meta_content).expect("Failed to write meta.toml");
    println!("  Wrote {}", meta_path.display());

    // Copy training logs to a training/ subdir alongside models if it exists
    let training_dir = models_dir.join("../training");
    if training_dir.is_dir() {
        copy_if_different(log_path, &training_dir.join(log_path));
        copy_if_different(config_path, &training_dir.join(config_path));
    }

    // Regenerate models.json — look for generate_models next to this binary
    let self_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let generate_models_path = self_dir
        .map(|d| d.join("generate_models"))
        .filter(|p| p.is_file());

    if let Some(generate_models) = generate_models_path {
        println!("  Regenerating models.json...");
        let status = std::process::Command::new(&generate_models)
            .current_dir(models_dir)
            .status()
            .expect("Failed to run generate_models");
        if status.success() {
            println!("  models.json regenerated.");
        } else {
            eprintln!("  WARNING: generate_models exited with {status}");
        }
    } else {
        println!("  Skipping models.json regeneration (generate_models not found)");
    }

    println!("Deploy complete.");
}
