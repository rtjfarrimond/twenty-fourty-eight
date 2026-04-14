use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use game_engine::MoveTables;
use serde::Serialize;
use training::config::{select_patterns, validate_algorithm};
use training::eval;
use training::ntuple::NTupleNetwork;
use training::training::{train_hogwild_batch, train_one_game};

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

    /// Training algorithm: "serial" (single-threaded) or "hogwild" (lock-free
    /// shared-memory parallelism). Hogwild not yet implemented.
    #[arg(long, default_value = "serial")]
    algorithm: String,

    /// Number of worker threads. Must be 1 for the serial algorithm.
    #[arg(long, default_value_t = 1)]
    threads: u32,
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
    algorithm: String,
    num_threads: u32,
}

fn main() {
    let args = Args::parse();

    validate_algorithm(&args.algorithm, args.threads);

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
        algorithm: args.algorithm.clone(),
        num_threads: args.threads,
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
    println!("  Algorithm: {} ({} thread(s))", args.algorithm, args.threads);
    println!("  Log file: {log_path}");
    println!("  Config file: {config_path}");
    if let Some(ref dir) = args.models_dir {
        println!("  Deploy to: {}", dir.display());
    }
    println!();

    let tables = MoveTables::new();
    let network = NTupleNetwork::with_symmetry_expansion(&patterns, args.optimistic_init);

    let log_file = File::create(&log_path).expect("Failed to create log file");
    let mut log_writer = BufWriter::new(log_file);

    run_training(&network, &tables, &args, &mut log_writer);

    let model_path = format!("{}.bin", args.model_name);
    network.save(&model_path).expect("Failed to save model");
    println!("\nTraining complete.");
    println!("  Log: {log_path}");
    println!("  Model: {model_path}");

    if let Some(ref models_dir) = args.models_dir {
        deploy_model(&args, models_dir, &model_path, &log_path, &config_path);
    }
}

/// Dispatches to the configured training algorithm. Both paths share the
/// same eval/logging cadence — the difference is whether each chunk of
/// `eval_interval` games is played sequentially on one thread or distributed
/// across `threads` worker threads via Hogwild.
fn run_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &Args,
    log_writer: &mut BufWriter<File>,
) {
    match args.algorithm.as_str() {
        "serial" => run_serial_training(network, tables, args, log_writer),
        "hogwild" => run_hogwild_training(network, tables, args, log_writer),
        other => panic!("Unreachable: validate_algorithm should reject {other}"),
    }
}

fn run_serial_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &Args,
    log_writer: &mut BufWriter<File>,
) {
    let mut rng = rand::rng();
    for game in 1..=args.games {
        train_one_game(network, tables, args.learning_rate, &mut rng);
        if game % args.eval_interval == 0 {
            log_eval_checkpoint(network, tables, args, game, log_writer);
        }
    }
}

fn run_hogwild_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &Args,
    log_writer: &mut BufWriter<File>,
) {
    let mut games_completed: u32 = 0;
    while games_completed < args.games {
        let remaining = args.games - games_completed;
        let batch_size = remaining.min(args.eval_interval);
        let batch_seed = 0xC0FFEE_u64 ^ (games_completed as u64);
        train_hogwild_batch(
            network,
            tables,
            args.learning_rate,
            args.threads,
            batch_size,
            batch_seed,
        );
        games_completed += batch_size;
        log_eval_checkpoint(network, tables, args, games_completed, log_writer);
    }
}

fn log_eval_checkpoint(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &Args,
    games_trained: u32,
    log_writer: &mut BufWriter<File>,
) {
    let result = eval::evaluate(network, tables, args.eval_games, games_trained);
    let json = serde_json::to_string(&result).unwrap();
    writeln!(log_writer, "{json}").expect("Failed to write log");
    log_writer.flush().expect("Failed to flush log");
    println!(
        "Games {games_trained}/{}: avg {:.0}, max {}, \
         2048 {:.1}%, 4096 {:.1}%, 8192 {:.1}%",
        args.games,
        result.avg_score,
        result.max_score,
        result.tile_2048_pct,
        result.tile_4096_pct,
        result.tile_8192_pct,
    );
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
            "N-tuple network ({} preset) trained with TD(0) via {} algorithm \
             ({} thread(s)). {} training games, lr={}, optimistic_init={}.",
            args.patterns,
            args.algorithm,
            args.threads,
            args.games,
            args.learning_rate,
            args.optimistic_init
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
