//! In-process execution of a single training run from a `RunArgs`.
//!
//! Extracted from `main.rs` so that the `run` CLI subcommand and any
//! programmatic caller (tests, future embedded use cases) share one path.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use game_engine::MoveTables;

use crate::config::select_patterns;
use crate::eval;
use crate::ntuple::NTupleNetwork;
use crate::run_args::RunArgs;
use crate::tc_state::TcState;
use crate::training::{train_hogwild_batch, train_one_game, train_one_game_tc, train_tc_hogwild_batch};

/// Execute a training run end-to-end: print config banner, train, save
/// model, optionally deploy. Returns `Ok` on success, `Err` with a message
/// on any I/O or training failure.
pub fn execute(args: &RunArgs) -> Result<(), String> {
    let log_path = format!("{}.log.jsonl", args.model_name);
    let config_path = format!("{}.config.json", args.model_name);

    let patterns = select_patterns(&args.patterns);
    write_config_sidecar(args, &patterns, &config_path)?;
    print_banner(args, &patterns, &log_path, &config_path);

    let tables = MoveTables::new();
    let network = build_network(args, &patterns);

    let log_file = File::create(&log_path)
        .map_err(|err| format!("Failed to create {log_path}: {err}"))?;
    let mut log_writer = BufWriter::new(log_file);
    run_training(&network, &tables, args, &mut log_writer);

    let model_path = format!("{}.bin", args.model_name);
    network.save(&model_path)
        .map_err(|err| format!("Failed to save model: {err}"))?;
    println!("\nTraining complete.");

    deploy_model(args, &args.models_dir, &model_path, &log_path, &config_path);
    Ok(())
}

fn build_network(
    args: &RunArgs,
    patterns: &[Vec<(usize, usize)>],
) -> NTupleNetwork {
    if args.random_init_amplitude > 0.0 {
        NTupleNetwork::with_random_init(
            patterns,
            args.random_init_amplitude,
            args.random_init_seed,
        )
    } else {
        NTupleNetwork::with_symmetry_expansion(patterns, args.optimistic_init)
    }
}

#[derive(serde::Serialize)]
struct TrainingConfig<'a> {
    model_name: &'a str,
    model_type: &'a str,
    total_games: u32,
    eval_interval: u32,
    eval_games: u32,
    learning_rate: f32,
    num_base_patterns: usize,
    num_total_patterns: usize,
    algorithm: &'a str,
    num_threads: u32,
}

fn write_config_sidecar(
    args: &RunArgs,
    patterns: &[Vec<(usize, usize)>],
    config_path: &str,
) -> Result<(), String> {
    let config = TrainingConfig {
        model_name: &args.model_name,
        model_type: "n-tuple",
        total_games: args.games,
        eval_interval: args.eval_interval,
        eval_games: args.eval_games,
        learning_rate: args.learning_rate,
        num_base_patterns: patterns.len(),
        num_total_patterns: patterns.len() * 8,
        algorithm: &args.algorithm,
        num_threads: args.threads,
    };
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|err| format!("Failed to serialize config: {err}"))?;
    std::fs::write(config_path, &config_json)
        .map_err(|err| format!("Failed to write {config_path}: {err}"))
}

fn print_banner(
    args: &RunArgs,
    patterns: &[Vec<(usize, usize)>],
    log_path: &str,
    config_path: &str,
) {
    println!("Training config:");
    println!("  Model: {}", args.model_name);
    println!("  Games: {}", args.games);
    println!("  Eval interval: {}", args.eval_interval);
    println!("  Eval games per checkpoint: {}", args.eval_games);
    let rate_label = if args.algorithm.starts_with("tc") {
        "Beta (TC meta-learning rate)"
    } else {
        "Learning rate"
    };
    println!("  {rate_label}: {}", args.learning_rate);
    println!("  Optimistic init: {}", args.optimistic_init);
    println!(
        "  Random init: amplitude={}, seed={}",
        args.random_init_amplitude, args.random_init_seed
    );
    println!(
        "  Patterns: {} ({} base x 8 symmetries = {})",
        args.patterns,
        patterns.len(),
        patterns.len() * 8
    );
    println!("  Algorithm: {} ({} thread(s))", args.algorithm, args.threads);
    println!("  Log file: {log_path}");
    println!("  Config file: {config_path}");
    println!("  Deploy to: {}", args.models_dir.display());
    println!();
}

fn run_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &RunArgs,
    log_writer: &mut BufWriter<File>,
) {
    match args.algorithm.as_str() {
        "serial" => run_serial_training(network, tables, args, log_writer),
        "hogwild" => run_hogwild_training(network, tables, args, log_writer),
        "tc" => run_tc_serial_training(network, tables, args, log_writer),
        "tc-hogwild" => run_tc_hogwild_training(network, tables, args, log_writer),
        other => panic!("Unreachable: validate_algorithm should reject {other}"),
    }
}

fn run_serial_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &RunArgs,
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
    args: &RunArgs,
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

fn run_tc_serial_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &RunArgs,
    log_writer: &mut BufWriter<File>,
) {
    let tc_state = TcState::new(network.num_weights());
    let mut rng = rand::rng();
    for game in 1..=args.games {
        train_one_game_tc(network, &tc_state, tables, args.learning_rate, &mut rng);
        if game % args.eval_interval == 0 {
            log_eval_checkpoint(network, tables, args, game, log_writer);
        }
    }
}

fn run_tc_hogwild_training(
    network: &NTupleNetwork,
    tables: &MoveTables,
    args: &RunArgs,
    log_writer: &mut BufWriter<File>,
) {
    let tc_state = TcState::new(network.num_weights());
    let mut games_completed: u32 = 0;
    while games_completed < args.games {
        let remaining = args.games - games_completed;
        let batch_size = remaining.min(args.eval_interval);
        let batch_seed = 0xC0FFEE_u64 ^ (games_completed as u64);
        train_tc_hogwild_batch(
            network,
            &tc_state,
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
    args: &RunArgs,
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

/// Moves a file to the destination, skipping if already in place.
fn move_to(source: &str, destination: &PathBuf) {
    let source_canonical = std::fs::canonicalize(source).ok();
    let dest_canonical = std::fs::canonicalize(destination).ok();

    if source_canonical.is_some() && source_canonical == dest_canonical {
        println!("  Already in place: {source}");
        return;
    }

    if let Err(err) = std::fs::rename(source, destination) {
        // rename fails across filesystems; fall back to copy + remove.
        std::fs::copy(source, destination)
            .unwrap_or_else(|_| panic!("Failed to copy {source}: {err}"));
        std::fs::remove_file(source).ok();
    }
    println!("  Moved {source} → {}", destination.display());
}

fn deploy_model(
    args: &RunArgs,
    models_dir: &PathBuf,
    model_path: &str,
    log_path: &str,
    config_path: &str,
) {
    println!("\nDeploying to {}...", models_dir.display());

    move_to(model_path, &models_dir.join(format!("{}.bin", args.model_name)));
    move_to(log_path, &models_dir.join(log_path));
    move_to(config_path, &models_dir.join(config_path));

    let description = args.description.clone().unwrap_or_else(|| {
        format!(
            "N-tuple network ({} preset) trained via {} algorithm \
             ({} thread(s)). {} training games, lr={}, optimistic_init={}, \
             random_init_amplitude={} (seed={}).",
            args.patterns,
            args.algorithm,
            args.threads,
            args.games,
            args.learning_rate,
            args.optimistic_init,
            args.random_init_amplitude,
            args.random_init_seed,
        )
    });

    let meta_path = models_dir.join(format!("{}.meta.toml", args.model_name));
    let meta_content = format!(
        "name = \"{}\"\ndescription = \"{}\"\n",
        args.model_name, description
    );
    std::fs::write(&meta_path, &meta_content).expect("Failed to write meta.toml");
    println!("  Wrote {}", meta_path.display());

    run_generate_models(models_dir);
    println!("Deploy complete.");
}

fn run_generate_models(models_dir: &PathBuf) {
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
}
