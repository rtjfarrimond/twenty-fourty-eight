//! Training throughput benchmark.
//!
//! Runs a fixed number of training games under a given (algorithm, threads)
//! configuration and reports games/sec, moves/sec, wall-clock seconds, peak
//! RSS, and the final averaged score. Outputs both human-readable stdout and
//! (optionally) a structured JSON file suitable for dashboard ingestion and
//! CI-driven regression tracking.
//!
//! Uses a seeded RNG so repeated runs with the same `--seed` are directly
//! comparable — required for convergence-parity checks and A/B benchmarking.

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use game_engine::MoveTables;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::Serialize;
use training::config::{select_patterns, validate_algorithm};
use training::ntuple::NTupleNetwork;
use training::training::{train_hogwild_batch, train_one_game};

/// Benchmark a training configuration and emit throughput + convergence
/// metrics.
#[derive(Parser)]
#[command(name = "benchmark")]
struct Args {
    /// Number of training games to measure.
    #[arg(long, default_value_t = 10_000)]
    games: u32,

    /// Number of warmup games before timing starts.
    #[arg(long, default_value_t = 100)]
    warmup_games: u32,

    /// Pattern preset: "4x6" or "8x6".
    #[arg(long, default_value = "4x6")]
    patterns: String,

    /// Training algorithm: "serial" or "hogwild".
    #[arg(long, default_value = "serial")]
    algorithm: String,

    /// Number of worker threads (serial requires 1).
    #[arg(long, default_value_t = 1)]
    threads: u32,

    /// Learning rate for TD(0) updates.
    #[arg(long, default_value_t = 0.0025)]
    learning_rate: f32,

    /// RNG seed — runs with the same seed produce identical training
    /// trajectories (for the same algorithm).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Optional JSON output path. When set, a structured result file is
    /// written in addition to human-readable stdout.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Optional label for this benchmark run (included in the JSON output).
    /// Useful for distinguishing A/B rows on the dashboard.
    #[arg(long)]
    label: Option<String>,
}

fn run_serial_bench(
    network: &NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    games: u32,
    seed: u64,
) -> (u64, u64) {
    // Offset the seed so the timed section uses a different trajectory than
    // the warmup games (which used `seed` directly).
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xDEADBEEF);
    let mut total_score: u64 = 0;
    let mut total_moves: u64 = 0;
    for _ in 0..games {
        let result = train_one_game(network, tables, learning_rate, &mut rng);
        total_score += result.score as u64;
        total_moves += result.moves as u64;
    }
    (total_score, total_moves)
}

fn run_hogwild_bench(
    network: &NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    threads: u32,
    games: u32,
    seed: u64,
) -> (u64, u64) {
    let stats = train_hogwild_batch(
        network,
        tables,
        learning_rate,
        threads,
        games,
        seed ^ 0xDEADBEEF,
    );
    (stats.total_score, stats.total_moves)
}

/// Pure throughput math — isolated for unit testing.
fn compute_throughput(games: u32, moves: u64, elapsed: Duration) -> (f64, f64) {
    let elapsed_secs = elapsed.as_secs_f64();
    if elapsed_secs <= 0.0 {
        return (0.0, 0.0);
    }
    let games_per_sec = games as f64 / elapsed_secs;
    let moves_per_sec = moves as f64 / elapsed_secs;
    (games_per_sec, moves_per_sec)
}

/// Reads peak resident set size (kB) from /proc/self/status.
/// Returns None on non-Linux systems or if the field isn't found.
fn read_peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb_str = rest.trim().split_whitespace().next()?;
            return kb_str.parse().ok();
        }
    }
    None
}

#[derive(Serialize)]
struct BenchmarkResult {
    label: Option<String>,
    algorithm: String,
    num_threads: u32,
    patterns: String,
    num_base_patterns: usize,
    num_total_patterns: usize,
    games: u32,
    warmup_games: u32,
    learning_rate: f32,
    seed: u64,
    wall_clock_seconds: f64,
    games_per_second: f64,
    moves_per_second: f64,
    total_moves: u64,
    avg_score_per_game: f64,
    avg_moves_per_game: f64,
    peak_rss_kb: Option<u64>,
}

fn main() {
    let args = Args::parse();
    validate_algorithm(&args.algorithm, args.threads);

    let patterns = select_patterns(&args.patterns);

    println!("Benchmark:");
    println!("  Label: {}", args.label.as_deref().unwrap_or("(none)"));
    println!("  Algorithm: {} ({} thread(s))", args.algorithm, args.threads);
    println!(
        "  Patterns: {} ({} base x 8 = {} total)",
        args.patterns,
        patterns.len(),
        patterns.len() * 8
    );
    println!("  Games: {} ({} warmup)", args.games, args.warmup_games);
    println!("  Learning rate: {}", args.learning_rate);
    println!("  Seed: {}", args.seed);
    println!();

    let tables = MoveTables::new();
    let network = NTupleNetwork::with_symmetry_expansion(&patterns, 0.0);
    let mut warmup_rng = SmallRng::seed_from_u64(args.seed);

    for _ in 0..args.warmup_games {
        train_one_game(&network, &tables, args.learning_rate, &mut warmup_rng);
    }

    let start = Instant::now();
    let (total_score, total_moves) = match args.algorithm.as_str() {
        "serial" => run_serial_bench(&network, &tables, args.learning_rate, args.games, args.seed),
        "hogwild" => run_hogwild_bench(
            &network,
            &tables,
            args.learning_rate,
            args.threads,
            args.games,
            args.seed,
        ),
        other => panic!("Unreachable: validate_algorithm should reject {other}"),
    };
    let elapsed = start.elapsed();
    let (games_per_sec, moves_per_sec) = compute_throughput(args.games, total_moves, elapsed);
    let avg_score = total_score as f64 / args.games as f64;
    let avg_moves = total_moves as f64 / args.games as f64;
    let peak_rss_kb = read_peak_rss_kb();

    println!("Results:");
    println!("  Wall-clock: {:.2}s", elapsed.as_secs_f64());
    println!("  Games/sec: {games_per_sec:.0}");
    println!("  Moves/sec: {moves_per_sec:.0}");
    println!("  Avg score/game: {avg_score:.0}");
    println!("  Avg moves/game: {avg_moves:.0}");
    if let Some(kb) = peak_rss_kb {
        println!("  Peak RSS: {} MB", kb / 1024);
    }

    let result = BenchmarkResult {
        label: args.label.clone(),
        algorithm: args.algorithm.clone(),
        num_threads: args.threads,
        patterns: args.patterns.clone(),
        num_base_patterns: patterns.len(),
        num_total_patterns: patterns.len() * 8,
        games: args.games,
        warmup_games: args.warmup_games,
        learning_rate: args.learning_rate,
        seed: args.seed,
        wall_clock_seconds: elapsed.as_secs_f64(),
        games_per_second: games_per_sec,
        moves_per_second: moves_per_sec,
        total_moves,
        avg_score_per_game: avg_score,
        avg_moves_per_game: avg_moves,
        peak_rss_kb,
    };

    if let Some(ref path) = args.output_json {
        let json = serde_json::to_string_pretty(&result).expect("serialize result");
        fs::write(path, json).expect("write JSON output");
        println!("  JSON written: {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_throughput_scales_with_elapsed() {
        let (gps, mps) = compute_throughput(1000, 500_000, Duration::from_secs(1));
        assert!((gps - 1000.0).abs() < 1e-9);
        assert!((mps - 500_000.0).abs() < 1e-9);
    }

    #[test]
    fn compute_throughput_handles_sub_second() {
        let (gps, mps) = compute_throughput(100, 50_000, Duration::from_millis(500));
        assert!((gps - 200.0).abs() < 1e-9);
        assert!((mps - 100_000.0).abs() < 1e-9);
    }

    #[test]
    fn compute_throughput_handles_zero_elapsed() {
        let (gps, mps) = compute_throughput(1000, 1000, Duration::ZERO);
        assert_eq!(gps, 0.0);
        assert_eq!(mps, 0.0);
    }

    #[test]
    fn benchmark_result_serializes_to_json() {
        let result = BenchmarkResult {
            label: Some("test".to_string()),
            algorithm: "serial".to_string(),
            num_threads: 1,
            patterns: "4x6".to_string(),
            num_base_patterns: 4,
            num_total_patterns: 32,
            games: 1000,
            warmup_games: 100,
            learning_rate: 0.0025,
            seed: 42,
            wall_clock_seconds: 1.5,
            games_per_second: 666.67,
            moves_per_second: 50000.0,
            total_moves: 75000,
            avg_score_per_game: 5000.0,
            avg_moves_per_game: 75.0,
            peak_rss_kb: Some(102400),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"algorithm\":\"serial\""));
        assert!(json.contains("\"num_threads\":1"));
        assert!(json.contains("\"games_per_second\":666.67"));
        assert!(json.contains("\"peak_rss_kb\":102400"));
        assert!(json.contains("\"label\":\"test\""));
    }

    #[test]
    fn benchmark_result_serializes_with_missing_optionals() {
        let result = BenchmarkResult {
            label: None,
            algorithm: "serial".to_string(),
            num_threads: 1,
            patterns: "4x6".to_string(),
            num_base_patterns: 4,
            num_total_patterns: 32,
            games: 10,
            warmup_games: 0,
            learning_rate: 0.0025,
            seed: 1,
            wall_clock_seconds: 0.1,
            games_per_second: 100.0,
            moves_per_second: 1000.0,
            total_moves: 100,
            avg_score_per_game: 1.0,
            avg_moves_per_game: 10.0,
            peak_rss_kb: None,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"label\":null"));
        assert!(json.contains("\"peak_rss_kb\":null"));
    }
}
