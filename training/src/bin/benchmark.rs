use std::time::Instant;

use game_engine::MoveTables;
use training::ntuple::NTupleNetwork;
use training::training::train_one_game;

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
        .unwrap_or(10_000);

    println!("Benchmark: {num_games} training games");
    println!("Patterns: 4 base x 8 symmetries = 32");
    println!();

    let tables = MoveTables::new();
    let patterns = standard_6tuple_patterns();
    let mut network = NTupleNetwork::with_symmetry_expansion(&patterns, 0.0);
    let mut rng = rand::rng();

    // Warmup
    for _ in 0..100 {
        train_one_game(&mut network, &tables, 0.0025, &mut rng);
    }

    let start = Instant::now();
    let mut total_score: u64 = 0;

    for _ in 0..num_games {
        let score = train_one_game(&mut network, &tables, 0.0025, &mut rng);
        total_score += score as u64;
    }

    let elapsed = start.elapsed();
    let games_per_sec = num_games as f64 / elapsed.as_secs_f64();
    let avg_score = total_score as f64 / num_games as f64;

    // Rough estimate: avg game is ~200 moves at these score levels
    let estimated_moves_per_game = avg_score / 20.0; // ~20 points per move average
    let moves_per_sec = games_per_sec * estimated_moves_per_game;

    println!("Results:");
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Games/sec: {games_per_sec:.0}");
    println!("  Avg score: {avg_score:.0}");
    println!("  Est. moves/game: {estimated_moves_per_game:.0}");
    println!("  Est. moves/sec: {moves_per_sec:.0}");
}
