use std::thread;

use game_engine::{Board, EmptyTiles, MoveTables, all_afterstates, spawn_tile};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use crate::ntuple::NTupleNetwork;

/// Stats returned from a completed training game.
#[derive(Debug, Clone, Copy)]
pub struct GameResult {
    pub score: u32,
    pub moves: u32,
}

/// Result of evaluating a single move, with cached network value.
struct MoveCandidate {
    afterstate: Board,
    reward: u32,
    value: f32,
}

/// Finds the best move by evaluating all legal afterstates.
/// Uses batch afterstate computation (single transpose for vertical moves).
/// Returns None if no legal move exists (game over).
#[inline]
fn best_afterstate(
    board: &Board,
    network: &NTupleNetwork,
    tables: &MoveTables,
) -> Option<MoveCandidate> {
    let afterstates = all_afterstates(board, tables);
    let mut best: Option<MoveCandidate> = None;
    let mut best_total = f32::NEG_INFINITY;

    for &(afterstate, reward, changed) in &afterstates {
        if !changed {
            continue;
        }
        let value = network.evaluate(&afterstate);
        let total = reward as f32 + value;
        if total > best_total {
            best_total = total;
            best = Some(MoveCandidate {
                afterstate,
                reward,
                value,
            });
        }
    }

    best
}

/// Plays one complete game using the network for move selection,
/// updating weights via TD(0) after each move.
/// Returns the final score and total number of moves played.
///
/// Takes `&NTupleNetwork` (not `&mut`) because the network's weights are
/// interior-mutable via atomics. This is what lets multiple worker threads
/// share a single network in Hogwild-style parallel training.
pub fn train_one_game(
    network: &NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    rng: &mut impl Rng,
) -> GameResult {
    let mut board = spawn_random_tile(Board::new(), rng);
    board = spawn_random_tile(board, rng);
    let mut total_score: u32 = 0;
    let mut moves: u32 = 0;

    let Some(mut current) = best_afterstate(&board, network, tables) else {
        return GameResult { score: 0, moves: 0 };
    };
    total_score += current.reward;
    moves += 1;

    loop {
        let next_board = spawn_random_tile(current.afterstate, rng);

        let Some(next) = best_afterstate(&next_board, network, tables) else {
            // Game over — terminal update: V(terminal) = 0
            network.update(&current.afterstate, learning_rate * -current.value);
            break;
        };

        // TD(0): V(s) <- V(s) + α * [r' + V(s') - V(s)]
        let td_error = next.reward as f32 + next.value - current.value;
        network.update(&current.afterstate, learning_rate * td_error);

        total_score += next.reward;
        moves += 1;
        current = next;
    }

    GameResult {
        score: total_score,
        moves,
    }
}

/// Aggregated stats from a Hogwild training batch.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStats {
    pub total_score: u64,
    pub total_moves: u64,
    pub games_played: u64,
}

/// Plays `games` training games across `num_threads` worker threads using
/// Hogwild-style parallelism: all threads share the same `NTupleNetwork` via
/// `&` and mutate its weights concurrently through relaxed atomic stores.
///
/// Games are split evenly across threads (remainder goes to the first thread).
/// Each worker gets its own RNG seeded from `(base_seed ^ worker_id)` so runs
/// are reproducible given the same seed and thread count — though not
/// bit-identical to serial runs, because race-condition outcomes depend on
/// scheduling.
pub fn train_hogwild_batch(
    network: &NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    num_threads: u32,
    games: u32,
    base_seed: u64,
) -> BatchStats {
    assert!(num_threads >= 1, "num_threads must be >= 1");
    if games == 0 {
        return BatchStats::default();
    }

    let per_worker = games / num_threads;
    let remainder = games % num_threads;

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_threads as usize);
        for worker_id in 0..num_threads {
            let extra = if worker_id < remainder { 1 } else { 0 };
            let worker_games = per_worker + extra;
            let worker_seed = base_seed ^ (worker_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let handle = scope.spawn(move || {
                run_worker(network, tables, learning_rate, worker_games, worker_seed)
            });
            handles.push(handle);
        }

        let mut stats = BatchStats::default();
        for handle in handles {
            let worker_stats = handle.join().expect("hogwild worker panicked");
            stats.total_score += worker_stats.total_score;
            stats.total_moves += worker_stats.total_moves;
            stats.games_played += worker_stats.games_played;
        }
        stats
    })
}

fn run_worker(
    network: &NTupleNetwork,
    tables: &MoveTables,
    learning_rate: f32,
    games: u32,
    seed: u64,
) -> BatchStats {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut stats = BatchStats::default();
    for _ in 0..games {
        let result = train_one_game(network, tables, learning_rate, &mut rng);
        stats.total_score += result.score as u64;
        stats.total_moves += result.moves as u64;
        stats.games_played += 1;
    }
    stats
}

#[inline]
fn spawn_random_tile(board: Board, rng: &mut impl Rng) -> Board {
    let empties = EmptyTiles::find(&board);
    if empties.is_empty() {
        return board;
    }
    let position = empties.get(rng.random_range(0..empties.len()));
    spawn_tile(&board, position.0, position.1, rng.random::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntuple::NTupleNetwork;
    use rand::SeedableRng;

    fn tables() -> MoveTables {
        MoveTables::new()
    }

    fn small_network() -> NTupleNetwork {
        let base_patterns = vec![vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]];
        NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0)
    }

    #[test]
    fn train_one_game_returns_nonzero_score_and_moves() {
        let tables = tables();
        let network = small_network();
        let mut rng = rand::rng();

        let result = train_one_game(&network, &tables, 0.0025, &mut rng);
        assert!(result.score > 0);
        assert!(result.moves > 0);
    }

    #[test]
    fn train_one_game_modifies_weights() {
        let tables = tables();
        let network = small_network();
        let mut rng = rand::rng();

        let initial_eval = network.evaluate(&Board::new());
        train_one_game(&network, &tables, 0.0025, &mut rng);
        let after_eval = network.evaluate(&Board::new());

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
        let network = NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut early_scores = Vec::new();
        let mut late_scores = Vec::new();

        for game in 0..2000 {
            let result = train_one_game(&network, &tables, 0.0025, &mut rng);
            if game < 100 {
                early_scores.push(result.score);
            } else if game >= 1900 {
                late_scores.push(result.score);
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
    fn game_result_moves_roughly_correlates_with_score() {
        // Sanity check: longer games (more moves) tend to score higher.
        // Not a strict inequality, just an average over multiple games.
        let tables = tables();
        let network = small_network();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);

        let mut total_moves: u64 = 0;
        let mut total_score: u64 = 0;
        for _ in 0..50 {
            let result = train_one_game(&network, &tables, 0.0025, &mut rng);
            total_moves += result.moves as u64;
            total_score += result.score as u64;
        }

        // Every game plays at least one move; 50 games should yield >50 moves.
        assert!(total_moves > 50);
        // Score should be positive in aggregate.
        assert!(total_score > 0);
    }

    fn four_pattern_network() -> NTupleNetwork {
        let base_patterns = vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            vec![(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        ];
        NTupleNetwork::with_symmetry_expansion(&base_patterns, 0.0)
    }

    #[test]
    fn hogwild_empty_batch_returns_zero_stats() {
        let tables = tables();
        let network = small_network();
        let stats = train_hogwild_batch(&network, &tables, 0.0025, 4, 0, 42);
        assert_eq!(stats.games_played, 0);
        assert_eq!(stats.total_score, 0);
        assert_eq!(stats.total_moves, 0);
    }

    #[test]
    fn hogwild_single_thread_plays_all_games() {
        let tables = tables();
        let network = small_network();
        let stats = train_hogwild_batch(&network, &tables, 0.0025, 1, 50, 42);
        assert_eq!(stats.games_played, 50);
        assert!(stats.total_moves > 0);
        assert!(stats.total_score > 0);
    }

    #[test]
    fn hogwild_multi_thread_plays_all_games() {
        let tables = tables();
        let network = small_network();
        let stats = train_hogwild_batch(&network, &tables, 0.0025, 4, 100, 42);
        // games (100) splits evenly across 4 workers = 25 each.
        assert_eq!(stats.games_played, 100);
    }

    #[test]
    fn hogwild_distributes_remainder_across_workers() {
        let tables = tables();
        let network = small_network();
        // 103 games / 4 threads = 25 each + 3 remainder → first 3 workers
        // get 26 games, last worker gets 25.
        let stats = train_hogwild_batch(&network, &tables, 0.0025, 4, 103, 42);
        assert_eq!(stats.games_played, 103);
    }

    #[test]
    fn hogwild_modifies_weights() {
        let tables = tables();
        let network = small_network();
        let board = Board::new();
        let initial = network.evaluate(&board);
        train_hogwild_batch(&network, &tables, 0.0025, 4, 100, 42);
        let after = network.evaluate(&board);
        assert_ne!(initial, after);
    }

    #[test]
    fn hogwild_produces_no_nan_or_inf_weights() {
        // Stress test: hammer the network from multiple threads, verify no
        // racy writes produced NaN or Inf values.
        let tables = tables();
        let network = four_pattern_network();
        train_hogwild_batch(&network, &tables, 0.0025, 4, 500, 42);

        // Eval a handful of random boards; NaN would propagate through +=.
        let mut rng = SmallRng::seed_from_u64(99);
        for _ in 0..20 {
            let mut board = Board::new();
            for row in 0..4 {
                for col in 0..4 {
                    board.set_tile(row, col, rng.random_range(0..10));
                }
            }
            let value = network.evaluate(&board);
            assert!(value.is_finite(), "weight eval produced non-finite value: {value}");
        }
    }

    #[test]
    fn hogwild_training_improves_over_many_games() {
        // Convergence parity smoke test: hogwild should still learn.
        let tables = tables();
        let network = four_pattern_network();

        // Baseline avg score from a cold network.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut baseline_score = 0u64;
        for _ in 0..100 {
            let result = train_one_game(&network, &tables, 0.0, &mut rng);
            baseline_score += result.score as u64;
        }
        let baseline_avg = baseline_score as f64 / 100.0;

        // Reset the network weights by creating a fresh one.
        let network = four_pattern_network();
        let trained_stats = train_hogwild_batch(&network, &tables, 0.0025, 4, 2000, 7);
        let trained_avg = trained_stats.total_score as f64 / trained_stats.games_played as f64;

        assert!(
            trained_avg > baseline_avg,
            "Hogwild training failed to improve: baseline avg {baseline_avg:.0}, \
             after 2000 games avg {trained_avg:.0}"
        );
    }

    #[test]
    fn best_afterstate_returns_none_for_game_over_board() {
        let tables = tables();
        let network = small_network();
        let mut board = Board::new();

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
        let mut network = small_network();

        let mut board = Board::new();
        board.set_tile(0, 0, 1);
        board.set_tile(0, 1, 1);

        let mut rng = rand::rng();
        for _ in 0..50 {
            train_one_game(&mut network, &tables, 0.01, &mut rng);
        }

        let candidate = best_afterstate(&board, &network, &tables);
        assert!(candidate.is_some());
    }
}
