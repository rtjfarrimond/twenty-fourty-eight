use game_engine::{Board, MoveTables, apply_move, empty_tiles, is_game_over, spawn_tile};
use model::Agent;
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time;

use crate::protocol::{ServerMessage, board_to_tile_values};

/// Runs the agent game loop, broadcasting state after each move.
/// When the game ends, starts a new one automatically.
pub async fn run_agent_loop(
    agent: Arc<dyn Agent + Send + Sync>,
    tables: Arc<MoveTables>,
    sender: broadcast::Sender<String>,
    move_interval: Duration,
) {
    loop {
        let mut board = spawn_initial_tiles(Board::new());
        let mut score: u32 = 0;

        broadcast_state(&sender, &board, score, false);

        while !is_game_over(&board, &tables) {
            time::sleep(move_interval).await;

            let direction = agent.best_move(&board);
            if let Some((new_board, move_score)) = apply_move(&board, direction, &tables) {
                board = spawn_random_tile(new_board);
                score += move_score;
            }

            let game_over = is_game_over(&board, &tables);
            broadcast_state(&sender, &board, score, game_over);
        }

        // Pause briefly before starting a new game
        time::sleep(Duration::from_secs(3)).await;
    }
}

fn spawn_initial_tiles(mut board: Board) -> Board {
    let mut rng = rand::rng();
    for _ in 0..2 {
        let empties = empty_tiles(&board);
        let position = empties[rng.random_range(0..empties.len())];
        board = spawn_tile(&board, position.0, position.1, rng.random::<f64>());
    }
    board
}

fn spawn_random_tile(board: Board) -> Board {
    let mut rng = rand::rng();
    let empties = empty_tiles(&board);
    if empties.is_empty() {
        return board;
    }
    let position = empties[rng.random_range(0..empties.len())];
    spawn_tile(&board, position.0, position.1, rng.random::<f64>())
}

fn broadcast_state(sender: &broadcast::Sender<String>, board: &Board, score: u32, game_over: bool) {
    let message = ServerMessage::GameState {
        board: board_to_tile_values(board),
        score,
        game_over,
    };
    if let Ok(json) = serde_json::to_string(&message) {
        // Ignore send errors — just means no watchers are connected
        let _ = sender.send(json);
    }
}
