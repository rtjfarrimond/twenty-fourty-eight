use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::protocol::{ClientMessage, ServerMessage, board_to_tile_values};
use crate::session::SessionManager;

pub type SharedSessionManager = Arc<Mutex<SessionManager>>;

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(manager): State<SharedSessionManager>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_connection(socket, manager))
}

async fn handle_connection(mut socket: WebSocket, manager: SharedSessionManager) {
    let mut session_id: Option<String> = None;

    while let Some(Ok(message)) = socket.recv().await {
        let Message::Text(text) = message else {
            continue;
        };

        let response = match serde_json::from_str::<ClientMessage>(&text) {
            Ok(client_message) => process_message(client_message, &mut session_id, &manager).await,
            Err(err) => ServerMessage::Error {
                message: format!("Invalid message: {err}"),
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }

    // Clean up session on disconnect
    if let Some(session_id) = session_id {
        manager.lock().await.remove_session(&session_id);
    }
}

async fn process_message(
    message: ClientMessage,
    session_id: &mut Option<String>,
    manager: &SharedSessionManager,
) -> ServerMessage {
    let mut manager = manager.lock().await;

    match message {
        ClientMessage::NewGame => {
            // Clean up any existing session
            if let Some(old_id) = session_id.take() {
                manager.remove_session(&old_id);
            }

            let (new_id, session) = manager.create_session();
            let board = board_to_tile_values(&session.board);
            let game_over = false; // a fresh game is never over
            let score = session.score;
            *session_id = Some(new_id);

            ServerMessage::GameState {
                board,
                score,
                game_over,
            }
        }
        ClientMessage::Move { direction } => {
            let Some(current_id) = session_id.as_ref() else {
                return ServerMessage::Error {
                    message: "No active game. Send NewGame first.".to_string(),
                };
            };

            // Apply the move (may be a no-op if illegal)
            manager.apply_move(current_id, direction.into());

            // Read state after the move attempt
            let session = manager.get_session(current_id).unwrap();
            let board = board_to_tile_values(&session.board);
            let score = session.score;
            let game_over = manager.is_game_over(current_id);

            ServerMessage::GameState {
                board,
                score,
                game_over,
            }
        }
    }
}
