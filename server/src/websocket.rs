use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};

use crate::protocol::{ClientMessage, ServerMessage, board_to_tile_values, direction_to_string};
use crate::session::SessionManager;

pub struct AppState {
    pub session_manager: Arc<Mutex<SessionManager>>,
    pub agent_broadcast: broadcast::Sender<String>,
}

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_connection(socket, state))
}

async fn handle_connection(mut socket: WebSocket, state: Arc<AppState>) {
    let mut session_id: Option<String> = None;
    let mut agent_receiver: Option<broadcast::Receiver<String>> = None;

    loop {
        // If watching the agent, forward broadcasts to the client
        if let Some(ref mut receiver) = agent_receiver {
            tokio::select! {
                result = receiver.recv() => {
                    match result {
                        Ok(json) => {
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                incoming = socket.recv() => {
                    match incoming {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(response) = handle_client_message(
                                &text,
                                &mut session_id,
                                &mut agent_receiver,
                                &state,
                            ).await {
                                let json = serde_json::to_string(&response).unwrap();
                                if socket.send(Message::Text(json.into())).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Some(Ok(_)) => continue,
                        _ => break,
                    }
                }
            }
        } else {
            // Not watching agent — just handle client messages
            match socket.recv().await {
                Some(Ok(Message::Text(text))) => {
                    if let Some(response) =
                        handle_client_message(&text, &mut session_id, &mut agent_receiver, &state)
                            .await
                    {
                        let json = serde_json::to_string(&response).unwrap();
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
                Some(Ok(_)) => continue,
                _ => break,
            }
        }
    }

    // Clean up session on disconnect
    if let Some(session_id) = session_id {
        state
            .session_manager
            .lock()
            .await
            .remove_session(&session_id);
    }
}

async fn handle_client_message(
    text: &str,
    session_id: &mut Option<String>,
    agent_receiver: &mut Option<broadcast::Receiver<String>>,
    state: &AppState,
) -> Option<ServerMessage> {
    match serde_json::from_str::<ClientMessage>(text) {
        Ok(ClientMessage::WatchAgent) => {
            // Stop any active user game
            if let Some(old_id) = session_id.take() {
                state.session_manager.lock().await.remove_session(&old_id);
            }
            // Subscribe to agent broadcast
            *agent_receiver = Some(state.agent_broadcast.subscribe());
            None // State will arrive via the broadcast channel
        }
        Ok(ClientMessage::NewGame) => {
            // Stop watching agent if we were
            *agent_receiver = None;

            let mut manager = state.session_manager.lock().await;
            if let Some(old_id) = session_id.take() {
                manager.remove_session(&old_id);
            }

            let (new_id, session) = manager.create_session();
            let board = board_to_tile_values(&session.board);
            let score = session.score;
            *session_id = Some(new_id);

            Some(ServerMessage::GameState {
                board,
                score,
                game_over: false,
                last_move: None,
            })
        }
        Ok(ClientMessage::Move { direction }) => {
            let Some(current_id) = session_id.as_ref() else {
                return Some(ServerMessage::Error {
                    message: "No active game. Send NewGame first.".to_string(),
                });
            };

            let engine_direction = game_engine::Direction::from(direction);
            let mut manager = state.session_manager.lock().await;
            manager.apply_move(current_id, engine_direction);

            let session = manager.get_session(current_id).unwrap();
            let board = board_to_tile_values(&session.board);
            let score = session.score;
            let game_over = manager.is_game_over(current_id);

            Some(ServerMessage::GameState {
                board,
                score,
                game_over,
                last_move: Some(direction_to_string(&engine_direction)),
            })
        }
        Err(err) => Some(ServerMessage::Error {
            message: format!("Invalid message: {err}"),
        }),
    }
}
