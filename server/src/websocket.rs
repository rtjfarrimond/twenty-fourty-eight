use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};

use crate::model_registry::ModelRegistry;
use crate::protocol::{ClientMessage, ServerMessage, board_to_tile_values, direction_to_string};
use crate::session::SessionManager;

pub struct AppState {
    pub session_manager: Arc<Mutex<SessionManager>>,
    pub model_registry: Arc<ModelRegistry>,
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

    // Send model list on connect
    let model_list = ServerMessage::ModelList {
        models: state.model_registry.list(),
    };
    let json = serde_json::to_string(&model_list).unwrap();
    if socket.send(Message::Text(json.into())).await.is_err() {
        return;
    }

    // Auto-watch the default model
    if let Some(default_name) = state.model_registry.default_model() {
        if let Some(model) = state.model_registry.get(default_name) {
            agent_receiver = Some(model.sender.subscribe());
        }
    }

    loop {
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
                                &text, &mut session_id, &mut agent_receiver, &state,
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
        Ok(ClientMessage::ListModels) => Some(ServerMessage::ModelList {
            models: state.model_registry.list(),
        }),
        Ok(ClientMessage::WatchAgent { model }) => {
            // Keep the session alive — user can resume later
            match state.model_registry.get(&model) {
                Some(registered) => {
                    *agent_receiver = Some(registered.sender.subscribe());
                    None
                }
                None => Some(ServerMessage::Error {
                    message: format!("Unknown model: {model}"),
                }),
            }
        }
        Ok(ClientMessage::ResumeGame) => {
            *agent_receiver = None;

            match session_id.as_ref() {
                Some(current_id) => {
                    let manager = state.session_manager.lock().await;
                    let session = manager.get_session(current_id).unwrap();
                    let board = board_to_tile_values(&session.board);
                    let score = session.score;
                    let last_move = session.last_move.map(|d| direction_to_string(&d));
                    let game_over = manager.is_game_over(current_id);

                    Some(ServerMessage::GameState {
                        board,
                        score,
                        game_over,
                        last_move,
                    })
                }
                None => {
                    // No existing session — create a new one
                    let mut manager = state.session_manager.lock().await;
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
            }
        }
        Ok(ClientMessage::NewGame) => {
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
