mod agent_loop;
mod protocol;
mod session;
mod websocket;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::get;
use game_engine::MoveTables;
use model::dummy::DummyAgent;
use session::SessionManager;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, broadcast};
use tower_http::services::ServeDir;
use websocket::{AppState, websocket_handler};

#[tokio::main]
async fn main() {
    let tables = Arc::new(MoveTables::new());
    let session_manager = Arc::new(Mutex::new(SessionManager::new(tables.clone())));

    let (agent_broadcast, _) = broadcast::channel(16);

    let agent: Arc<dyn model::Agent + Send + Sync> = Arc::new(DummyAgent::new(tables.clone()));
    let move_interval = Duration::from_millis(500);

    let broadcast_sender = agent_broadcast.clone();
    tokio::spawn(agent_loop::run_agent_loop(
        agent,
        tables,
        broadcast_sender,
        move_interval,
    ));

    let state = Arc::new(AppState {
        session_manager,
        agent_broadcast,
    });

    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .route("/training_log.jsonl", get(serve_training_log))
        .fallback_service(ServeDir::new("../frontend/dist"))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("[::]:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn serve_training_log() -> impl IntoResponse {
    let path = "../training/training_log.jsonl";
    match tokio::fs::read_to_string(path).await {
        Ok(contents) => (
            [(axum::http::header::CONTENT_TYPE, "application/jsonl")],
            contents,
        )
            .into_response(),
        Err(_) => (axum::http::StatusCode::NOT_FOUND, "No training log found").into_response(),
    }
}
