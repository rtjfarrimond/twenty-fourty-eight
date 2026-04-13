mod agent_loop;
mod model_registry;
mod protocol;
mod session;
mod websocket;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::get;
use game_engine::MoveTables;
use model::Agent;
use model::dummy::DummyAgent;
use model::ntuple_agent::NTupleAgent;
use model_registry::ModelRegistry;
use session::SessionManager;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use websocket::{AppState, websocket_handler};

#[tokio::main]
async fn main() {
    let tables = Arc::new(MoveTables::new());
    let session_manager = Arc::new(Mutex::new(SessionManager::new(tables.clone())));

    let mut registry = ModelRegistry::new();

    // Register trained n-tuple model if available
    let model_path = "../training/ntuple-4x6-td0-v1.bin";
    match NTupleAgent::load(
        model_path,
        "ntuple-4x6-td0-v1",
        "N-tuple network (4 base 6-tuple patterns) trained with TD(0) \
         and afterstate value functions. 100K training games.",
        tables.clone(),
    ) {
        Ok(agent) => {
            println!("Loaded trained model: {}", agent.name());
            registry.register(Arc::new(agent));
        }
        Err(err) => {
            println!("No trained model at {model_path}: {err}");
        }
    }

    // Register dummy heuristic agent
    let dummy = DummyAgent::new(tables.clone());
    println!("Registered model: {}", dummy.name());
    registry.register(Arc::new(dummy));

    let registry = Arc::new(registry);

    // Spawn an agent game loop for each registered model
    let move_interval = Duration::from_millis(500);
    for (name, model) in registry.iter() {
        println!("Starting game loop for: {name}");
        let agent = model.agent.clone();
        let sender = model.sender.clone();
        let tables = tables.clone();
        tokio::spawn(agent_loop::run_agent_loop(
            agent,
            tables,
            sender,
            move_interval,
        ));
    }

    let state = Arc::new(AppState {
        session_manager,
        model_registry: registry,
    });

    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .route("/training_log.jsonl", get(serve_training_log))
        .route("/training_config.json", get(serve_training_config))
        .fallback_service(ServeDir::new("../frontend/dist"))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("[::]:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn serve_training_log() -> impl IntoResponse {
    serve_file("../training/training_log.jsonl", "application/jsonl").await
}

async fn serve_training_config() -> impl IntoResponse {
    serve_file("../training/training_config.json", "application/json").await
}

async fn serve_file(path: &str, content_type: &str) -> axum::response::Response {
    match tokio::fs::read_to_string(path).await {
        Ok(contents) => (
            [(axum::http::header::CONTENT_TYPE, content_type.to_string())],
            contents,
        )
            .into_response(),
        Err(_) => (
            axum::http::StatusCode::NOT_FOUND,
            format!("File not found: {path}"),
        )
            .into_response(),
    }
}
