mod protocol;
mod session;
mod websocket;

use axum::Router;
use axum::routing::get;
use game_engine::MoveTables;
use session::SessionManager;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use websocket::{SharedSessionManager, websocket_handler};

#[tokio::main]
async fn main() {
    let tables = Arc::new(MoveTables::new());
    let manager: SharedSessionManager = Arc::new(Mutex::new(SessionManager::new(tables)));

    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .fallback_service(ServeDir::new("../frontend/dist"))
        .with_state(manager);

    let listener = tokio::net::TcpListener::bind("[::]:3000").await.unwrap();
    println!("Server running on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
