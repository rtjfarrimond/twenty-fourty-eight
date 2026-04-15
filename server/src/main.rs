mod agent_loop;
mod config;
mod model_registry;
mod model_watcher;
mod protocol;
mod queue_stream;
mod session;
mod training_stream;
mod websocket;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::get;
use config::ServerConfig;
use game_engine::MoveTables;
use model::Agent;
use model::dummy::DummyAgent;
use model::ntuple_agent::NTupleAgent;
use model_registry::ModelRegistry;
use session::SessionManager;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use websocket::{AppState, websocket_handler};

#[tokio::main]
async fn main() {
    let config = ServerConfig::load();

    println!("Server config:");
    println!("  Port: {}", config.port);
    println!("  Frontend: {}", config.frontend_dir.display());
    println!("  Models: {}", config.models_dir.display());
    println!("  Training: {}", config.training_dir.display());
    println!("  Queue: {}", config.queue_dir.display());
    println!("  Move interval: {}ms", config.move_interval_ms);
    println!();

    let tables = Arc::new(MoveTables::new());
    let session_manager = Arc::new(Mutex::new(SessionManager::new(tables.clone())));

    let registry = Arc::new(ModelRegistry::new());

    // Scan models directory for .bin files and load each with its .meta.toml
    if config.models_dir.exists() {
        load_models_from_directory(&config.models_dir, &tables, &registry).await;
    } else {
        println!(
            "Models directory not found: {}",
            config.models_dir.display()
        );
    }

    // Always register the dummy heuristic as a baseline
    let dummy = DummyAgent::new(tables.clone());
    println!("Registered model: {}", dummy.name());
    registry.register(Arc::new(dummy)).await;

    let move_interval = Duration::from_millis(config.move_interval_ms);
    for (name, agent, sender) in registry.snapshot().await {
        println!("Starting game loop for: {name}");
        let tables = tables.clone();
        tokio::spawn(agent_loop::run_agent_loop(
            agent,
            tables,
            sender,
            move_interval,
        ));
    }

    // Watch for new models dropped into the models directory
    if config.models_dir.exists() {
        tokio::spawn(model_watcher::watch_models_directory(
            config.models_dir.clone(),
            registry.clone(),
            tables.clone(),
            move_interval,
        ));
    }

    let state = Arc::new(AppState {
        session_manager,
        model_registry: registry,
    });

    let training_dir = config.training_dir.clone();
    let training_dir_sse = config.training_dir.clone();
    let queue_dir_sse = config.queue_dir.clone();
    let frontend_dir = config.frontend_dir.clone();
    let models_json_path = config.models_dir.join("models.json");

    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .route(
            "/training/stream",
            get(move || training_stream::training_stream(training_dir_sse.clone())),
        )
        .route(
            "/queue/stream",
            get(move || queue_stream::queue_stream(queue_dir_sse.clone())),
        )
        .route(
            "/models.json",
            get(move || serve_file(models_json_path.clone(), "application/json")),
        )
        .route(
            "/training_log.jsonl",
            get(move || serve_latest(training_dir.clone(), "log.jsonl", "application/jsonl")),
        )
        .route(
            "/training_config.json",
            get(move || {
                serve_latest(
                    config.training_dir.clone(),
                    "config.json",
                    "application/json",
                )
            }),
        )
        .fallback_service(ServeDir::new(&frontend_dir))
        .with_state(state);

    let bind_addr = format!("[::]:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    println!("\nServer running on http://localhost:{}", config.port);
    axum::serve(listener, app).await.unwrap();
}

/// Scans a directory for .bin model files and loads each one.
/// Looks for a matching .meta.toml sidecar for name/description.
async fn load_models_from_directory(
    directory: &Path,
    tables: &Arc<MoveTables>,
    registry: &Arc<ModelRegistry>,
) {
    let mut entries: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "bin"))
        .collect();

    // Sort by filename for consistent ordering
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let bin_path = entry.path();
        let stem = bin_path.file_stem().unwrap().to_string_lossy().to_string();

        // Look for a sidecar .meta.toml
        let meta_path = directory.join(format!("{stem}.meta.toml"));
        let (name, description) = if meta_path.exists() {
            let meta_str = std::fs::read_to_string(&meta_path).unwrap();
            let meta: toml::Value = toml::from_str(&meta_str).unwrap();
            (
                meta.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&stem)
                    .to_string(),
                meta.get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            )
        } else {
            (
                stem.clone(),
                format!("N-tuple model loaded from {stem}.bin"),
            )
        };

        match NTupleAgent::load(
            bin_path.to_str().unwrap(),
            &name,
            &description,
            tables.clone(),
        ) {
            Ok(agent) => {
                println!("Loaded model: {}", agent.name());
                registry.register(Arc::new(agent)).await;
            }
            Err(err) => {
                println!("Failed to load {}: {err}", bin_path.display());
            }
        }
    }
}

/// Finds the most recently modified file matching `*.{suffix}` in a directory
/// and serves it.
async fn serve_latest(
    directory: std::path::PathBuf,
    suffix: &str,
    content_type: &str,
) -> axum::response::Response {
    let latest = find_latest(&directory, suffix).await;
    match latest {
        Some(path) => match tokio::fs::read_to_string(&path).await {
            Ok(contents) => (
                [(axum::http::header::CONTENT_TYPE, content_type.to_string())],
                contents,
            )
                .into_response(),
            Err(_) => not_found(),
        },
        None => not_found(),
    }
}

fn not_found() -> axum::response::Response {
    (axum::http::StatusCode::NOT_FOUND, "Not found").into_response()
}

/// Read a file and serve it with the given content type. Used for runtime
/// data (models.json) that lives in /var/lib/ rather than alongside the
/// static frontend assets.
async fn serve_file(path: std::path::PathBuf, content_type: &str) -> axum::response::Response {
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, content_type.to_string())],
            bytes,
        )
            .into_response(),
        Err(_) => not_found(),
    }
}

async fn find_latest(directory: &Path, suffix: &str) -> Option<String> {
    let mut entries = tokio::fs::read_dir(directory).await.ok()?;
    let mut latest: Option<(String, std::time::SystemTime)> = None;

    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(suffix) {
            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    if latest.as_ref().map_or(true, |(_, prev)| modified > *prev) {
                        latest = Some((entry.path().to_string_lossy().to_string(), modified));
                    }
                }
            }
        }
    }

    latest.map(|(path, _)| path)
}
