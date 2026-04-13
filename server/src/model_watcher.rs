use game_engine::MoveTables;
use model::ntuple_agent::NTupleAgent;
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::agent_loop;
use crate::model_registry::ModelRegistry;

/// Watches a directory for new `.bin` model files and hot-loads them into the
/// registry, spawning an agent game loop for each.
pub async fn watch_models_directory(
    models_dir: PathBuf,
    registry: Arc<ModelRegistry>,
    tables: Arc<MoveTables>,
    move_interval: Duration,
) {
    let (sender, mut receiver) = mpsc::channel::<PathBuf>(32);

    let watch_dir = models_dir.clone();
    let _watcher = std::thread::spawn(move || {
        run_filesystem_watcher(&watch_dir, sender);
    });

    while let Some(bin_path) = receiver.recv().await {
        // Brief delay to let .meta.toml arrive alongside the .bin
        tokio::time::sleep(Duration::from_secs(1)).await;

        let stem = match bin_path.file_stem() {
            Some(s) => s.to_string_lossy().to_string(),
            None => continue,
        };

        let (name, description) = load_metadata(&models_dir, &stem);

        match NTupleAgent::load(
            bin_path.to_str().unwrap(),
            &name,
            &description,
            tables.clone(),
        ) {
            Ok(agent) => {
                let agent = Arc::new(agent);
                if let Some(sender) = registry.register(agent.clone()).await {
                    println!("Hot-loaded model: {name}");
                    tokio::spawn(agent_loop::run_agent_loop(
                        agent,
                        tables.clone(),
                        sender,
                        move_interval,
                    ));
                } else {
                    println!("Model already registered, skipping: {name}");
                }
            }
            Err(err) => {
                eprintln!("Failed to hot-load {}: {err}", bin_path.display());
            }
        }
    }
}

/// Reads the `.meta.toml` sidecar for a model, falling back to defaults.
fn load_metadata(models_dir: &Path, stem: &str) -> (String, String) {
    let meta_path = models_dir.join(format!("{stem}.meta.toml"));
    if meta_path.exists() {
        if let Ok(meta_str) = std::fs::read_to_string(&meta_path) {
            if let Ok(meta) = toml::from_str::<toml::Value>(&meta_str) {
                let name = meta
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(stem)
                    .to_string();
                let description = meta
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                return (name, description);
            }
        }
    }
    (stem.to_string(), format!("N-tuple model loaded from {stem}.bin"))
}

/// Runs the blocking filesystem watcher on a dedicated thread, forwarding
/// new `.bin` file paths to the async receiver.
fn run_filesystem_watcher(
    watch_dir: &Path,
    sender: mpsc::Sender<PathBuf>,
) {
    let sender_clone = sender.clone();
    let mut watcher = notify::recommended_watcher(
        move |result: Result<notify::Event, notify::Error>| {
            let Ok(event) = result else { return };

            let is_create_or_modify = matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_)
            );
            if !is_create_or_modify {
                return;
            }

            for path in event.paths {
                if path.extension().map_or(false, |ext| ext == "bin") {
                    let _ = sender_clone.blocking_send(path);
                }
            }
        },
    )
    .expect("Failed to create filesystem watcher");

    watcher
        .watch(watch_dir, RecursiveMode::NonRecursive)
        .expect("Failed to watch models directory");

    println!("Watching for new models in: {}", watch_dir.display());

    // Block this thread forever — the watcher must stay alive
    std::thread::park();
}
