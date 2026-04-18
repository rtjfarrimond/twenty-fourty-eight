use game_engine::MoveTables;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use training::eval;
use training::eval_dummy;

#[derive(Serialize)]
struct ModelEntry {
    model_name: String,
    model_type: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    training_curve: Option<Vec<eval::EvalResult>>,
    final_eval: eval::EvalResult,
}

#[derive(Serialize)]
struct ModelsManifest {
    models: Vec<ModelEntry>,
}

fn main() {
    let eval_games: u32 = 1000;

    // Models dir: first arg, or current directory
    let models_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // Output path: second arg, or models_dir/models.json. Co-locating with
    // the .bin files keeps everything under /var/lib/2048-solver/ which the
    // 2048-solver service user owns; the server has a /models.json route
    // that reads from this location.
    let output_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| models_dir.join("models.json"));

    // Training logs dir: third arg, or /var/lib/2048-solver/training,
    // or fall back to models_dir (for dev)
    let training_dir = std::env::args()
        .nth(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let default_fhs = PathBuf::from("/var/lib/2048-solver/training");
            if default_fhs.exists() {
                default_fhs
            } else {
                models_dir.clone()
            }
        });

    println!("Models dir: {}", models_dir.display());
    println!("Output: {}", output_path.display());
    println!("Training logs: {}", training_dir.display());
    println!();

    let tables = MoveTables::new();
    let mut models = Vec::new();

    // Discover models: scan for .meta.toml files. Models with a .bin
    // can fall back to fresh eval; ephemeral models (no .bin) require
    // a training log for their eval data.
    if models_dir.exists() {
        let mut meta_files: Vec<_> = fs::read_dir(&models_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().map_or(false, |ext| ext == "toml")
                    && entry.path().to_string_lossy().ends_with(".meta.toml")
            })
            .collect();

        meta_files.sort_by_key(|entry| entry.file_name());

        for entry in meta_files {
            let meta_path = entry.path();
            let filename = meta_path.file_name().unwrap().to_string_lossy();
            let stem = filename.strip_suffix(".meta.toml").unwrap().to_string();

            let (name, description) = load_metadata(&meta_path, &stem);

            // Look for training log in both models dir and training dir
            let log_path_models = models_dir.join(format!("{stem}.log.jsonl"));
            let log_path_training = training_dir.join(format!("{stem}.log.jsonl"));
            let log_path = if log_path_models.exists() {
                log_path_models
            } else {
                log_path_training
            };
            let training_curve = load_training_curve(&log_path);

            let final_eval = match training_curve.as_ref().and_then(|c| c.last().cloned()) {
                Some(last) => {
                    println!(
                        "Using logged final eval for {name} ({} games trained).",
                        last.games_trained
                    );
                    last
                }
                None => {
                    eprintln!(
                        "WARNING: skipping {name} — no training log found. \
                         Re-run training with eval logging to include this model."
                    );
                    continue;
                }
            };
            println!(
                "  {name}: avg {:.0}, max {}",
                final_eval.avg_score, final_eval.max_score
            );

            models.push(ModelEntry {
                model_name: name,
                model_type: "n-tuple".to_string(),
                description,
                training_curve,
                final_eval,
            });
        }
    }

    // Evaluate the dummy heuristic agent
    println!("Evaluating dummy agent ({eval_games} games)...");
    let dummy_eval = eval_dummy::evaluate_dummy(&tables, eval_games);
    println!(
        "  Dummy: avg {:.0}, max {}",
        dummy_eval.avg_score, dummy_eval.max_score
    );

    models.push(ModelEntry {
        model_name: "heuristic-down-right".to_string(),
        model_type: "heuristic".to_string(),
        description: "Baseline heuristic: prefers Down, Right, Left, Up in that order.".to_string(),
        training_curve: None,
        final_eval: dummy_eval,
    });

    let manifest = ModelsManifest { models };
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(&output_path, &json).expect("Failed to write models.json");
    println!(
        "\nWritten {} with {} model(s)",
        output_path.display(),
        manifest.models.len()
    );
}

fn load_metadata(meta_path: &Path, default_name: &str) -> (String, String) {
    if meta_path.exists() {
        let meta_str = fs::read_to_string(meta_path).unwrap();
        let meta: toml::Value = toml::from_str(&meta_str).unwrap();
        (
            meta.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(default_name)
                .to_string(),
            meta.get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        )
    } else {
        (
            default_name.to_string(),
            format!("N-tuple model loaded from {default_name}.bin"),
        )
    }
}

fn load_training_curve(log_path: &Path) -> Option<Vec<eval::EvalResult>> {
    if !log_path.exists() {
        return None;
    }
    let text = fs::read_to_string(log_path).ok()?;
    let curve: Vec<eval::EvalResult> = text
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    if curve.is_empty() { None } else { Some(curve) }
}
