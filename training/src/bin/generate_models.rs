use game_engine::MoveTables;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use training::eval;
use training::eval_dummy;
use training::ntuple::NTupleNetwork;

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

    // Scan for .bin files in the models directory
    if models_dir.exists() {
        let mut bin_files: Vec<_> = fs::read_dir(&models_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "bin"))
            .collect();

        bin_files.sort_by_key(|entry| entry.file_name());

        for entry in bin_files {
            let bin_path = entry.path();
            let stem = bin_path.file_stem().unwrap().to_string_lossy().to_string();

            // Load metadata from sidecar .meta.toml
            let meta_path = models_dir.join(format!("{stem}.meta.toml"));
            let (name, description) = load_metadata(&meta_path, &stem);

            // Look for training log
            let log_path = training_dir.join(format!("{stem}.log.jsonl"));
            let training_curve = load_training_curve(&log_path);

            // Prefer the last training-log checkpoint over a fresh eval —
            // fresh eval uses a non-deterministic RNG, so re-running this
            // binary produced different numbers than the live training
            // dashboard showed at training-end time. Using the logged
            // checkpoint keeps the results table and live view consistent.
            let final_eval = match training_curve.as_ref().and_then(|c| c.last().cloned()) {
                Some(last) => {
                    println!(
                        "Using logged final eval for {name} ({} games trained).",
                        last.games_trained
                    );
                    last
                }
                None => {
                    println!(
                        "No training log for {name}; running fresh eval ({eval_games} games)..."
                    );
                    let network = NTupleNetwork::load(bin_path.to_str().unwrap())
                        .unwrap_or_else(|err| {
                            panic!("Failed to load {}: {err}", bin_path.display())
                        });
                    eval::evaluate(&network, &tables, eval_games, 0)
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
