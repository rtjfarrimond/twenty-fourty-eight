use game_engine::MoveTables;
use serde::Serialize;
use std::fs;
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

struct TrainedModel {
    bin_path: String,
    name: String,
    description: String,
    log_path: Option<String>,
}

fn main() {
    let eval_games: u32 = 1000;
    let tables = MoveTables::new();
    let mut models = Vec::new();

    // Define trained models to include
    let trained_models = vec![
        TrainedModel {
            bin_path: "ntuple-4x6-td0-1M.bin".to_string(),
            name: "ntuple-4x6-td0-1M".to_string(),
            description: "N-tuple TD(0), 32 patterns, lr=0.0025, 1M training games.".to_string(),
            log_path: None, // Log was overwritten; eval from model directly
        },
        TrainedModel {
            bin_path: "ntuple-4x6-td0-v1.bin".to_string(),
            name: "ntuple-4x6-td0-100K".to_string(),
            description: "N-tuple TD(0), 32 patterns, lr=0.0025, 100K training games.".to_string(),
            log_path: None,
        },
    ];

    for trained in &trained_models {
        if !std::path::Path::new(&trained.bin_path).exists() {
            println!("  Skipping {} (file not found)", trained.bin_path);
            continue;
        }

        println!("Evaluating {} ({eval_games} games)...", trained.name);
        let network = NTupleNetwork::load(&trained.bin_path)
            .unwrap_or_else(|err| panic!("Failed to load {}: {err}", trained.bin_path));

        let final_eval = eval::evaluate(&network, &tables, eval_games, 0);
        println!(
            "  {}: avg {:.0}, max {}",
            trained.name, final_eval.avg_score, final_eval.max_score
        );

        // Load training curve from log if available
        let training_curve = trained.log_path.as_ref().and_then(|path| {
            fs::read_to_string(path).ok().map(|text| {
                text.lines()
                    .filter(|line| !line.is_empty())
                    .filter_map(|line| serde_json::from_str(line).ok())
                    .collect()
            })
        });

        models.push(ModelEntry {
            model_name: trained.name.clone(),
            model_type: "n-tuple".to_string(),
            description: trained.description.clone(),
            training_curve,
            final_eval,
        });
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
    let output_path = "../frontend/dist/models.json";
    fs::write(output_path, &json).expect("Failed to write models.json");
    println!(
        "\nWritten {output_path} with {} model(s)",
        manifest.models.len()
    );
}
