use game_engine::MoveTables;
use serde::Serialize;
use std::fs;
use training::eval::EvalResult;
use training::eval_dummy;

#[derive(Serialize)]
struct ModelEntry {
    model_name: String,
    model_type: String,
    description: String,
    training_curve: Option<Vec<EvalResult>>,
    final_eval: EvalResult,
}

#[derive(Serialize)]
struct ModelsManifest {
    models: Vec<ModelEntry>,
}

fn main() {
    let tables = MoveTables::new();
    let mut models = Vec::new();

    // Evaluate the dummy heuristic agent
    println!("Evaluating dummy agent (1000 games)...");
    let dummy_eval = eval_dummy::evaluate_dummy(&tables, 1000);
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

    // Check for trained model results
    let training_log_path = "training_log.jsonl";
    let training_config_path = "training_config.json";

    if let (Ok(log_text), Ok(config_text)) = (
        fs::read_to_string(training_log_path),
        fs::read_to_string(training_config_path),
    ) {
        let curve: Vec<EvalResult> = log_text
            .lines()
            .filter(|line| !line.is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        if let Some(final_eval) = curve.last() {
            let config: serde_json::Value = serde_json::from_str(&config_text).unwrap();
            let model_name = config["model_name"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();
            let model_type = config["model_type"]
                .as_str()
                .unwrap_or("unknown")
                .to_string();

            println!(
                "  {model_name}: avg {:.0}, max {}",
                final_eval.avg_score, final_eval.max_score
            );

            models.push(ModelEntry {
                model_name,
                model_type,
                description: format!(
                    "N-tuple TD(0), {} patterns, lr={}",
                    config["num_total_patterns"], config["learning_rate"]
                ),
                training_curve: Some(curve),
                final_eval: serde_json::from_str(&log_text.lines().last().unwrap_or("")).unwrap(),
            });
        }
    }

    let manifest = ModelsManifest { models };
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let output_path = "../frontend/dist/models.json";
    fs::write(output_path, &json).expect("Failed to write models.json");
    println!(
        "\nWritten {output_path} with {} model(s)",
        manifest.models.len()
    );
}
