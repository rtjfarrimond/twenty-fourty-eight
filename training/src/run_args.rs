//! The full set of arguments needed to execute a training run.
//!
//! Shared between the `run` subcommand (parses these directly from CLI) and
//! the `submit` subcommand / queue daemon (serialize/deserialize via JSON).
//! Keeping a single source of truth means the queue payload format and the
//! foreground CLI never drift.

use std::path::PathBuf;

use clap::Args;
use serde::{Deserialize, Serialize};

/// Arguments for a single training execution. Matches the historical flat
/// CLI for backward compatibility — the `run` and `submit` subcommands both
/// expose this set of flags verbatim.
#[derive(Args, Serialize, Deserialize, Clone, Debug)]
pub struct RunArgs {
    /// Number of training games to play
    #[arg(long, default_value_t = 100_000)]
    pub games: u32,

    /// Evaluate every N games
    #[arg(long, default_value_t = 10_000)]
    pub eval_interval: u32,

    /// Number of games per evaluation checkpoint
    #[arg(long, default_value_t = 1000)]
    pub eval_games: u32,

    /// Name for the model artefact (used in filenames)
    #[arg(long, default_value = "ntuple-4x6-td0")]
    pub model_name: String,

    /// Optimistic weight initialization value (0.0 = no optimistic init).
    /// Sets every weight to this constant; with 8 symmetries × N base
    /// patterns, V(any board) starts at K × 8N. Make K large enough that
    /// V is comparable to the natural game-score scale or it isn't really
    /// "optimistic" relative to truth.
    #[arg(long, default_value_t = 0.0)]
    pub optimistic_init: f32,

    /// Random weight initialization amplitude — each weight is sampled
    /// uniformly from `[-amplitude, +amplitude]`. 0.0 = zero init.
    /// Mutually exclusive with --optimistic-init.
    #[arg(long, default_value_t = 0.0)]
    pub random_init_amplitude: f32,

    /// Seed for random init. Only used when --random-init-amplitude > 0.
    #[arg(long, default_value_t = 42)]
    pub random_init_seed: u64,

    /// Pattern preset: "4x6" (4 base 6-tuples) or "8x6" (8 base 6-tuples)
    #[arg(long, default_value = "4x6")]
    pub patterns: String,

    /// Learning rate (alpha) for TD algorithms, or meta-learning rate (beta)
    /// for TC algorithms. For TD, this is the fixed per-update step size
    /// (typical: 0.0025). For TC, this scales the per-weight adaptive rates
    /// that TC computes automatically (typical: 1.0 for serial, lower for
    /// tc-hogwild where concurrent updates add noise to the coherence signal).
    #[arg(long, default_value_t = 0.0025)]
    pub learning_rate: f32,

    /// Directory to deploy the trained model into. The .bin, .log.jsonl,
    /// and .config.json are moved here and models.json is regenerated.
    #[arg(long, default_value = "/var/lib/2048-solver/models")]
    pub models_dir: PathBuf,

    /// Human-readable description for the .meta.toml sidecar
    #[arg(long)]
    pub description: Option<String>,

    /// Training algorithm: "serial", "hogwild", "tc" (TC learning, serial),
    /// or "tc-hogwild" (TC learning, parallel). For TC algorithms,
    /// --learning-rate is used as the beta meta-learning rate (default 1.0
    /// recommended for serial TC, lower for tc-hogwild).
    #[arg(long, default_value = "serial")]
    pub algorithm: String,

    /// Number of worker threads. Must be 1 for the serial algorithm.
    #[arg(long, default_value_t = 1)]
    pub threads: u32,

    /// Ephemeral run: train and evaluate but do not save the model .bin
    /// or deploy. Only the .log.jsonl and .config.json are written.
    /// Useful for parameter sweeps where the training curve is the goal,
    /// not the model artefact.
    #[arg(long, default_value_t = false)]
    pub ephemeral: bool,
}

impl RunArgs {
    /// Reconstruct the argv tail (without the `run` subcommand or program
    /// name) for spawning a subprocess. The daemon uses this to invoke
    /// `training run <argv...>` with deserialized job args.
    pub fn to_argv(&self) -> Vec<String> {
        let mut argv: Vec<String> = vec![
            "--games".into(), self.games.to_string(),
            "--eval-interval".into(), self.eval_interval.to_string(),
            "--eval-games".into(), self.eval_games.to_string(),
            "--model-name".into(), self.model_name.clone(),
            "--optimistic-init".into(), self.optimistic_init.to_string(),
            "--random-init-amplitude".into(), self.random_init_amplitude.to_string(),
            "--random-init-seed".into(), self.random_init_seed.to_string(),
            "--patterns".into(), self.patterns.clone(),
            "--learning-rate".into(), self.learning_rate.to_string(),
            "--algorithm".into(), self.algorithm.clone(),
            "--threads".into(), self.threads.to_string(),
        ];
        argv.push("--models-dir".into());
        argv.push(self.models_dir.display().to_string());
        if let Some(desc) = &self.description {
            argv.push("--description".into());
            argv.push(desc.clone());
        }
        if self.ephemeral {
            argv.push("--ephemeral".into());
        }
        argv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_args() -> RunArgs {
        RunArgs {
            games: 1000,
            eval_interval: 100,
            eval_games: 50,
            model_name: "test".into(),
            optimistic_init: 0.0,
            random_init_amplitude: 0.0,
            random_init_seed: 42,
            patterns: "4x6".into(),
            learning_rate: 0.0025,
            models_dir: PathBuf::from("/var/lib/2048-solver/models"),
            description: None,
            algorithm: "serial".into(),
            threads: 1,
            ephemeral: false,
        }
    }

    #[test]
    fn json_round_trip_preserves_all_fields() {
        let args = sample_args();
        let json = serde_json::to_string(&args).unwrap();
        let parsed: RunArgs = serde_json::from_str(&json).unwrap();
        assert_eq!(args.games, parsed.games);
        assert_eq!(args.model_name, parsed.model_name);
        assert_eq!(args.algorithm, parsed.algorithm);
        assert_eq!(args.learning_rate, parsed.learning_rate);
        assert_eq!(args.models_dir, parsed.models_dir);
        assert_eq!(args.ephemeral, parsed.ephemeral);
    }

    #[test]
    fn ephemeral_defaults_to_false() {
        let args = sample_args();
        assert!(!args.ephemeral);
    }

    #[test]
    fn to_argv_includes_ephemeral_when_set() {
        let mut args = sample_args();
        args.ephemeral = true;
        let argv = args.to_argv();
        assert!(argv.iter().any(|s| s == "--ephemeral"));
    }

    #[test]
    fn to_argv_omits_ephemeral_when_false() {
        let args = sample_args();
        let argv = args.to_argv();
        assert!(!argv.iter().any(|s| s == "--ephemeral"));
    }

    #[test]
    fn to_argv_includes_required_flags() {
        let args = sample_args();
        let argv = args.to_argv();
        assert!(argv.windows(2).any(|w| w[0] == "--games" && w[1] == "1000"));
        assert!(argv.windows(2).any(|w| w[0] == "--model-name" && w[1] == "test"));
        assert!(argv.windows(2).any(|w| w[0] == "--models-dir"));
    }

    #[test]
    fn to_argv_includes_description_when_present() {
        let mut args = sample_args();
        args.description = Some("Test run".into());
        let argv = args.to_argv();
        assert!(argv.windows(2).any(|w| w[0] == "--description" && w[1] == "Test run"));
    }

    #[test]
    fn to_argv_omits_description_when_absent() {
        let args = sample_args();
        let argv = args.to_argv();
        assert!(!argv.iter().any(|s| s == "--description"));
    }
}
