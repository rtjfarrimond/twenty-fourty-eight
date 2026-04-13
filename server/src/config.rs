use serde::Deserialize;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "/etc/2048-solver/config.toml";

#[derive(Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_move_interval_ms")]
    pub move_interval_ms: u64,

    #[serde(default = "default_frontend_dir")]
    pub frontend_dir: PathBuf,

    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    #[serde(default = "default_training_dir")]
    pub training_dir: PathBuf,
}

fn default_port() -> u16 {
    3000
}
fn default_move_interval_ms() -> u64 {
    500
}
fn default_frontend_dir() -> PathBuf {
    PathBuf::from("../frontend/dist")
}
fn default_models_dir() -> PathBuf {
    PathBuf::from("../training")
}
fn default_training_dir() -> PathBuf {
    PathBuf::from("../training")
}

impl ServerConfig {
    /// Loads config from the given path, or the default path, or falls back
    /// to defaults if no config file exists.
    pub fn load() -> Self {
        let config_path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_string());

        let path = Path::new(&config_path);
        if path.exists() {
            let contents = std::fs::read_to_string(path)
                .unwrap_or_else(|err| panic!("Failed to read config {}: {err}", path.display()));
            let config: ServerConfig = toml::from_str(&contents)
                .unwrap_or_else(|err| panic!("Failed to parse config {}: {err}", path.display()));
            println!("Loaded config from {}", path.display());
            config
        } else if config_path == DEFAULT_CONFIG_PATH {
            // Default path doesn't exist — use all defaults silently
            println!("No config file found, using defaults");
            toml::from_str("").unwrap()
        } else {
            // Explicit path doesn't exist — that's an error
            panic!("Config file not found: {}", path.display());
        }
    }
}
