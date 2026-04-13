use serde::{Deserialize, Serialize};

/// Messages sent from the client to the server.
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    NewGame,
    Move { direction: String },
    WatchAgent { model: String },
}

#[derive(Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
}

/// Messages received from the server.
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    GameState {
        board: [[u16; 4]; 4],
        score: u32,
        game_over: bool,
        #[serde(default)]
        last_move: Option<String>,
    },
    ModelList {
        models: Vec<ModelInfo>,
    },
    Error {
        message: String,
    },
}

impl ClientMessage {
    pub fn new_game() -> String {
        serde_json::to_string(&ClientMessage::NewGame).unwrap()
    }

    pub fn make_move(direction: &str) -> String {
        serde_json::to_string(&ClientMessage::Move {
            direction: direction.to_string(),
        })
        .unwrap()
    }

    pub fn watch_agent(model: &str) -> String {
        serde_json::to_string(&ClientMessage::WatchAgent {
            model: model.to_string(),
        })
        .unwrap()
    }
}
