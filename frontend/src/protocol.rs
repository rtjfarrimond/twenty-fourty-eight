use serde::{Deserialize, Serialize};

/// Messages sent from the client to the server.
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    NewGame,
    Move { direction: String },
    WatchAgent,
}

/// Messages received from the server.
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    GameState {
        board: [[u16; 4]; 4],
        score: u32,
        game_over: bool,
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

    pub fn watch_agent() -> String {
        serde_json::to_string(&ClientMessage::WatchAgent).unwrap()
    }
}
