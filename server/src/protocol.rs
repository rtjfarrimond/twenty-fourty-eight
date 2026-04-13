use game_engine::Direction;
use serde::{Deserialize, Serialize};

/// Messages sent from the client to the server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    NewGame,
    Move { direction: ClientDirection },
}

/// Direction as received from the client (serialization-friendly).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClientDirection {
    Up,
    Down,
    Left,
    Right,
}

impl From<ClientDirection> for Direction {
    fn from(direction: ClientDirection) -> Self {
        match direction {
            ClientDirection::Up => Direction::Up,
            ClientDirection::Down => Direction::Down,
            ClientDirection::Left => Direction::Left,
            ClientDirection::Right => Direction::Right,
        }
    }
}

/// Messages sent from the server to the client.
#[derive(Debug, Serialize)]
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

/// Converts a game engine Board into the 2D array of tile face values
/// (0 for empty, 2, 4, 8, ...) that the client expects.
pub fn board_to_tile_values(board: &game_engine::Board) -> [[u16; 4]; 4] {
    let mut tiles = [[0u16; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            let exponent = board.get_tile(row, col);
            tiles[row][col] = if exponent == 0 { 0 } else { 1 << exponent };
        }
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_engine::Board;

    #[test]
    fn board_to_tile_values_converts_exponents_to_face_values() {
        let mut board = Board::new();
        board.set_tile(0, 0, 1); // 2
        board.set_tile(1, 1, 3); // 8
        board.set_tile(2, 2, 11); // 2048

        let tiles = board_to_tile_values(&board);
        assert_eq!(tiles[0][0], 2);
        assert_eq!(tiles[1][1], 8);
        assert_eq!(tiles[2][2], 2048);
        assert_eq!(tiles[3][3], 0); // empty
    }

    #[test]
    fn client_direction_converts_to_engine_direction() {
        assert_eq!(Direction::from(ClientDirection::Up), Direction::Up);
        assert_eq!(Direction::from(ClientDirection::Down), Direction::Down);
        assert_eq!(Direction::from(ClientDirection::Left), Direction::Left);
        assert_eq!(Direction::from(ClientDirection::Right), Direction::Right);
    }

    #[test]
    fn client_message_deserializes_new_game() {
        let json = r#"{"type":"NewGame"}"#;
        let message: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(message, ClientMessage::NewGame));
    }

    #[test]
    fn client_message_deserializes_move() {
        let json = r#"{"type":"Move","direction":"left"}"#;
        let message: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(
            message,
            ClientMessage::Move {
                direction: ClientDirection::Left
            }
        ));
    }

    #[test]
    fn server_message_serializes_game_state() {
        let message = ServerMessage::GameState {
            board: [[0; 4]; 4],
            score: 100,
            game_over: false,
        };
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"type\":\"GameState\""));
        assert!(json.contains("\"score\":100"));
    }
}
