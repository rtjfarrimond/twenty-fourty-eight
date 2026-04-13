use game_engine::{
    Board, Direction, MoveTables, apply_move, empty_tiles, is_game_over, spawn_tile,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// A single game session's state.
pub struct GameSession {
    pub board: Board,
    pub score: u32,
    pub last_activity: Instant,
}

/// Manages all active game sessions.
pub struct SessionManager {
    sessions: HashMap<String, GameSession>,
    tables: Arc<MoveTables>,
}

impl SessionManager {
    pub fn new(tables: Arc<MoveTables>) -> Self {
        Self {
            sessions: HashMap::new(),
            tables,
        }
    }

    /// Creates a new game session with two random tiles spawned.
    /// Returns the session ID and initial board state.
    pub fn create_session(&mut self) -> (String, &GameSession) {
        let session_id = Uuid::new_v4().to_string();
        let mut board = Board::new();
        let mut rng = rand::rng();

        // Spawn two initial tiles
        for _ in 0..2 {
            let empties = empty_tiles(&board);
            let position = empties[rng.random_range(0..empties.len())];
            board = spawn_tile(&board, position.0, position.1, rng.random::<f64>());
        }

        let session = GameSession {
            board,
            score: 0,
            last_activity: Instant::now(),
        };

        self.sessions.insert(session_id.clone(), session);
        (session_id.clone(), self.sessions.get(&session_id).unwrap())
    }

    /// Applies a move to an existing session. Returns the updated session,
    /// or None if the session doesn't exist or the move is illegal.
    pub fn apply_move(&mut self, session_id: &str, direction: Direction) -> Option<&GameSession> {
        let session = self.sessions.get_mut(session_id)?;
        session.last_activity = Instant::now();

        let (new_board, move_score) = apply_move(&session.board, direction, &self.tables)?;

        // Spawn a new tile after the move
        let mut rng = rand::rng();
        let empties = empty_tiles(&new_board);
        let board_with_spawn = if empties.is_empty() {
            new_board
        } else {
            let position = empties[rng.random_range(0..empties.len())];
            spawn_tile(&new_board, position.0, position.1, rng.random::<f64>())
        };

        session.board = board_with_spawn;
        session.score += move_score;

        Some(session)
    }

    pub fn get_session(&self, session_id: &str) -> Option<&GameSession> {
        self.sessions.get(session_id)
    }

    pub fn remove_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    pub fn is_game_over(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .map(|session| is_game_over(&session.board, &self.tables))
            .unwrap_or(true)
    }

    /// Removes sessions that have been idle longer than the given duration.
    pub fn cleanup_stale_sessions(&mut self, max_idle: std::time::Duration) {
        let now = Instant::now();
        self.sessions
            .retain(|_, session| now.duration_since(session.last_activity) < max_idle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> Arc<MoveTables> {
        Arc::new(MoveTables::new())
    }

    #[test]
    fn create_session_returns_board_with_two_tiles() {
        let mut manager = SessionManager::new(tables());
        let (_, session) = manager.create_session();

        let occupied_count = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .filter(|&(row, col)| session.board.get_tile(row, col) != 0)
            .count();

        assert_eq!(occupied_count, 2);
        assert_eq!(session.score, 0);
    }

    #[test]
    fn apply_move_updates_board_and_score() {
        let tables = tables();
        let mut manager = SessionManager::new(tables);

        let (session_id, _) = manager.create_session();

        // We can't predict the exact board, but we can try all directions
        // and at least one should succeed (a fresh board always has legal moves)
        let directions = [
            Direction::Left,
            Direction::Right,
            Direction::Up,
            Direction::Down,
        ];

        let moved = directions
            .iter()
            .any(|&dir| manager.apply_move(&session_id, dir).is_some());

        assert!(moved, "at least one direction should be a legal move");
    }

    #[test]
    fn apply_move_on_nonexistent_session_returns_none() {
        let mut manager = SessionManager::new(tables());
        assert!(manager.apply_move("nonexistent", Direction::Left).is_none());
    }

    #[test]
    fn remove_session_makes_it_inaccessible() {
        let mut manager = SessionManager::new(tables());
        let (session_id, _) = manager.create_session();

        manager.remove_session(&session_id);
        assert!(manager.get_session(&session_id).is_none());
    }

    #[test]
    fn cleanup_removes_stale_sessions() {
        let mut manager = SessionManager::new(tables());
        let (session_id, _) = manager.create_session();

        // Set last_activity to the past
        manager.sessions.get_mut(&session_id).unwrap().last_activity =
            Instant::now() - std::time::Duration::from_secs(600);

        manager.cleanup_stale_sessions(std::time::Duration::from_secs(300));
        assert!(manager.get_session(&session_id).is_none());
    }

    #[test]
    fn cleanup_keeps_active_sessions() {
        let mut manager = SessionManager::new(tables());
        let (session_id, _) = manager.create_session();

        manager.cleanup_stale_sessions(std::time::Duration::from_secs(300));
        assert!(manager.get_session(&session_id).is_some());
    }
}
