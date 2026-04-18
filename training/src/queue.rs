//! Persistent training-job queue backed by directory-state filesystem layout.
//!
//! Each lifecycle state is its own subdirectory under the queue root. State
//! transitions are atomic `rename(2)` calls between sibling subdirectories,
//! which is safe across crashes and across daemon restarts (no in-memory
//! state to lose, no lock files to clean up).
//!
//! ```text
//! /var/lib/2048-solver/queue/
//!   pending/   <id>.json   — submitted, waiting for the daemon
//!   running/   <id>.json   — currently being executed (at most one)
//!   completed/ <id>.json   — finished successfully
//!   failed/    <id>.json   — exited non-zero, see captured stderr inside
//!   cancelled/ <id>.json   — cancelled by user before it ran
//! ```

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::run_args::RunArgs;

/// Non-terminal lifecycle states: the job is still in flight.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ActiveState {
    Pending,
    Running,
}

impl ActiveState {
    pub fn all() -> [ActiveState; 2] {
        [ActiveState::Pending, ActiveState::Running]
    }

    pub fn dir_name(self) -> &'static str {
        match self {
            ActiveState::Pending => "pending",
            ActiveState::Running => "running",
        }
    }
}

/// Terminal lifecycle states: the job has reached a final outcome.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TerminalState {
    Completed,
    Failed,
    Cancelled,
}

impl TerminalState {
    pub fn all() -> [TerminalState; 3] {
        [
            TerminalState::Completed,
            TerminalState::Failed,
            TerminalState::Cancelled,
        ]
    }

    pub fn dir_name(self) -> &'static str {
        match self {
            TerminalState::Completed => "completed",
            TerminalState::Failed => "failed",
            TerminalState::Cancelled => "cancelled",
        }
    }
}

/// The five possible lifecycle positions for a queued job. Each maps to a
/// dedicated subdirectory in the queue root. Partitioned into `Active` and
/// `Terminal` so the type system enforces which operations apply where.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum JobState {
    Active(ActiveState),
    Terminal(TerminalState),
}

impl JobState {
    pub fn all() -> [JobState; 5] {
        [
            JobState::Active(ActiveState::Pending),
            JobState::Active(ActiveState::Running),
            JobState::Terminal(TerminalState::Completed),
            JobState::Terminal(TerminalState::Failed),
            JobState::Terminal(TerminalState::Cancelled),
        ]
    }

    pub fn dir_name(self) -> &'static str {
        match self {
            JobState::Active(state) => state.dir_name(),
            JobState::Terminal(state) => state.dir_name(),
        }
    }
}

impl From<ActiveState> for JobState {
    fn from(state: ActiveState) -> Self {
        JobState::Active(state)
    }
}

impl From<TerminalState> for JobState {
    fn from(state: TerminalState) -> Self {
        JobState::Terminal(state)
    }
}

impl Serialize for JobState {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.dir_name())
    }
}

impl<'de> Deserialize<'de> for JobState {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        match name.as_str() {
            "pending" => Ok(JobState::Active(ActiveState::Pending)),
            "running" => Ok(JobState::Active(ActiveState::Running)),
            "completed" => Ok(JobState::Terminal(TerminalState::Completed)),
            "failed" => Ok(JobState::Terminal(TerminalState::Failed)),
            "cancelled" => Ok(JobState::Terminal(TerminalState::Cancelled)),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["pending", "running", "completed", "failed", "cancelled"],
            )),
        }
    }
}

/// Sortable, collision-resistant identifier for a queued job.
///
/// Format: `<10-digit-unix-secs>_<6-hex-random>`, e.g. `1744740754_a3f2b1`.
/// The padded numeric prefix means lexicographic sort = chronological sort,
/// which is exactly what the FIFO daemon needs.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct JobId(String);

impl JobId {
    pub fn generate() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let suffix: u32 = rand::rng().random_range(0..=0xFF_FFFF);
        Self(format!("{secs:010}_{suffix:06x}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Extract the submission unix timestamp from the id prefix.
    /// Safe to call on any valid `JobId` (the constructor and parser both
    /// guarantee the first 10 chars are ascii digits).
    pub fn submitted_at_unix(&self) -> u64 {
        self.0[..10].parse().expect("JobId prefix is always 10 ascii digits")
    }

    /// Validate a user-supplied id: must match the generated format.
    /// Stricter than necessary but rejects path-traversal and stray chars.
    pub fn parse(input: &str) -> Result<Self, String> {
        if input.len() != 17 {
            return Err(format!("expected 17-char id, got {} chars", input.len()));
        }
        let bytes = input.as_bytes();
        if bytes[10] != b'_' {
            return Err("expected underscore at position 10".into());
        }
        if !bytes[..10].iter().all(|b| b.is_ascii_digit()) {
            return Err("expected 10 digits before underscore".into());
        }
        if !bytes[11..].iter().all(|b| b.is_ascii_hexdigit()) {
            return Err("expected 6 hex chars after underscore".into());
        }
        Ok(Self(input.to_string()))
    }
}

/// A queued job: its identity, when/who submitted it, and the args needed
/// to execute it. Serialized to JSON in the queue directory.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Job {
    pub id: JobId,
    pub submitted_at_unix: u64,
    pub submitted_by: String,
    pub args: RunArgs,
}

impl Job {
    pub fn new(args: RunArgs, submitted_by: String) -> Self {
        let submitted_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            id: JobId::generate(),
            submitted_at_unix,
            submitted_by,
            args,
        }
    }
}

/// Filesystem-backed queue storage. All operations are atomic at the
/// `rename(2)` level on a single filesystem.
pub struct QueueDir {
    root: PathBuf,
}

impl QueueDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Create the root and all five state subdirectories. Idempotent.
    pub fn ensure_dirs(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        for state in JobState::all() {
            fs::create_dir_all(self.dir_for(state))?;
        }
        Ok(())
    }

    pub fn dir_for(&self, state: impl Into<JobState>) -> PathBuf {
        self.root.join(state.into().dir_name())
    }

    fn path_for(&self, id: &JobId, state: impl Into<JobState>) -> PathBuf {
        self.dir_for(state).join(format!("{}.json", id.as_str()))
    }

    /// Write a job atomically: serialize to a temp file, then rename into
    /// place. Concurrent readers see either the old file or the new one,
    /// never a partial write.
    pub fn write(&self, job: &Job, state: impl Into<JobState>) -> io::Result<()> {
        let final_path = self.path_for(&job.id, state);
        let temp_path = final_path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(job)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))?;
        fs::write(&temp_path, json)?;
        fs::rename(&temp_path, &final_path)?;
        Ok(())
    }

    /// Read a job from a specific state directory.
    pub fn read(&self, id: &JobId, state: impl Into<JobState>) -> io::Result<Job> {
        let path = self.path_for(id, state);
        let contents = fs::read_to_string(&path)?;
        serde_json::from_str(&contents)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))
    }

    /// Atomic state transition. Returns `NotFound` if the job isn't in the
    /// `from` state, which is the expected outcome when two daemons race
    /// to claim the same job.
    pub fn transition(
        &self,
        id: &JobId,
        from: impl Into<JobState>,
        to: impl Into<JobState>,
    ) -> io::Result<()> {
        let source = self.path_for(id, from);
        let destination = self.path_for(id, to);
        fs::rename(&source, &destination)
    }

    /// List job ids currently in `state`, sorted lexicographically (which
    /// equals chronological order due to JobId's padded-unix-secs prefix).
    pub fn list(&self, state: impl Into<JobState>) -> io::Result<Vec<JobId>> {
        let dir = self.dir_for(state);
        let mut ids = Vec::new();
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(ids),
            Err(err) => return Err(err),
        };
        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(stem) = name.strip_suffix(".json")
                && let Ok(id) = JobId::parse(stem)
            {
                ids.push(id);
            }
        }
        ids.sort_by(|left, right| left.as_str().cmp(right.as_str()));
        Ok(ids)
    }

    /// Remove a job file from the given state directory. Returns
    /// `NotFound` if the file doesn't exist (race with daemon).
    pub fn remove(&self, id: &JobId, state: impl Into<JobState>) -> io::Result<()> {
        let path = self.path_for(id, state);
        fs::remove_file(&path)
    }

    /// Find which state (if any) currently holds the given job id.
    pub fn find(&self, id: &JobId) -> io::Result<Option<JobState>> {
        for state in JobState::all() {
            if self.path_for(id, state).exists() {
                return Ok(Some(state));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;

    fn sample_run_args() -> RunArgs {
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

    fn fresh_queue() -> (tempfile::TempDir, QueueDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let queue = QueueDir::new(dir.path());
        queue.ensure_dirs().expect("ensure_dirs");
        (dir, queue)
    }

    #[test]
    fn terminal_states_cover_completed_failed_cancelled() {
        let states = TerminalState::all();
        assert_eq!(states.len(), 3);
        assert!(states.contains(&TerminalState::Completed));
        assert!(states.contains(&TerminalState::Failed));
        assert!(states.contains(&TerminalState::Cancelled));
    }

    #[test]
    fn active_states_cover_pending_running() {
        let states = ActiveState::all();
        assert_eq!(states.len(), 2);
        assert!(states.contains(&ActiveState::Pending));
        assert!(states.contains(&ActiveState::Running));
    }

    #[test]
    fn job_state_from_active_and_terminal() {
        let pending: JobState = ActiveState::Pending.into();
        let completed: JobState = TerminalState::Completed.into();
        assert!(matches!(pending, JobState::Active(ActiveState::Pending)));
        assert!(matches!(completed, JobState::Terminal(TerminalState::Completed)));
    }

    #[test]
    fn job_id_format_is_padded_secs_and_hex() {
        let id = JobId::generate();
        let s = id.as_str();
        assert_eq!(s.len(), 17);
        assert_eq!(s.as_bytes()[10], b'_');
        assert!(s[..10].chars().all(|c| c.is_ascii_digit()));
        assert!(s[11..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn job_ids_collision_resistant_across_many_calls() {
        let mut seen = HashSet::new();
        for _ in 0..1000 {
            seen.insert(JobId::generate().as_str().to_string());
        }
        assert_eq!(seen.len(), 1000);
    }

    #[test]
    fn job_id_parse_round_trips_generated_id() {
        let original = JobId::generate();
        let parsed = JobId::parse(original.as_str()).expect("parse");
        assert_eq!(original.as_str(), parsed.as_str());
    }

    #[test]
    fn job_id_submitted_at_unix_extracts_timestamp() {
        let id = JobId::parse("1744740754_a3f2b1").unwrap();
        assert_eq!(id.submitted_at_unix(), 1744740754);
    }

    #[test]
    fn job_id_parse_rejects_path_traversal() {
        assert!(JobId::parse("../etc/passwd").is_err());
        assert!(JobId::parse("0000000000_zzzzzz").is_err());
        assert!(JobId::parse("short").is_err());
    }

    #[test]
    fn ensure_dirs_creates_all_state_subdirs() {
        let (_dir, queue) = fresh_queue();
        for state in JobState::all() {
            assert!(queue.dir_for(state).is_dir(), "{state:?} dir missing");
        }
    }

    #[test]
    fn write_then_read_round_trips_job() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        let read = queue.read(&job.id, ActiveState::Pending).unwrap();
        assert_eq!(read.id.as_str(), job.id.as_str());
        assert_eq!(read.submitted_by, "alice");
        assert_eq!(read.args.model_name, "test");
    }

    #[test]
    fn list_returns_pending_in_chronological_order() {
        let (_dir, queue) = fresh_queue();
        let mut written_ids = Vec::new();
        for _ in 0..3 {
            let job = Job::new(sample_run_args(), "alice".into());
            queue.write(&job, ActiveState::Pending).unwrap();
            written_ids.push(job.id.0.clone());
            // Ensure distinct ids (random suffix usually does this, but be safe)
            thread::sleep(Duration::from_millis(2));
        }
        written_ids.sort();
        let listed: Vec<String> = queue
            .list(ActiveState::Pending)
            .unwrap()
            .into_iter()
            .map(|id| id.0)
            .collect();
        assert_eq!(listed, written_ids);
    }

    #[test]
    fn list_on_empty_state_returns_empty_vec() {
        let (_dir, queue) = fresh_queue();
        assert!(queue.list(ActiveState::Running).unwrap().is_empty());
    }

    #[test]
    fn transition_moves_file_between_state_dirs() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        queue
            .transition(&job.id, ActiveState::Pending, ActiveState::Running)
            .unwrap();
        assert!(queue.list(ActiveState::Pending).unwrap().is_empty());
        let running = queue.list(ActiveState::Running).unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].as_str(), job.id.as_str());
    }

    #[test]
    fn transition_fails_when_source_state_wrong() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        let result = queue.transition(
            &job.id,
            ActiveState::Running,
            TerminalState::Completed,
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
    }

    #[test]
    fn find_returns_current_state() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        assert_eq!(
            queue.find(&job.id).unwrap(),
            Some(ActiveState::Pending.into()),
        );
        queue
            .transition(&job.id, ActiveState::Pending, TerminalState::Failed)
            .unwrap();
        assert_eq!(
            queue.find(&job.id).unwrap(),
            Some(TerminalState::Failed.into()),
        );
    }

    #[test]
    fn find_returns_none_for_unknown_job() {
        let (_dir, queue) = fresh_queue();
        let id = JobId::generate();
        assert_eq!(queue.find(&id).unwrap(), None);
    }

    #[test]
    fn write_is_atomic_no_partial_files_visible() {
        // Verifies the temp-then-rename pattern by checking no .tmp file
        // remains after a successful write.
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        let entries: Vec<PathBuf> = fs::read_dir(queue.dir_for(ActiveState::Pending))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0].extension().and_then(|s| s.to_str()) == Some("json"),
            "expected only the .json file, found {:?}",
            entries[0]
        );
    }

    #[test]
    fn list_ignores_files_with_invalid_id_format() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, ActiveState::Pending).unwrap();
        // Drop a junk file in the same dir
        fs::write(queue.dir_for(ActiveState::Pending).join("not-a-job.json"), "{}").unwrap();
        fs::write(queue.dir_for(ActiveState::Pending).join("README"), "x").unwrap();
        let listed = queue.list(ActiveState::Pending).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].as_str(), job.id.as_str());
    }
}
