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

/// The five possible lifecycle positions for a queued job. Each maps to a
/// dedicated subdirectory in the queue root.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn all() -> [JobState; 5] {
        [
            JobState::Pending,
            JobState::Running,
            JobState::Completed,
            JobState::Failed,
            JobState::Cancelled,
        ]
    }

    pub fn dir_name(self) -> &'static str {
        match self {
            JobState::Pending => "pending",
            JobState::Running => "running",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
            JobState::Cancelled => "cancelled",
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

    pub fn dir_for(&self, state: JobState) -> PathBuf {
        self.root.join(state.dir_name())
    }

    fn path_for(&self, id: &JobId, state: JobState) -> PathBuf {
        self.dir_for(state).join(format!("{}.json", id.as_str()))
    }

    /// Write a job atomically: serialize to a temp file, then rename into
    /// place. Concurrent readers see either the old file or the new one,
    /// never a partial write.
    pub fn write(&self, job: &Job, state: JobState) -> io::Result<()> {
        let final_path = self.path_for(&job.id, state);
        let temp_path = final_path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(job)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err))?;
        fs::write(&temp_path, json)?;
        fs::rename(&temp_path, &final_path)?;
        Ok(())
    }

    /// Read a job from a specific state directory.
    pub fn read(&self, id: &JobId, state: JobState) -> io::Result<Job> {
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
        from: JobState,
        to: JobState,
    ) -> io::Result<()> {
        let source = self.path_for(id, from);
        let destination = self.path_for(id, to);
        fs::rename(&source, &destination)
    }

    /// List job ids currently in `state`, sorted lexicographically (which
    /// equals chronological order due to JobId's padded-unix-secs prefix).
    pub fn list(&self, state: JobState) -> io::Result<Vec<JobId>> {
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
        }
    }

    fn fresh_queue() -> (tempfile::TempDir, QueueDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let queue = QueueDir::new(dir.path());
        queue.ensure_dirs().expect("ensure_dirs");
        (dir, queue)
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
        queue.write(&job, JobState::Pending).unwrap();
        let read = queue.read(&job.id, JobState::Pending).unwrap();
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
            queue.write(&job, JobState::Pending).unwrap();
            written_ids.push(job.id.0.clone());
            // Ensure distinct ids (random suffix usually does this, but be safe)
            thread::sleep(Duration::from_millis(2));
        }
        written_ids.sort();
        let listed: Vec<String> = queue
            .list(JobState::Pending)
            .unwrap()
            .into_iter()
            .map(|id| id.0)
            .collect();
        assert_eq!(listed, written_ids);
    }

    #[test]
    fn list_on_empty_state_returns_empty_vec() {
        let (_dir, queue) = fresh_queue();
        assert!(queue.list(JobState::Running).unwrap().is_empty());
    }

    #[test]
    fn transition_moves_file_between_state_dirs() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, JobState::Pending).unwrap();
        queue
            .transition(&job.id, JobState::Pending, JobState::Running)
            .unwrap();
        assert!(queue.list(JobState::Pending).unwrap().is_empty());
        let running = queue.list(JobState::Running).unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].as_str(), job.id.as_str());
    }

    #[test]
    fn transition_fails_when_source_state_wrong() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, JobState::Pending).unwrap();
        let result = queue.transition(&job.id, JobState::Running, JobState::Completed);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ErrorKind::NotFound);
    }

    #[test]
    fn find_returns_current_state() {
        let (_dir, queue) = fresh_queue();
        let job = Job::new(sample_run_args(), "alice".into());
        queue.write(&job, JobState::Pending).unwrap();
        assert_eq!(queue.find(&job.id).unwrap(), Some(JobState::Pending));
        queue
            .transition(&job.id, JobState::Pending, JobState::Failed)
            .unwrap();
        assert_eq!(queue.find(&job.id).unwrap(), Some(JobState::Failed));
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
        queue.write(&job, JobState::Pending).unwrap();
        let entries: Vec<PathBuf> = fs::read_dir(queue.dir_for(JobState::Pending))
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
        queue.write(&job, JobState::Pending).unwrap();
        // Drop a junk file in the same dir
        fs::write(queue.dir_for(JobState::Pending).join("not-a-job.json"), "{}").unwrap();
        fs::write(queue.dir_for(JobState::Pending).join("README"), "x").unwrap();
        let listed = queue.list(JobState::Pending).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].as_str(), job.id.as_str());
    }
}
