//! User-facing operations on the queue: submit, list, cancel, reap.
//!
//! Kept separate from the storage layer (`queue::QueueDir`) and the CLI
//! (`main.rs`) so each piece is testable in isolation. The CLI is a thin
//! adapter that calls into these functions.

use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::queue::{ActiveState, Job, JobId, JobState, QueueDir, TerminalState};
use crate::run_args::RunArgs;

/// Write a new job to the pending queue. Returns the assigned job id.
pub fn submit(
    queue: &QueueDir,
    args: RunArgs,
    submitted_by: String,
) -> io::Result<JobId> {
    queue.ensure_dirs()?;
    let job = Job::new(args, submitted_by);
    let id = job.id.clone();
    queue.write(&job, ActiveState::Pending)?;
    Ok(id)
}

/// Cancel a pending job. Returns `Ok(())` on success.
/// Returns `Err` with `NotFound` if the job isn't in pending — the caller
/// can then call `find` to report the actual current state.
pub fn cancel_pending(queue: &QueueDir, id: &JobId) -> io::Result<()> {
    queue.transition(id, ActiveState::Pending, TerminalState::Cancelled)
}

/// Snapshot of the queue: jobs grouped by state, each fully loaded.
/// Used by both the CLI list command and the HTTP `/queue/stream` endpoint.
#[derive(Debug, serde::Serialize)]
pub struct QueueSnapshot {
    pub pending: Vec<Job>,
    pub running: Vec<Job>,
    pub completed: Vec<Job>,
    pub failed: Vec<Job>,
    pub cancelled: Vec<Job>,
}

impl QueueSnapshot {
    pub fn load(queue: &QueueDir) -> io::Result<Self> {
        Ok(Self {
            pending: load_state(queue, ActiveState::Pending)?,
            running: load_state(queue, ActiveState::Running)?,
            completed: load_state(queue, TerminalState::Completed)?,
            failed: load_state(queue, TerminalState::Failed)?,
            cancelled: load_state(queue, TerminalState::Cancelled)?,
        })
    }

    pub fn total(&self) -> usize {
        self.pending.len()
            + self.running.len()
            + self.completed.len()
            + self.failed.len()
            + self.cancelled.len()
    }
}

/// Remove terminal jobs older than `max_age`. Returns the count removed.
/// Only operates on `TerminalState` variants — the type system prevents
/// accidentally reaping active (pending/running) jobs.
pub fn reap_expired(queue: &QueueDir, max_age: Duration) -> io::Result<usize> {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let cutoff_secs = now_secs.saturating_sub(max_age.as_secs());
    let mut reaped = 0;
    for terminal_state in TerminalState::all() {
        for id in queue.list(terminal_state)? {
            if id.submitted_at_unix() < cutoff_secs {
                match queue.remove(&id, terminal_state) {
                    Ok(()) => reaped += 1,
                    Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                    Err(err) => return Err(err),
                }
            }
        }
    }
    Ok(reaped)
}

fn load_state(queue: &QueueDir, state: impl Into<JobState> + Copy) -> io::Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for id in queue.list(state)? {
        // Skip jobs that vanish between list and read — race with the daemon
        // moving them to a different state. Don't fail the whole listing.
        match queue.read(&id, state) {
            Ok(job) => jobs.push(job),
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

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

    fn fresh_queue() -> (tempfile::TempDir, QueueDir) {
        let dir = tempfile::tempdir().unwrap();
        let queue = QueueDir::new(dir.path());
        queue.ensure_dirs().unwrap();
        (dir, queue)
    }

    #[test]
    fn submit_writes_pending_job_and_returns_id() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        assert_eq!(queue.find(&id).unwrap(), Some(ActiveState::Pending.into()));
    }

    #[test]
    fn submit_creates_dirs_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let queue = QueueDir::new(dir.path().join("brand-new-queue"));
        // No ensure_dirs call — submit should self-bootstrap
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        assert_eq!(queue.find(&id).unwrap(), Some(ActiveState::Pending.into()));
    }

    #[test]
    fn cancel_pending_moves_to_cancelled() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        cancel_pending(&queue, &id).unwrap();
        assert_eq!(queue.find(&id).unwrap(), Some(TerminalState::Cancelled.into()));
    }

    #[test]
    fn cancel_pending_fails_on_running_job() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        queue
            .transition(&id, ActiveState::Pending, ActiveState::Running)
            .unwrap();
        let result = cancel_pending(&queue, &id);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn snapshot_groups_jobs_by_state() {
        let (_dir, queue) = fresh_queue();
        let id1 = submit(&queue, sample_args(), "alice".into()).unwrap();
        let id2 = submit(&queue, sample_args(), "alice".into()).unwrap();
        let id3 = submit(&queue, sample_args(), "alice".into()).unwrap();
        queue.transition(&id2, ActiveState::Pending, ActiveState::Running).unwrap();
        queue.transition(&id3, ActiveState::Pending, TerminalState::Completed).unwrap();

        let snap = QueueSnapshot::load(&queue).unwrap();
        assert_eq!(snap.pending.len(), 1);
        assert_eq!(snap.pending[0].id.as_str(), id1.as_str());
        assert_eq!(snap.running.len(), 1);
        assert_eq!(snap.running[0].id.as_str(), id2.as_str());
        assert_eq!(snap.completed.len(), 1);
        assert_eq!(snap.completed[0].id.as_str(), id3.as_str());
        assert_eq!(snap.failed.len(), 0);
        assert_eq!(snap.cancelled.len(), 0);
        assert_eq!(snap.total(), 3);
    }

    /// Create a job with an artificially old timestamp for retention tests.
    fn submit_with_age(
        queue: &QueueDir,
        age: Duration,
    ) -> JobId {
        use std::time::{SystemTime, UNIX_EPOCH};
        let old_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - age.as_secs();
        let suffix: u32 = rand::Rng::random_range(&mut rand::rng(), 0..=0xFF_FFFF);
        let raw_id = format!("{old_secs:010}_{suffix:06x}");
        let id = JobId::parse(&raw_id).unwrap();
        let job = Job {
            id: id.clone(),
            submitted_at_unix: old_secs,
            submitted_by: "test".into(),
            args: sample_args(),
        };
        queue.write(&job, ActiveState::Pending).unwrap();
        id
    }

    #[test]
    fn reap_expired_removes_old_terminal_jobs() {
        let (_dir, queue) = fresh_queue();
        let max_age = Duration::from_secs(86_400); // 1 day

        // Old completed job (2 days old) — should be reaped
        let old_id = submit_with_age(&queue, Duration::from_secs(2 * 86_400));
        queue.transition(&old_id, ActiveState::Pending, TerminalState::Completed).unwrap();

        // Recent completed job (1 hour old) — should survive
        let recent_id = submit_with_age(&queue, Duration::from_secs(3_600));
        queue.transition(&recent_id, ActiveState::Pending, TerminalState::Completed).unwrap();

        let reaped = reap_expired(&queue, max_age).unwrap();

        assert_eq!(reaped, 1);
        assert_eq!(queue.find(&old_id).unwrap(), None);
        assert_eq!(queue.find(&recent_id).unwrap(), Some(TerminalState::Completed.into()));
    }

    #[test]
    fn reap_expired_never_touches_active_jobs() {
        let (_dir, queue) = fresh_queue();
        let max_age = Duration::from_secs(86_400);

        // Old pending job — must NOT be reaped
        let old_pending = submit_with_age(&queue, Duration::from_secs(2 * 86_400));

        let reaped = reap_expired(&queue, max_age).unwrap();

        assert_eq!(reaped, 0);
        assert_eq!(queue.find(&old_pending).unwrap(), Some(ActiveState::Pending.into()));
    }

    #[test]
    fn reap_expired_covers_all_terminal_states() {
        let (_dir, queue) = fresh_queue();
        let max_age = Duration::from_secs(86_400);

        let completed = submit_with_age(&queue, Duration::from_secs(2 * 86_400));
        queue.transition(&completed, ActiveState::Pending, TerminalState::Completed).unwrap();

        let failed = submit_with_age(&queue, Duration::from_secs(2 * 86_400));
        queue.transition(&failed, ActiveState::Pending, TerminalState::Failed).unwrap();

        let cancelled = submit_with_age(&queue, Duration::from_secs(2 * 86_400));
        queue.transition(&cancelled, ActiveState::Pending, TerminalState::Cancelled).unwrap();

        let reaped = reap_expired(&queue, max_age).unwrap();

        assert_eq!(reaped, 3);
        assert_eq!(queue.find(&completed).unwrap(), None);
        assert_eq!(queue.find(&failed).unwrap(), None);
        assert_eq!(queue.find(&cancelled).unwrap(), None);
    }

    #[test]
    fn snapshot_preserves_job_payload() {
        let (_dir, queue) = fresh_queue();
        let mut args = sample_args();
        args.model_name = "my-special-model".into();
        args.games = 9_999_999;
        args.models_dir = PathBuf::from("/tmp/x");
        let id = submit(&queue, args, "bob".into()).unwrap();

        let snap = QueueSnapshot::load(&queue).unwrap();
        let job = snap.pending.iter().find(|j| j.id == id).unwrap();
        assert_eq!(job.submitted_by, "bob");
        assert_eq!(job.args.model_name, "my-special-model");
        assert_eq!(job.args.games, 9_999_999);
        assert_eq!(job.args.models_dir, PathBuf::from("/tmp/x"));
    }
}
