//! User-facing operations on the queue: submit, list, cancel.
//!
//! Kept separate from the storage layer (`queue::QueueDir`) and the CLI
//! (`main.rs`) so each piece is testable in isolation. The CLI is a thin
//! adapter that calls into these functions.

use std::io;

use crate::queue::{Job, JobId, JobState, QueueDir};
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
    queue.write(&job, JobState::Pending)?;
    Ok(id)
}

/// Cancel a pending job. Returns `Ok(())` on success.
/// Returns `Err` with `NotFound` if the job isn't in pending — the caller
/// can then call `find` to report the actual current state.
pub fn cancel_pending(queue: &QueueDir, id: &JobId) -> io::Result<()> {
    queue.transition(id, JobState::Pending, JobState::Cancelled)
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
            pending: load_state(queue, JobState::Pending)?,
            running: load_state(queue, JobState::Running)?,
            completed: load_state(queue, JobState::Completed)?,
            failed: load_state(queue, JobState::Failed)?,
            cancelled: load_state(queue, JobState::Cancelled)?,
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

fn load_state(queue: &QueueDir, state: JobState) -> io::Result<Vec<Job>> {
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
        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Pending));
    }

    #[test]
    fn submit_creates_dirs_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let queue = QueueDir::new(dir.path().join("brand-new-queue"));
        // No ensure_dirs call — submit should self-bootstrap
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Pending));
    }

    #[test]
    fn cancel_pending_moves_to_cancelled() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        cancel_pending(&queue, &id).unwrap();
        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Cancelled));
    }

    #[test]
    fn cancel_pending_fails_on_running_job() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        queue
            .transition(&id, JobState::Pending, JobState::Running)
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
        queue.transition(&id2, JobState::Pending, JobState::Running).unwrap();
        queue.transition(&id3, JobState::Pending, JobState::Completed).unwrap();

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
