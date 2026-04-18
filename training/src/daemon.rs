//! Long-running daemon that drains the job queue.
//!
//! - On startup: any `running/` file is an orphan from a previous crash;
//!   move it to `failed/` (we can't safely resume mid-training).
//! - Drain `pending/` in FIFO order (oldest job first).
//! - Watch `pending/` via inotify; wake on new files and drain again.
//! - Reap terminal jobs (completed/failed/cancelled) older than the
//!   configured retention period, on startup and every hour thereafter.
//!
//! Execution is delegated through the `JobExecutor` trait so tests can
//! substitute a synchronous mock without spawning subprocesses.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::channel;
use std::time::Duration;

use notify::{EventKind, RecursiveMode, Watcher};

use crate::queue::{ActiveState, Job, JobId, QueueDir, TerminalState};

/// What happened when the executor ran a job.
pub enum ExecutionResult {
    Success,
    Failure(String),
}

/// Executes a queued job. Implementors decide what "execute" means —
/// production spawns a subprocess; tests use an in-memory mock.
pub trait JobExecutor: Send + Sync {
    fn execute(&self, job: &Job, training_dir: &Path) -> ExecutionResult;
}

/// Production executor: spawn `<binary> run <argv>` with CWD=training_dir,
/// inherit stdout/stderr (so logs go to journalctl via systemd).
pub struct SubprocessExecutor {
    pub binary: PathBuf,
}

impl JobExecutor for SubprocessExecutor {
    fn execute(&self, job: &Job, training_dir: &Path) -> ExecutionResult {
        let argv = job.args.to_argv();
        let outcome = Command::new(&self.binary)
            .arg("run")
            .args(&argv)
            .current_dir(training_dir)
            .status();
        match outcome {
            Ok(status) if status.success() => ExecutionResult::Success,
            Ok(status) => ExecutionResult::Failure(format!("training exited: {status}")),
            Err(err) => ExecutionResult::Failure(format!("spawn failed: {err}")),
        }
    }
}

pub struct Daemon<'a, E: JobExecutor> {
    queue: &'a QueueDir,
    executor: &'a E,
    training_dir: PathBuf,
    retention_max_age: Duration,
    reap_interval: Duration,
}

impl<'a, E: JobExecutor> Daemon<'a, E> {
    pub fn new(
        queue: &'a QueueDir,
        executor: &'a E,
        training_dir: PathBuf,
        retention_max_age: Duration,
        reap_interval: Duration,
    ) -> Self {
        Self { queue, executor, training_dir, retention_max_age, reap_interval }
    }

    /// Sweep `running/` on startup. Anything present is an orphan from a
    /// previous daemon crash — we can't resume mid-training, so it goes to
    /// `failed/`. Returns the count moved.
    pub fn recover_orphans(&self) -> io::Result<usize> {
        let orphans = self.queue.list(ActiveState::Running)?;
        let count = orphans.len();
        for id in orphans {
            self.queue.transition(&id, ActiveState::Running, TerminalState::Failed)?;
        }
        Ok(count)
    }

    /// Process the oldest pending job. Returns the id processed, or `None`
    /// if the queue was empty. Errors that affect a single job (transition
    /// race, missing file) are logged and swallowed; an outer loop should
    /// keep calling `process_one` until it returns `None`.
    pub fn process_one(&self) -> io::Result<Option<JobId>> {
        let pending = self.queue.list(ActiveState::Pending)?;
        let Some(id) = pending.into_iter().next() else {
            return Ok(None);
        };

        let job = match self.queue.read(&id, ActiveState::Pending) {
            Ok(job) => job,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // Cancelled/claimed between list and read — fine, try next time.
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        if let Err(err) = self.queue.transition(&id, ActiveState::Pending, ActiveState::Running) {
            if err.kind() == io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(err);
        }

        let result = self.executor.execute(&job, &self.training_dir);
        let final_state = match result {
            ExecutionResult::Success => TerminalState::Completed,
            ExecutionResult::Failure(message) => {
                eprintln!("Job {} failed: {message}", id.as_str());
                TerminalState::Failed
            }
        };
        self.queue.transition(&id, ActiveState::Running, final_state)?;
        Ok(Some(id))
    }

    /// Repeatedly process jobs until the queue is empty. Returns the count.
    pub fn drain_pending(&self) -> io::Result<usize> {
        let mut count = 0;
        while self.process_one()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Remove terminal jobs older than `max_age`. Thin wrapper around
    /// `queue_ops::reap_expired` so the daemon can call it at the right
    /// points in its lifecycle.
    pub fn reap_expired(&self, max_age: Duration) -> io::Result<usize> {
        crate::queue_ops::reap_expired(self.queue, max_age)
    }

    fn reap_and_log(&self) -> io::Result<()> {
        let reaped = self.reap_expired(self.retention_max_age)?;
        if reaped > 0 {
            eprintln!("Reaped {reaped} expired terminal job(s)");
        }
        Ok(())
    }

    /// Long-running entrypoint: ensure dirs, recover orphans, drain initial
    /// pending, then watch for new submissions and drain on each event.
    /// Blocks forever (until the process is killed).
    pub fn serve_forever(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.queue.ensure_dirs()?;

        let orphan_count = self.recover_orphans()?;
        if orphan_count > 0 {
            eprintln!(
                "Recovered {orphan_count} orphan(s) from running/ → failed/"
            );
        }

        let initial = self.drain_pending()?;
        if initial > 0 {
            eprintln!("Processed {initial} initial pending job(s)");
        }

        self.reap_and_log()?;

        let pending_dir = self.queue.dir_for(ActiveState::Pending);
        let (sender, receiver) = channel::<()>();
        let mut watcher = notify::recommended_watcher(
            move |event: Result<notify::Event, notify::Error>| {
                if let Ok(event) = event
                    && matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_))
                {
                    let _ = sender.send(());
                }
            },
        )?;
        watcher.watch(&pending_dir, RecursiveMode::NonRecursive)?;
        eprintln!("Watching {} for new jobs", pending_dir.display());

        loop {
            // Block until either an inotify event or the reap interval,
            // whichever comes first. This ensures retention runs hourly
            // even when no new jobs are submitted.
            match receiver.recv_timeout(self.reap_interval) {
                Ok(()) => {
                    // Coalesce a burst of events (e.g. tmp-then-rename
                    // emits two) so we drain once, not twice.
                    while receiver.recv_timeout(Duration::from_millis(100)).is_ok() {}
                    self.drain_pending()?;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No new jobs — just run the reaper.
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("inotify watcher disconnected".into());
                }
            }
            self.reap_and_log()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::{ActiveState, TerminalState};
    use crate::queue_ops::submit;
    use crate::run_args::RunArgs;
    use std::path::PathBuf;
    use std::sync::Mutex;

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

    const TEST_RETENTION: Duration = Duration::from_secs(86_400);
    const TEST_REAP_INTERVAL: Duration = Duration::from_secs(3_600);

    fn fresh_queue() -> (tempfile::TempDir, QueueDir) {
        let dir = tempfile::tempdir().unwrap();
        let queue = QueueDir::new(dir.path());
        queue.ensure_dirs().unwrap();
        (dir, queue)
    }

    /// Records every job it sees and replays a fixed result list. If results
    /// run out, defaults to Success.
    struct RecordingExecutor {
        seen: Mutex<Vec<JobId>>,
        results: Mutex<Vec<ExecutionResult>>,
    }

    impl RecordingExecutor {
        fn always_succeed() -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                results: Mutex::new(Vec::new()),
            }
        }

        fn with_results(results: Vec<ExecutionResult>) -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                results: Mutex::new(results),
            }
        }

        fn seen_ids(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|id| id.as_str().to_string())
                .collect()
        }
    }

    impl JobExecutor for RecordingExecutor {
        fn execute(&self, job: &Job, _training_dir: &Path) -> ExecutionResult {
            self.seen.lock().unwrap().push(job.id.clone());
            self.results
                .lock()
                .unwrap()
                .pop()
                .unwrap_or(ExecutionResult::Success)
        }
    }

    #[test]
    fn recover_orphans_moves_running_to_failed() {
        let (_dir, queue) = fresh_queue();
        // Submit and force into running, simulating prior daemon crash
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        queue.transition(&id, ActiveState::Pending, ActiveState::Running).unwrap();

        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);
        let recovered = daemon.recover_orphans().unwrap();

        assert_eq!(recovered, 1);
        assert_eq!(queue.find(&id).unwrap(), Some(TerminalState::Failed.into()));
        assert!(queue.list(ActiveState::Running).unwrap().is_empty());
    }

    #[test]
    fn process_one_returns_none_on_empty_queue() {
        let (_dir, queue) = fresh_queue();
        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);
        assert!(daemon.process_one().unwrap().is_none());
    }

    #[test]
    fn process_one_transitions_pending_to_completed_on_success() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);

        let processed = daemon.process_one().unwrap().unwrap();

        assert_eq!(processed.as_str(), id.as_str());
        assert_eq!(queue.find(&id).unwrap(), Some(TerminalState::Completed.into()));
        assert_eq!(executor.seen_ids(), vec![id.as_str().to_string()]);
    }

    #[test]
    fn process_one_transitions_pending_to_failed_on_failure() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        let executor = RecordingExecutor::with_results(vec![
            ExecutionResult::Failure("simulated".into()),
        ]);
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);

        daemon.process_one().unwrap().unwrap();

        assert_eq!(queue.find(&id).unwrap(), Some(TerminalState::Failed.into()));
    }

    #[test]
    fn drain_pending_processes_all_in_fifo_order() {
        let (_dir, queue) = fresh_queue();
        let mut submitted_ids = Vec::new();
        for i in 0..3 {
            let mut args = sample_args();
            args.model_name = format!("job-{i}");
            let id = submit(&queue, args, "alice".into()).unwrap();
            submitted_ids.push(id.as_str().to_string());
            // Tiny sleep so JobId timestamps differ
            std::thread::sleep(Duration::from_millis(2));
        }
        submitted_ids.sort();

        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);
        let processed = daemon.drain_pending().unwrap();

        assert_eq!(processed, 3);
        assert_eq!(executor.seen_ids(), submitted_ids);
        assert_eq!(queue.list(TerminalState::Completed).unwrap().len(), 3);
        assert!(queue.list(ActiveState::Pending).unwrap().is_empty());
    }

    #[test]
    fn drain_pending_handles_mixed_outcomes() {
        let (_dir, queue) = fresh_queue();
        for i in 0..3 {
            let mut args = sample_args();
            args.model_name = format!("job-{i}");
            submit(&queue, args, "alice".into()).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }

        // Results popped from the back: position 2 → first job, etc.
        let executor = RecordingExecutor::with_results(vec![
            ExecutionResult::Failure("third".into()),
            ExecutionResult::Success,
            ExecutionResult::Failure("first".into()),
        ]);
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);
        daemon.drain_pending().unwrap();

        assert_eq!(queue.list(TerminalState::Completed).unwrap().len(), 1);
        assert_eq!(queue.list(TerminalState::Failed).unwrap().len(), 2);
    }

    #[test]
    fn process_one_skips_when_pending_file_vanishes() {
        // Race: list shows the file, but it gets cancelled before transition.
        // Implemented by submitting then cancelling before process_one.
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        crate::queue_ops::cancel_pending(&queue, &id).unwrap();

        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);

        // No pending now, so process_one returns None
        assert!(daemon.process_one().unwrap().is_none());
        assert!(executor.seen_ids().is_empty());
    }

    #[test]
    fn reap_expired_removes_old_terminal_jobs_after_drain() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let (_dir, queue) = fresh_queue();
        let max_age = Duration::from_secs(86_400);

        // Craft an old completed job (2 days ago)
        let old_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 2 * 86_400;
        let suffix: u32 = rand::Rng::random_range(&mut rand::rng(), 0..=0xFF_FFFF);
        let raw_id = format!("{old_secs:010}_{suffix:06x}");
        let old_id = crate::queue::JobId::parse(&raw_id).unwrap();
        let old_job = crate::queue::Job {
            id: old_id.clone(),
            submitted_at_unix: old_secs,
            submitted_by: "test".into(),
            args: sample_args(),
        };
        queue.write(&old_job, TerminalState::Completed).unwrap();

        // Submit a fresh job that will complete normally
        let fresh_id = submit(&queue, sample_args(), "alice".into()).unwrap();
        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"), TEST_RETENTION, TEST_REAP_INTERVAL);

        daemon.drain_pending().unwrap();
        let reaped = daemon.reap_expired(max_age).unwrap();

        assert_eq!(reaped, 1);
        assert_eq!(queue.find(&old_id).unwrap(), None);
        // Fresh completed job is still there
        assert_eq!(queue.find(&fresh_id).unwrap(), Some(TerminalState::Completed.into()));
    }
}
