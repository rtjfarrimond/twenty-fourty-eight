//! Long-running daemon that drains the job queue.
//!
//! - On startup: any `running/` file is an orphan from a previous crash;
//!   move it to `failed/` (we can't safely resume mid-training).
//! - Drain `pending/` in FIFO order (oldest job first).
//! - Watch `pending/` via inotify; wake on new files and drain again.
//!
//! Execution is delegated through the `JobExecutor` trait so tests can
//! substitute a synchronous mock without spawning subprocesses.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::channel;
use std::time::Duration;

use notify::{EventKind, RecursiveMode, Watcher};

use crate::queue::{Job, JobId, JobState, QueueDir};

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
}

impl<'a, E: JobExecutor> Daemon<'a, E> {
    pub fn new(queue: &'a QueueDir, executor: &'a E, training_dir: PathBuf) -> Self {
        Self { queue, executor, training_dir }
    }

    /// Sweep `running/` on startup. Anything present is an orphan from a
    /// previous daemon crash — we can't resume mid-training, so it goes to
    /// `failed/`. Returns the count moved.
    pub fn recover_orphans(&self) -> io::Result<usize> {
        let orphans = self.queue.list(JobState::Running)?;
        let count = orphans.len();
        for id in orphans {
            self.queue.transition(&id, JobState::Running, JobState::Failed)?;
        }
        Ok(count)
    }

    /// Process the oldest pending job. Returns the id processed, or `None`
    /// if the queue was empty. Errors that affect a single job (transition
    /// race, missing file) are logged and swallowed; an outer loop should
    /// keep calling `process_one` until it returns `None`.
    pub fn process_one(&self) -> io::Result<Option<JobId>> {
        let pending = self.queue.list(JobState::Pending)?;
        let Some(id) = pending.into_iter().next() else {
            return Ok(None);
        };

        let job = match self.queue.read(&id, JobState::Pending) {
            Ok(job) => job,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // Cancelled/claimed between list and read — fine, try next time.
                return Ok(None);
            }
            Err(err) => return Err(err),
        };

        if let Err(err) = self.queue.transition(&id, JobState::Pending, JobState::Running) {
            if err.kind() == io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(err);
        }

        let result = self.executor.execute(&job, &self.training_dir);
        let final_state = match result {
            ExecutionResult::Success => JobState::Completed,
            ExecutionResult::Failure(message) => {
                eprintln!("Job {} failed: {message}", id.as_str());
                JobState::Failed
            }
        };
        self.queue.transition(&id, JobState::Running, final_state)?;
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

        let pending_dir = self.queue.dir_for(JobState::Pending);
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
            // Block until the next inotify event.
            receiver.recv()?;
            // Coalesce a burst of events (e.g. tmp-then-rename emits two)
            // so we drain once per logical submission, not twice.
            while receiver.recv_timeout(Duration::from_millis(100)).is_ok() {}
            self.drain_pending()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        }
    }

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
        queue.transition(&id, JobState::Pending, JobState::Running).unwrap();

        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));
        let recovered = daemon.recover_orphans().unwrap();

        assert_eq!(recovered, 1);
        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Failed));
        assert!(queue.list(JobState::Running).unwrap().is_empty());
    }

    #[test]
    fn process_one_returns_none_on_empty_queue() {
        let (_dir, queue) = fresh_queue();
        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));
        assert!(daemon.process_one().unwrap().is_none());
    }

    #[test]
    fn process_one_transitions_pending_to_completed_on_success() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));

        let processed = daemon.process_one().unwrap().unwrap();

        assert_eq!(processed.as_str(), id.as_str());
        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Completed));
        assert_eq!(executor.seen_ids(), vec![id.as_str().to_string()]);
    }

    #[test]
    fn process_one_transitions_pending_to_failed_on_failure() {
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        let executor = RecordingExecutor::with_results(vec![
            ExecutionResult::Failure("simulated".into()),
        ]);
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));

        daemon.process_one().unwrap().unwrap();

        assert_eq!(queue.find(&id).unwrap(), Some(JobState::Failed));
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
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));
        let processed = daemon.drain_pending().unwrap();

        assert_eq!(processed, 3);
        assert_eq!(executor.seen_ids(), submitted_ids);
        assert_eq!(queue.list(JobState::Completed).unwrap().len(), 3);
        assert!(queue.list(JobState::Pending).unwrap().is_empty());
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
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));
        daemon.drain_pending().unwrap();

        assert_eq!(queue.list(JobState::Completed).unwrap().len(), 1);
        assert_eq!(queue.list(JobState::Failed).unwrap().len(), 2);
    }

    #[test]
    fn process_one_skips_when_pending_file_vanishes() {
        // Race: list shows the file, but it gets cancelled before transition.
        // Implemented by submitting then cancelling before process_one.
        let (_dir, queue) = fresh_queue();
        let id = submit(&queue, sample_args(), "alice".into()).unwrap();
        crate::queue_ops::cancel_pending(&queue, &id).unwrap();

        let executor = RecordingExecutor::always_succeed();
        let daemon = Daemon::new(&queue, &executor, PathBuf::from("/tmp"));

        // No pending now, so process_one returns None
        assert!(daemon.process_one().unwrap().is_none());
        assert!(executor.seen_ids().is_empty());
    }
}
