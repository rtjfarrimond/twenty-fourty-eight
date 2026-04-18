//! Training CLI: dispatches subcommands. The actual work lives in the
//! `runner` and `queue_ops` modules — `main.rs` is intentionally thin so
//! the CLI surface is easy to change without touching execution logic.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use training::config::{validate_algorithm, validate_init_mode};
use training::daemon::{Daemon, SubprocessExecutor};
use training::queue::{JobId, JobState, QueueDir};
use training::queue_ops::{QueueSnapshot, cancel_pending, submit};
use training::run_args::RunArgs;
use training::runner;

const DEFAULT_QUEUE_DIR: &str = "/var/lib/2048-solver/queue";
const DEFAULT_TRAINING_DIR: &str = "/var/lib/2048-solver/training";

#[derive(Parser)]
#[command(name = "training", about = "2048 RL training pipeline")]
struct Cli {
    /// Root directory for the persistent job queue.
    #[arg(long, default_value = DEFAULT_QUEUE_DIR, global = true)]
    queue_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute a training run inline (foreground). Used directly for one-off
    /// runs and indirectly by the daemon when it picks up a queued job.
    Run(RunArgs),

    /// Submit a training run to the persistent queue. Prints the job id and
    /// exits immediately. The daemon will pick it up FIFO.
    Submit(RunArgs),

    /// Inspect or modify the queue.
    #[command(subcommand)]
    Queue(QueueCommand),

    /// Long-running daemon: watches the queue and executes pending jobs
    /// sequentially via inotify. Intended to run under systemd.
    Daemon {
        /// Working directory for spawned training subprocesses (where
        /// per-job log/config sidecars are written).
        #[arg(long, default_value = DEFAULT_TRAINING_DIR)]
        training_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum QueueCommand {
    /// Print all jobs grouped by state (pending, running, completed, ...).
    List,

    /// Cancel a pending job by id. Refuses to cancel running jobs (see
    /// FUTURE.md "Cancel running training job" for that work).
    Cancel { id: String },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(&args),
        Command::Submit(args) => submit_command(&args, &cli.queue_dir),
        Command::Queue(QueueCommand::List) => list_command(&cli.queue_dir),
        Command::Queue(QueueCommand::Cancel { id }) => cancel_command(&id, &cli.queue_dir),
        Command::Daemon { training_dir } => daemon_command(&cli.queue_dir, training_dir),
    }
}

fn run(args: &RunArgs) -> ExitCode {
    validate_algorithm(&args.algorithm, args.threads);
    validate_init_mode(args.optimistic_init, args.random_init_amplitude);
    match runner::execute(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ERROR: {message}");
            ExitCode::FAILURE
        }
    }
}

fn submit_command(args: &RunArgs, queue_dir: &PathBuf) -> ExitCode {
    validate_algorithm(&args.algorithm, args.threads);
    validate_init_mode(args.optimistic_init, args.random_init_amplitude);
    let queue = QueueDir::new(queue_dir);
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".into());
    match submit(&queue, args.clone(), user) {
        Ok(id) => {
            println!("{}", id.as_str());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("ERROR: failed to submit job: {err}");
            ExitCode::FAILURE
        }
    }
}

fn list_command(queue_dir: &PathBuf) -> ExitCode {
    let queue = QueueDir::new(queue_dir);
    let snapshot = match QueueSnapshot::load(&queue) {
        Ok(snap) => snap,
        Err(err) => {
            eprintln!("ERROR: failed to read queue at {}: {err}", queue_dir.display());
            return ExitCode::FAILURE;
        }
    };
    if snapshot.total() == 0 {
        println!("Queue is empty.");
        return ExitCode::SUCCESS;
    }
    print_section("PENDING", &snapshot.pending);
    print_section("RUNNING", &snapshot.running);
    print_section("COMPLETED", &snapshot.completed);
    print_section("FAILED", &snapshot.failed);
    print_section("CANCELLED", &snapshot.cancelled);
    ExitCode::SUCCESS
}

fn print_section(label: &str, jobs: &[training::queue::Job]) {
    if jobs.is_empty() {
        return;
    }
    println!("{label} ({}):", jobs.len());
    for job in jobs {
        let ephemeral_tag = if job.args.ephemeral { " [ephemeral]" } else { "" };
        println!(
            "  {}  by {:>10}  {} games {}/{}t  model={}{}",
            job.id.as_str(),
            job.submitted_by,
            job.args.games,
            job.args.algorithm,
            job.args.threads,
            job.args.model_name,
            ephemeral_tag,
        );
    }
}

fn cancel_command(id_str: &str, queue_dir: &PathBuf) -> ExitCode {
    let queue = QueueDir::new(queue_dir);
    let id = match JobId::parse(id_str) {
        Ok(id) => id,
        Err(err) => {
            eprintln!("ERROR: invalid job id: {err}");
            return ExitCode::FAILURE;
        }
    };
    match cancel_pending(&queue, &id) {
        Ok(()) => {
            println!("Cancelled {}", id.as_str());
            ExitCode::SUCCESS
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            // Help the user understand why
            match queue.find(&id) {
                Ok(Some(state)) if state != JobState::Pending => {
                    eprintln!(
                        "ERROR: cannot cancel — job is in state '{}', only pending jobs are cancellable",
                        state.dir_name()
                    );
                }
                Ok(_) => {
                    eprintln!("ERROR: no such job: {}", id.as_str());
                }
                Err(read_err) => {
                    eprintln!("ERROR: queue read failed: {read_err}");
                }
            }
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("ERROR: cancel failed: {err}");
            ExitCode::FAILURE
        }
    }
}

fn daemon_command(queue_dir: &PathBuf, training_dir: PathBuf) -> ExitCode {
    let queue = QueueDir::new(queue_dir);
    let binary = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("ERROR: failed to locate own executable: {err}");
            return ExitCode::FAILURE;
        }
    };
    if !training_dir.is_dir() {
        eprintln!(
            "ERROR: training dir does not exist or is not a directory: {}",
            training_dir.display()
        );
        return ExitCode::FAILURE;
    }
    let executor = SubprocessExecutor { binary };
    let daemon = Daemon::new(&queue, &executor, training_dir);
    eprintln!("Starting training queue daemon");
    eprintln!("  Queue dir:    {}", queue_dir.display());
    if let Err(err) = daemon.serve_forever() {
        eprintln!("ERROR: daemon exited: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
