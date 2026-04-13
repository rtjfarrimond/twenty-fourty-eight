use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// SSE stream that sends the training config, then the full log, then tails
/// for new lines as they're appended.
pub async fn training_stream(
    training_dir: PathBuf,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (sender, receiver) = mpsc::channel::<Event>(64);

    tokio::spawn(async move {
        if let Err(err) = run_training_stream(&training_dir, sender).await {
            eprintln!("Training stream error: {err}");
        }
    });

    Sse::new(ReceiverStream { receiver })
}

async fn run_training_stream(
    training_dir: &Path,
    sender: mpsc::Sender<Event>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Find the latest config and log files
    let config_path = find_latest_file(training_dir, "config.json").await;
    let log_path = find_latest_file(training_dir, "log.jsonl").await;

    // Send config as first event
    if let Some(ref path) = config_path {
        if let Ok(contents) = tokio::fs::read_to_string(path).await {
            let event = Event::default().event("config").data(contents);
            if sender.send(event).await.is_err() {
                return Ok(());
            }
        }
    }

    let Some(log_path) = log_path else {
        let event = Event::default().event("no-training").data("{}");
        let _ = sender.send(event).await;
        return Ok(());
    };

    // Send all existing log lines
    let contents = tokio::fs::read_to_string(&log_path).await?;
    let mut lines_sent = 0;
    for line in contents.lines() {
        if line.is_empty() {
            continue;
        }
        let event = Event::default().event("eval").data(line.to_string());
        if sender.send(event).await.is_err() {
            return Ok(());
        }
        lines_sent += 1;
    }

    // Now watch for changes and tail new lines
    let (notify_sender, mut notify_receiver) = mpsc::channel::<()>(16);
    let watch_dir = training_dir.to_path_buf();

    let _watcher_thread = std::thread::spawn(move || {
        let notify_sender = notify_sender;
        let mut watcher = notify::recommended_watcher(
            move |result: Result<notify::Event, notify::Error>| {
                if let Ok(event) = result {
                    if matches!(event.kind, EventKind::Modify(_)) {
                        let _ = notify_sender.blocking_send(());
                    }
                }
            },
        )
        .expect("Failed to create watcher");

        watcher
            .watch(&watch_dir, RecursiveMode::NonRecursive)
            .expect("Failed to watch training dir");

        std::thread::park();
    });

    loop {
        // Wait for a file change notification
        if notify_receiver.recv().await.is_none() {
            break;
        }

        // Drain any queued notifications to avoid redundant reads
        while notify_receiver.try_recv().is_ok() {}

        // Re-read the file and send any new lines
        if let Ok(contents) = tokio::fs::read_to_string(&log_path).await {
            let all_lines: Vec<&str> = contents
                .lines()
                .filter(|line| !line.is_empty())
                .collect();

            for line in all_lines.iter().skip(lines_sent) {
                let event = Event::default().event("eval").data(line.to_string());
                if sender.send(event).await.is_err() {
                    return Ok(());
                }
            }
            lines_sent = all_lines.len();
        }
    }

    Ok(())
}

async fn find_latest_file(directory: &Path, suffix: &str) -> Option<PathBuf> {
    let mut entries = tokio::fs::read_dir(directory).await.ok()?;
    let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;

    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(suffix) {
            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    if latest.as_ref().map_or(true, |(_, prev)| modified > *prev) {
                        latest = Some((entry.path(), modified));
                    }
                }
            }
        }
    }

    latest.map(|(path, _)| path)
}

/// Adapter from tokio mpsc::Receiver to futures::Stream.
struct ReceiverStream {
    receiver: mpsc::Receiver<Event>,
}

impl Stream for ReceiverStream {
    type Item = Result<Event, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx).map(|opt| opt.map(Ok))
    }
}
