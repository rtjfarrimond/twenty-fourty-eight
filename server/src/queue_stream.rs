//! SSE stream of training queue snapshots.
//!
//! On connect: send the full snapshot once. Then watch the queue root
//! recursively via inotify and re-send the full snapshot whenever any
//! state subdirectory changes. Snapshots are tiny (a few jobs at most),
//! so re-sending the whole thing is simpler than computing diffs.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use queue::queue::QueueDir;
use queue::queue_ops::QueueSnapshot;

pub async fn queue_stream(
    queue_dir: PathBuf,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (sender, receiver) = mpsc::channel::<Event>(64);

    tokio::spawn(async move {
        if let Err(err) = run_queue_stream(&queue_dir, sender).await {
            eprintln!("Queue stream error: {err}");
        }
    });

    Sse::new(ReceiverStream { receiver })
}

async fn run_queue_stream(
    queue_dir: &Path,
    sender: mpsc::Sender<Event>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let queue = QueueDir::new(queue_dir);
    // Self-bootstrap so a fresh deploy doesn't leave the watcher swinging
    // at a non-existent path. Cheap, idempotent.
    queue.ensure_dirs()?;

    if !send_snapshot(&queue, &sender).await {
        return Ok(()); // client disconnected
    }

    let (notify_sender, mut notify_receiver) = mpsc::channel::<()>(16);
    let watch_dir = queue_dir.to_path_buf();

    let _watcher_thread = std::thread::spawn(move || {
        let notify_sender = notify_sender;
        let mut watcher = notify::recommended_watcher(
            move |result: Result<notify::Event, notify::Error>| {
                if let Ok(event) = result
                    && matches!(
                        event.kind,
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                    )
                {
                    let _ = notify_sender.blocking_send(());
                }
            },
        )
        .expect("Failed to create queue watcher");

        watcher
            .watch(&watch_dir, RecursiveMode::Recursive)
            .expect("Failed to watch queue dir");

        std::thread::park();
    });

    loop {
        if notify_receiver.recv().await.is_none() {
            break;
        }
        // Coalesce a burst (e.g. tmp-then-rename or simultaneous transitions
        // across multiple subdirs) into one snapshot.
        while notify_receiver.try_recv().is_ok() {}
        if !send_snapshot(&queue, &sender).await {
            return Ok(());
        }
    }

    Ok(())
}

/// Reads the queue, serializes the snapshot, and pushes it as an SSE event.
/// Returns `false` if the receiver dropped (client disconnected).
async fn send_snapshot(queue: &QueueDir, sender: &mpsc::Sender<Event>) -> bool {
    let snapshot = match QueueSnapshot::load(queue) {
        Ok(snap) => snap,
        Err(err) => {
            eprintln!("Failed to load queue snapshot: {err}");
            return true;
        }
    };
    let json = match serde_json::to_string(&snapshot) {
        Ok(json) => json,
        Err(err) => {
            eprintln!("Failed to serialize snapshot: {err}");
            return true;
        }
    };
    let event = Event::default().event("snapshot").data(json);
    sender.send(event).await.is_ok()
}

struct ReceiverStream {
    receiver: mpsc::Receiver<Event>,
}

impl Stream for ReceiverStream {
    type Item = Result<Event, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx).map(|opt| opt.map(Ok))
    }
}
