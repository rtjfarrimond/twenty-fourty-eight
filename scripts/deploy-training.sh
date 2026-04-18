#!/usr/bin/env bash
set -euo pipefail

# Deploy the training stack: queue daemon, training CLI, and utility binaries.
#
# Stops the training queue daemon, so any in-progress job will be marked
# failed on restart. Use deploy-server.sh to deploy the rest of the stack
# without interrupting training.
#
# Usage: ./scripts/deploy-training.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_SERVICE_NAME="2048-training-queue"
SERVICE_USER="2048-solver"

echo "=== Deploying Training Stack ==="
echo

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "ERROR: system user '$SERVICE_USER' does not exist." >&2
    echo "Run 'sudo ./scripts/bootstrap-user.sh' first (one-time setup)." >&2
    exit 1
fi

# Build training (pulls in queue crate as a cargo dependency)
echo "--- Building training binaries ---"
cd "$PROJECT_ROOT/training"
cargo build --release

# Stop queue daemon
if systemctl is-active "$QUEUE_SERVICE_NAME" >/dev/null 2>&1; then
    QUEUE_DIR=/var/lib/2048-solver/queue
    if [ -d "$QUEUE_DIR/running" ] && \
       [ -n "$(ls -A "$QUEUE_DIR/running" 2>/dev/null)" ]; then
        echo "  WARNING: a training job is currently running and will be"
        echo "           marked failed on next daemon start. Re-submit if"
        echo "           you want to retry."
    fi
    echo "--- Stopping $QUEUE_SERVICE_NAME ---"
    sudo systemctl stop "$QUEUE_SERVICE_NAME"
fi

# Create directories
echo "--- Creating directories ---"
sudo mkdir -p /opt/2048-solver/bin
sudo mkdir -p /var/lib/2048-solver/training
for state in pending running completed failed cancelled; do
    sudo mkdir -p "/var/lib/2048-solver/queue/$state"
done
sudo chown -R "$SERVICE_USER:$SERVICE_USER" /var/lib/2048-solver
sudo chmod -R g+w /var/lib/2048-solver
sudo find /var/lib/2048-solver -type d -exec chmod g+s {} \;

# Install binaries
echo "--- Installing training binaries ---"
sudo cp "$PROJECT_ROOT/training/target/release/training" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/generate_models" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/benchmark" /opt/2048-solver/bin/
sudo ln -sf /opt/2048-solver/bin/training /usr/local/bin/training

# Install systemd service
echo "--- Installing systemd service ---"
sudo cp "$PROJECT_ROOT/scripts/2048-training-queue.service" \
    /etc/systemd/system/"$QUEUE_SERVICE_NAME".service
sudo systemctl daemon-reload
sudo systemctl enable "$QUEUE_SERVICE_NAME"

# Restart
echo "--- Restarting $QUEUE_SERVICE_NAME ---"
sudo systemctl restart "$QUEUE_SERVICE_NAME"
sleep 1
if systemctl is-active "$QUEUE_SERVICE_NAME" >/dev/null 2>&1; then
    echo "  $QUEUE_SERVICE_NAME is running"
else
    echo "  WARNING: $QUEUE_SERVICE_NAME failed to start"
    sudo journalctl -u "$QUEUE_SERVICE_NAME" --no-pager -n 20
    exit 1
fi

echo
echo "=== Training deploy complete ==="
