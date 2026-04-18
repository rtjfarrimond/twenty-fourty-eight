#!/usr/bin/env bash
set -euo pipefail

# Deploy the 2048 solver following Linux FHS conventions.
#
# Installs to:
#   /opt/2048-solver/bin/       — server, training, and utility binaries
#   /opt/2048-solver/frontend/  — static web assets (HTML, CSS, WASM)
#   /etc/2048-solver/           — configuration
#   /var/lib/2048-solver/models/   — trained model artefacts (.bin + .meta.toml)
#   /var/lib/2048-solver/training/ — training run logs
#
# Usage: ./scripts/deploy.sh
# Remote: ssh user@host "cd /path/to/repo && git pull && ./scripts/deploy.sh"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="2048-server"
QUEUE_SERVICE_NAME="2048-training-queue"
SERVICE_USER="2048-solver"

echo "=== Deploying 2048 Solver ==="
echo

# Verify the service user exists. The bootstrap step is one-time and out
# of band — fail fast with a useful pointer if it hasn't been done.
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "ERROR: system user '$SERVICE_USER' does not exist." >&2
    echo "Run 'sudo ./scripts/bootstrap-user.sh' first (one-time setup)." >&2
    exit 1
fi

# Build everything
"$PROJECT_ROOT/scripts/build.sh"

# Stop services if running (can't overwrite a running binary)
if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "--- Stopping $SERVICE_NAME ---"
    sudo systemctl stop "$SERVICE_NAME"
fi
if systemctl is-active "$QUEUE_SERVICE_NAME" >/dev/null 2>&1; then
    # Warn if a job is actually mid-execution. The queue daemon's
    # orphan-recovery sweep will mark it 'failed' on next start.
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

# Create directory structure
echo "--- Creating directories ---"
sudo mkdir -p /opt/2048-solver/bin
sudo mkdir -p /opt/2048-solver/frontend
sudo mkdir -p /etc/2048-solver
sudo mkdir -p /var/lib/2048-solver/models
sudo mkdir -p /var/lib/2048-solver/training
# Queue subdirs match the JobState enum in training/src/queue.rs
for state in pending running completed failed cancelled; do
    sudo mkdir -p "/var/lib/2048-solver/queue/$state"
done

# Data dirs are owned by the service user so the systemd-managed server
# and queue daemon can read/write them. Setgid bit on directories means
# files an admin in the service group creates inherit the group, so the
# daemon can read jobs submitted via the CLI.
sudo chown -R "$SERVICE_USER:$SERVICE_USER" /var/lib/2048-solver
sudo chmod -R g+w /var/lib/2048-solver
sudo find /var/lib/2048-solver -type d -exec chmod g+s {} \;

# Install binaries
echo "--- Installing binaries ---"
sudo cp "$PROJECT_ROOT/server/target/release/server" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/training" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/generate_models" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/benchmark" /opt/2048-solver/bin/

# Expose the training CLI on $PATH for admin use (submit/list/cancel jobs).
# /usr/local/bin is the FHS-standard location for locally-installed binaries.
sudo ln -sf /opt/2048-solver/bin/training /usr/local/bin/training

# Install frontend (exclude benchmarks.json — it is generated fresh below
# and published to /opt directly; excluding it here keeps the existing
# live copy in place if benchmarks are skipped or fail)
echo "--- Installing frontend ---"
sudo rsync -a --delete --exclude='benchmarks.json' \
    "$PROJECT_ROOT/frontend/dist/" /opt/2048-solver/frontend/

# Install config (don't overwrite existing — user may have customised)
if [ ! -f /etc/2048-solver/config.toml ]; then
    echo "--- Installing default config ---"
    sudo cp "$PROJECT_ROOT/config/config.toml" /etc/2048-solver/config.toml
else
    echo "--- Config exists, not overwriting ---"
fi

# Install Claude Code skills (each is a <name>/SKILL.md directory)
echo "--- Installing Claude Code skills ---"
SKILLS_DIR="$HOME/.claude/skills"
for skill_dir in "$PROJECT_ROOT"/skills/*/; do
    [ -f "$skill_dir/SKILL.md" ] || continue
    skill_name="$(basename "$skill_dir")"
    mkdir -p "$SKILLS_DIR/$skill_name"
    cp "$skill_dir/SKILL.md" "$SKILLS_DIR/$skill_name/SKILL.md"
    echo "  Installed skill: $skill_name"
done

# Install model metadata (always update — these are checked into the repo)
echo "--- Installing model metadata ---"
for meta in "$PROJECT_ROOT"/config/models/*.meta.toml; do
    [ -f "$meta" ] && sudo cp "$meta" /var/lib/2048-solver/models/
done

# Copy trained models if they exist in the training dir and aren't installed yet
echo "--- Installing trained models ---"
for bin_file in "$PROJECT_ROOT"/training/*.bin; do
    [ -f "$bin_file" ] || continue
    dest="/var/lib/2048-solver/models/$(basename "$bin_file")"
    if [ ! -f "$dest" ]; then
        echo "  Installing $(basename "$bin_file")"
        sudo cp "$bin_file" "$dest"
    else
        echo "  $(basename "$bin_file") already installed"
    fi
done

# Training logs are now deployed alongside models via runner.rs move_to().
# No need to copy from the repo working directory.

# models.json now lives alongside the .bin files in the data dir, served
# by the /models.json axum route. Clean up any stale copy in the static
# frontend dir from the old layout.
sudo rm -f /opt/2048-solver/frontend/models.json

# Generate models.json from installed models (writes to models_dir by default)
echo "--- Generating models.json ---"
cd /var/lib/2048-solver/models
/opt/2048-solver/bin/generate_models
cd "$PROJECT_ROOT"

# Run the benchmark matrix and publish its output to the live server.
# Override via SKIP_BENCHMARKS=1 (e.g. to deploy while training is hogging
# CPU, or to skip a lengthy matrix for a quick config-only deploy).
if [ "${SKIP_BENCHMARKS:-0}" = "1" ]; then
    echo "--- Skipping benchmarks (SKIP_BENCHMARKS=1) ---"
else
    echo "--- Running benchmark matrix ---"
    BENCH_OUTDIR="$PROJECT_ROOT/training/bench-results/deploy-$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    OUTDIR="$BENCH_OUTDIR" SKIP_BUILD=1 "$PROJECT_ROOT/scripts/bench-matrix.sh"

    if [ -f "$BENCH_OUTDIR/benchmarks.json" ]; then
        echo "--- Publishing benchmarks.json ---"
        sudo cp "$BENCH_OUTDIR/benchmarks.json" \
            /opt/2048-solver/frontend/benchmarks.json
    else
        echo "  WARNING: bench-matrix did not produce benchmarks.json; "\
"live dashboard data left unchanged"
    fi
fi

# Install systemd services
echo "--- Installing systemd services ---"
sudo cp "$PROJECT_ROOT/scripts/2048-server.service" /etc/systemd/system/"$SERVICE_NAME".service
sudo cp "$PROJECT_ROOT/scripts/2048-training-queue.service" \
    /etc/systemd/system/"$QUEUE_SERVICE_NAME".service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl enable "$QUEUE_SERVICE_NAME"

# Restart services
restart_and_check() {
    local unit="$1"
    echo "--- Restarting $unit ---"
    sudo systemctl restart "$unit"
    sleep 1
    if systemctl is-active "$unit" >/dev/null 2>&1; then
        echo "  $unit is running"
    else
        echo "  WARNING: $unit failed to start"
        sudo journalctl -u "$unit" --no-pager -n 20
        exit 1
    fi
}
restart_and_check "$SERVICE_NAME"
restart_and_check "$QUEUE_SERVICE_NAME"

echo
echo "=== Deploy complete ==="
