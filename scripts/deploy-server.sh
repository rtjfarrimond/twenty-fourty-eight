#!/usr/bin/env bash
set -euo pipefail

# Deploy the server stack: web server, frontend, config, models, skills.
#
# Does NOT touch the training queue daemon, so in-progress training jobs
# continue uninterrupted. Use deploy-training.sh to deploy the training
# stack, or deploy.sh to deploy everything.
#
# Usage: ./scripts/deploy-server.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="2048-server"
SERVICE_USER="2048-solver"

echo "=== Deploying Server Stack ==="
echo

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "ERROR: system user '$SERVICE_USER' does not exist." >&2
    echo "Run 'sudo ./scripts/bootstrap-user.sh' first (one-time setup)." >&2
    exit 1
fi

# Build server (pulls in queue crate as a cargo dependency) and frontend
echo "--- Building server ---"
cd "$PROJECT_ROOT/server"
cargo build --release

echo "--- Building WASM frontend ---"
cd "$PROJECT_ROOT/frontend"
wasm-pack build --target web --out-dir dist/pkg

# Stop server
if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "--- Stopping $SERVICE_NAME ---"
    sudo systemctl stop "$SERVICE_NAME"
fi

# Create directories
echo "--- Creating directories ---"
sudo mkdir -p /opt/2048-solver/bin
sudo mkdir -p /opt/2048-solver/frontend
sudo mkdir -p /etc/2048-solver
sudo mkdir -p /var/lib/2048-solver/models
sudo chown -R "$SERVICE_USER:$SERVICE_USER" /var/lib/2048-solver
sudo chmod -R g+w /var/lib/2048-solver
sudo find /var/lib/2048-solver -type d -exec chmod g+s {} \;

# Install server binary
echo "--- Installing server binary ---"
sudo cp "$PROJECT_ROOT/server/target/release/server" /opt/2048-solver/bin/

# Install frontend
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

# Install Claude Code skills
echo "--- Installing Claude Code skills ---"
SKILLS_DIR="$HOME/.claude/skills"
for skill_dir in "$PROJECT_ROOT"/skills/*/; do
    [ -f "$skill_dir/SKILL.md" ] || continue
    skill_name="$(basename "$skill_dir")"
    mkdir -p "$SKILLS_DIR/$skill_name"
    cp "$skill_dir/SKILL.md" "$SKILLS_DIR/$skill_name/SKILL.md"
    echo "  Installed skill: $skill_name"
done

# Install model metadata
echo "--- Installing model metadata ---"
for meta in "$PROJECT_ROOT"/config/models/*.meta.toml; do
    [ -f "$meta" ] && sudo cp "$meta" /var/lib/2048-solver/models/
done

# Install trained models
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

sudo rm -f /opt/2048-solver/frontend/models.json

# Generate models.json (uses the already-installed generate_models binary)
echo "--- Generating models.json ---"
cd /var/lib/2048-solver/models
/opt/2048-solver/bin/generate_models
cd "$PROJECT_ROOT"

# Benchmarks
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

# Install systemd service
echo "--- Installing systemd service ---"
sudo cp "$PROJECT_ROOT/scripts/2048-server.service" \
    /etc/systemd/system/"$SERVICE_NAME".service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

# Restart
echo "--- Restarting $SERVICE_NAME ---"
sudo systemctl restart "$SERVICE_NAME"
sleep 1
if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "  $SERVICE_NAME is running"
else
    echo "  WARNING: $SERVICE_NAME failed to start"
    sudo journalctl -u "$SERVICE_NAME" --no-pager -n 20
    exit 1
fi

echo
echo "=== Server deploy complete ==="
