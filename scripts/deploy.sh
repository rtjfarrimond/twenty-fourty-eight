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

echo "=== Deploying 2048 Solver ==="
echo

# Build everything
"$PROJECT_ROOT/scripts/build.sh"

# Stop the service if running (can't overwrite a running binary)
if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "--- Stopping service ---"
    sudo systemctl stop "$SERVICE_NAME"
fi

# Create directory structure
echo "--- Creating directories ---"
sudo mkdir -p /opt/2048-solver/bin
sudo mkdir -p /opt/2048-solver/frontend
sudo mkdir -p /etc/2048-solver
sudo mkdir -p /var/lib/2048-solver/models
sudo mkdir -p /var/lib/2048-solver/training

# Install binaries
echo "--- Installing binaries ---"
sudo cp "$PROJECT_ROOT/server/target/release/server" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/training" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/generate_models" /opt/2048-solver/bin/
sudo cp "$PROJECT_ROOT/training/target/release/benchmark" /opt/2048-solver/bin/

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

# Copy training logs if they exist
for log_file in "$PROJECT_ROOT"/training/*.log.jsonl; do
    [ -f "$log_file" ] || continue
    sudo cp -p "$log_file" /var/lib/2048-solver/training/
done
for config_file in "$PROJECT_ROOT"/training/*.config.json; do
    [ -f "$config_file" ] || continue
    sudo cp -p "$config_file" /var/lib/2048-solver/training/
done

# Generate models.json from installed models
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

# Install systemd service
echo "--- Installing systemd service ---"
sudo cp "$PROJECT_ROOT/scripts/2048-server.service" /etc/systemd/system/"$SERVICE_NAME".service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

# Restart the service
echo "--- Restarting service ---"
sudo systemctl restart "$SERVICE_NAME"
sleep 1

if systemctl is-active "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "  Service is running"
else
    echo "  WARNING: Service failed to start"
    sudo journalctl -u "$SERVICE_NAME" --no-pager -n 20
    exit 1
fi

echo
echo "=== Deploy complete ==="
