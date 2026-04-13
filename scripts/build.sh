#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building 2048 Solver ==="
echo

echo "--- Building game engine ---"
cd "$PROJECT_ROOT/engine"
cargo build --release

echo "--- Building model crate ---"
cd "$PROJECT_ROOT/model"
cargo build --release

echo "--- Building training binaries ---"
cd "$PROJECT_ROOT/training"
cargo build --release

echo "--- Building server ---"
cd "$PROJECT_ROOT/server"
cargo build --release

echo "--- Building WASM frontend ---"
cd "$PROJECT_ROOT/frontend"
wasm-pack build --target web --out-dir dist/pkg

echo "--- Generating models.json ---"
cd "$PROJECT_ROOT/training"
if ls *.bin 1>/dev/null 2>&1; then
    ./target/release/generate_models
else
    echo "  No trained models found, skipping"
fi

echo
echo "=== Build complete ==="
