#!/usr/bin/env bash
#
# Run a matrix of benchmark configurations and collect structured results.
#
# Each row in the matrix produces one JSON file under $OUTDIR. A summary
# table is printed at the end (requires jq; without it, individual JSON
# files are just listed).
#
# Override the matrix via env vars:
#   GAMES=20000              Games per timed run (default 10000)
#   WARMUP=500               Warmup games per run (default 500)
#   PATTERNS=4x6             Pattern preset (default 4x6)
#   SEED=42                  RNG seed (default 42)
#   THREAD_COUNTS="1 2 4 8 12 15"   Hogwild thread counts to test
#   OUTDIR=/path/to/dir      Where to write JSON results (default dated dir
#                            under $PROJECT_ROOT/training/bench-results/)
#   SKIP_BUILD=1             Skip the cargo build step (assumes binary exists)
#
# Exit codes:
#   0  all runs succeeded
#   1  build failed
#   2  at least one benchmark run failed

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARK_BIN="$PROJECT_ROOT/training/target/release/benchmark"

GAMES="${GAMES:-10000}"
WARMUP="${WARMUP:-500}"
PATTERNS="${PATTERNS:-4x6}"
SEED="${SEED:-42}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 12 15}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ -z "${OUTDIR:-}" ]]; then
    TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    OUTDIR="$PROJECT_ROOT/training/bench-results/$TIMESTAMP"
fi

echo "=== Benchmark matrix ==="
echo "  Patterns:     $PATTERNS"
echo "  Games:        $GAMES"
echo "  Warmup:       $WARMUP"
echo "  Seed:         $SEED"
echo "  Thread counts: $THREAD_COUNTS"
echo "  Output dir:   $OUTDIR"
echo

if [[ "$SKIP_BUILD" != "1" ]]; then
    echo "--- Building release binary ---"
    (cd "$PROJECT_ROOT/training" && cargo build --release --bin benchmark)
    echo
fi

if [[ ! -x "$BENCHMARK_BIN" ]]; then
    echo "ERROR: benchmark binary not found at $BENCHMARK_BIN" >&2
    echo "       run with SKIP_BUILD=0 or build manually." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

run_one() {
    local algorithm="$1"
    local threads="$2"
    local label="$3"
    local json_path="$OUTDIR/${label}.json"

    echo "--- $label (algorithm=$algorithm, threads=$threads) ---"
    if ! "$BENCHMARK_BIN" \
            --algorithm "$algorithm" \
            --threads "$threads" \
            --games "$GAMES" \
            --warmup-games "$WARMUP" \
            --patterns "$PATTERNS" \
            --seed "$SEED" \
            --label "$label" \
            --output-json "$json_path"; then
        echo "FAILED: $label" >&2
        return 1
    fi
    echo
}

FAILED=0

# Serial baseline (single-threaded, no parallelism overhead).
run_one "serial" "1" "serial-1" || FAILED=$((FAILED + 1))

# Hogwild across the requested thread counts.
for threads in $THREAD_COUNTS; do
    run_one "hogwild" "$threads" "hogwild-$threads" || FAILED=$((FAILED + 1))
done

echo "=== Summary ==="
echo "  Results written to: $OUTDIR"
echo

if command -v jq >/dev/null 2>&1; then
    # Aggregate all per-run JSONs into a single benchmarks.json manifest
    # that the frontend dashboard can consume.
    GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    jq --arg generated_at "$GENERATED_AT" -s \
        '{generated_at: $generated_at, runs: .}' \
        "$OUTDIR"/*.json > "$OUTDIR/benchmarks.json"
    echo "  Aggregated manifest: $OUTDIR/benchmarks.json"
    echo

    printf "%-14s %-8s %8s %12s %14s %9s %12s\n" \
        "label" "algo" "threads" "games/sec" "moves/sec" "avg_score" "wall(s)"
    printf "%-14s %-8s %8s %12s %14s %9s %12s\n" \
        "--------------" "--------" "--------" "------------" \
        "--------------" "---------" "------------"
    # Collect rows, then sort: algorithm reverse-alpha (so "serial" before
    # "hogwild"), then thread count numeric ascending.
    jq -r '.runs[] | [
        .label,
        .algorithm,
        .num_threads,
        (.games_per_second | floor),
        (.moves_per_second | floor),
        (.avg_score_per_game | floor),
        (.wall_clock_seconds * 100 | floor / 100)
    ] | @tsv' "$OUTDIR/benchmarks.json" \
    | sort -t$'\t' -k2,2r -k3,3n \
    | awk -F'\t' '{printf "%-14s %-8s %8s %12s %14s %9s %12s\n", $1, $2, $3, $4, $5, $6, $7}'
    echo
    echo "  To view in the dashboard:"
    echo "    cp $OUTDIR/benchmarks.json frontend/dist/benchmarks.json"
    echo "    open frontend/dist/dashboard/benchmarks.html"
else
    echo "  (install jq for a formatted summary table and benchmarks.json)"
    ls -1 "$OUTDIR"
fi

echo

if [[ "$FAILED" -gt 0 ]]; then
    echo "$FAILED run(s) failed." >&2
    exit 2
fi

echo "All runs succeeded."
