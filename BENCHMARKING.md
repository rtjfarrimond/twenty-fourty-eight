# Benchmarking

The `benchmark` binary in the `training` crate measures training throughput
and verifies convergence under different algorithm/thread configurations. It
is the primary tool behind the SotA dimensions we track
(`DESIGN_DECISIONS.md` §17): training games/sec, moves/sec, wall-clock time
to target, core scaling efficiency, memory footprint.

It is intended for two use cases:

1. **A/B testing** new training algorithms or tuning knobs against a
   baseline — structured JSON output is designed to be ingested by the
   dashboard and diffed across runs.
2. **Regression detection** — CI/manual checkpoints before and after
   performance-sensitive changes to the training loop or network internals.

## Building

Always build release with LTO — debug builds produce misleading throughput
numbers (10–50× slower for CPU-bound Rust):

```
cargo build --release --bin benchmark
```

Binary lands at `training/target/release/benchmark` (or `target/release/`
if run from the repo root with a workspace).

## Running

```
./target/release/benchmark \
  --algorithm serial --threads 1 \
  --games 5000 --warmup-games 500 \
  --patterns 4x6 --learning-rate 0.0025 --seed 42 \
  --label my-run --output-json /tmp/bench.json
```

All flags have defaults — a bare `./benchmark` runs a 10k-game serial
benchmark on the 4x6 pattern set.

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--games` | 10000 | Number of training games in the timed section. |
| `--warmup-games` | 100 | Games played before timing starts, to eliminate cold-cache/allocator effects. Always runs single-threaded regardless of `--algorithm`. |
| `--patterns` | `4x6` | Pattern preset (see below). `4x6` or `8x6`. |
| `--algorithm` | `serial` | Training algorithm (see below). `serial` or `hogwild`. |
| `--threads` | 1 | Worker thread count. Must be 1 for `serial`; any value ≥1 for `hogwild`. |
| `--learning-rate` | 0.0025 | TD(0) step size α. Matches the default in the training binary. |
| `--seed` | 42 | RNG seed. Same seed + algorithm + thread count produces the same trajectory for the serial algorithm; hogwild runs are reproducible across RNG state but race outcomes depend on scheduler non-determinism. |
| `--output-json` | — | When set, writes a pretty-printed JSON result file at this path in addition to stdout. |
| `--label` | — | Optional free-form string identifying the run (persisted in the JSON output). Useful for distinguishing A/B rows on the dashboard. |

## Pattern presets

Patterns are the n-tuple feature shapes — positions on the 4×4 board that
feed into each lookup table. Both presets use 6-tuples (6 positions per
pattern) with full D4 symmetry expansion (×8), giving `N base × 8` total
feature instances per move.

- **`4x6`** — standard 4-pattern preset from Szubert & Jaśkowski. 32 total
  features. Weight table is ~268 MB.
- **`8x6`** — stronger 8-pattern preset (adds four L/rectangle shapes) from
  Wu et al. 2014 multi-stage paper. 64 total features. Weight table is
  ~537 MB. Slower per-game but reaches higher scores.

See `training/src/config.rs` for the exact pattern definitions.

## Algorithms

### `serial`
Single-threaded TD(0) training loop. Each training game evaluates afterstates
via the network, picks the best move, applies a TD(0) update to the chosen
afterstate, and repeats until game over. Reference implementation that
`hogwild` convergence is checked against.

### `hogwild`
Lock-free shared-memory parallel TD(0). `N` worker threads each play
independent games against the same `NTupleNetwork`. Weights are stored as
`AtomicU32` (reinterpreting `f32` bits); updates use relaxed load-modify-store
(not CAS) — concurrent writers may occasionally clobber each other's updates.
TD(0) on sparse weight tables tolerates this, empirically and by theory.

Paper: Niu, Recht, Ré, Wright — *Hogwild!: A Lock-Free Approach to
Parallelizing Stochastic Gradient Descent* (NeurIPS 2011).
<https://arxiv.org/abs/1106.5730>

2048-specific precedent: Wu et al. 2014 multi-stage TD and the
moporgic/TDL2048 implementation both use shared-memory parallel n-tuple TD.

## Output

### stdout (always)
Human-readable block: configuration, wall-clock seconds, games/sec,
moves/sec, avg score per game, avg moves per game, peak RSS.

### JSON (when `--output-json` is set)
Structured result for dashboard/CI consumption:

```json
{
  "label": "serial-baseline",
  "algorithm": "serial",
  "num_threads": 1,
  "patterns": "4x6",
  "num_base_patterns": 4,
  "num_total_patterns": 32,
  "games": 5000,
  "warmup_games": 500,
  "learning_rate": 0.0025,
  "seed": 42,
  "wall_clock_seconds": 1.00,
  "games_per_second": 5019.0,
  "moves_per_second": 1605839.0,
  "total_moves": 1605839,
  "avg_score_per_game": 4182.0,
  "avg_moves_per_game": 320.0,
  "peak_rss_kb": 265216
}
```

All fields are universal across algorithms — new algorithms (future
multi-stage, TC-learning, etc.) will emit the same shape. This is
deliberate per `DESIGN_DECISIONS.md` §16 (shared comparison surface must
stay algorithm-agnostic).

## Metrics reference

| Metric | What it measures | Why it matters |
|---|---|---|
| `games_per_second` | End-to-end games completed per wall-clock second. | Headline throughput number. Compounds iteration speed on experiments. |
| `moves_per_second` | Individual game moves (decisions + updates) per second. | More stable than games/sec across training progress, since avg moves/game grows as the network improves. Direct comparison point against published implementations (e.g. moporgic/TDL2048 reports 102M inference moves/sec). |
| `avg_score_per_game` | Mean score across timed games. | **Convergence parity check.** Hogwild and serial should agree within run-to-run noise. Large deviation signals a correctness bug. |
| `avg_moves_per_game` | Mean moves per game. | Indirect proxy for play quality — longer games generally correlate with higher scores. |
| `wall_clock_seconds` | Time spent in the timed section. | Ground truth for wall-clock-to-N-games calculations. |
| `peak_rss_kb` | Peak resident set size during the run, from `/proc/self/status` (Linux only; `null` elsewhere). | Memory footprint per algorithm/thread combination. Should not scale with thread count — shared weight storage means additional threads add stacks and RNG state only. |

## Caveats when interpreting results

- **Co-tenancy matters.** Running a benchmark on a machine where other
  heavy processes (e.g. a live training job) are consuming cores will
  depress absolute numbers. Relative comparisons at the same moment in
  time are still informative.
- **Thermal throttling** can bias high-thread-count runs on laptops and
  tight-chassis desktops. Watch for sublinear scaling that correlates with
  run duration.
- **Warmup games run single-threaded** regardless of `--algorithm`, so
  short runs with many warmup games under-represent the parallel path.
  For hogwild benchmarks, keep `--warmup-games` small relative to
  `--games` or just accept the small bias.
- **Convergence parity is a smoke test at short run lengths.** A 5000-game
  benchmark shows that training isn't wildly broken; a real parity check
  requires running to the target game count used by production models and
  comparing final eval scores against the baseline.

## Related docs

- `DESIGN_DECISIONS.md` §12 — Performance optimization philosophy.
- `DESIGN_DECISIONS.md` §16 — Results-table schema: keep shared views
  algorithm-agnostic.
- `DESIGN_DECISIONS.md` §17 — Tracked SotA dimensions and metric list.
- `FUTURE.md` — Parallel training (Hogwild) entry with implementation
  sketch and expected speedup.
- `RESEARCH.md` — Pattern definitions, academic references.

## References

- Niu, Recht, Ré, Wright. *Hogwild!: A Lock-Free Approach to Parallelizing
  Stochastic Gradient Descent.* NeurIPS 2011.
  <https://arxiv.org/abs/1106.5730>
- Szubert, Jaśkowski. *Temporal Difference Learning of N-Tuple Networks for
  the Game 2048.* IEEE CIG 2014.
- Wu, Guei, Chen, Lin, Hsiao, Hsueh, Wu. *Multi-Stage Temporal Difference
  Learning for 2048-like Games.* IEEE TCIAIG 2016.
- moporgic/TDL2048 — high-performance open implementation, reports
  ~100M inference moves/sec on consumer hardware.
  <https://github.com/moporgic/TDL2048>
