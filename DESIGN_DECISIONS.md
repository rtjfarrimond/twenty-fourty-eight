# Design Decisions

Record of architectural decisions made during the design phase.

**Goal:** Ultimately surpass the current state of the art for 2048 RL solvers,
starting by reproducing existing results first.

---

## 1. Game Engine Core Representation

**Decision:** Start with a `u64` bitboard (4 bits per tile, storing exponents).
Migrate to `u128` (5 bits per tile) later when pushing beyond state of the art.

**Rationale:**
- Tiles are powers of 2, so storing the exponent (0–17) is sufficient. Empty = 0.
- 4 bits per tile x 16 tiles = 64 bits. Enables single-register operations,
  trivial hashing, and cache-friendly precomputed row-move lookup tables.
- This caps representable tiles at 2^15 = 32768. The theoretical game maximum
  is 2^17 = 131072 (with 4-spawns), which needs 5 bits per tile (80 bits, fits
  in a `u128`).
- The 32768 cap is fine for reproducing prior work. When surpassing state of
  the art, the migration to `u128` is contained within the core representation
  module — the rest of the codebase depends on the API, not the internal
  encoding.
- The cost of `u128` is slightly slower operations (not single-register on most
  architectures) and larger lookup tables. Worth deferring until needed.

---

## 2. Model Architecture

**Decision:** Start with n-tuple networks and TD-learning to reproduce SOTA.
Explore hybrid/novel architectures in phase 2 to surpass it.

**Rationale:**
- N-tuple networks *are* the current SOTA for 2048 (Szubert & Jaśkowski; Wu et
  al.), not convnets. The README's original framing ("rather than using
  convolutional networks, as prior work does") was inverted — CNNs have been
  tried but haven't matched n-tuple performance.
- N-tuple networks are essentially large lookup tables indexed by patterns of
  tiles on the board. They are fast, simple, and entirely implementable in pure
  Rust with no Python/ML framework dependency.
- The model interface is clean (given a board state, return a value or action),
  so swapping in a different architecture later (MLP, transformer, hybrid) does
  not affect the game engine or training loop structure.
- **Phase 2 angles for surpassing SOTA:**
  - Larger/more n-tuple patterns, systematic pattern search
  - Deeper expectimax lookahead at inference using the learned value function
  - Hybrid architectures (neural net complementing n-tuple tables)
  - Training innovations: multi-stage curriculum, better TD variants, TD +
    Monte Carlo combinations

**NOTE:** The README should be corrected to reflect that prior SOTA work uses
n-tuple networks, not convnets.

---

## 3. RL Training Method

**Decision:** TD(0) with afterstate value functions.

**Rationale:**
- This is the proven pairing with n-tuple networks for 2048 in the literature.
- Afterstates (the board state after the player moves but before a random tile
  spawns) are the natural evaluation point — they remove the stochastic element
  from the value function, making learning more stable.
- More complex variants (TD(λ), multi-step returns) are possible future
  improvements but not needed to reproduce SOTA.

---

## 4. Training Hardware

**Decision:** CPU-only for phase 1. GPU may be needed for phase 2.

**Rationale:**
- N-tuple networks are lookup tables, not matrix multiplications. There is
  nothing to accelerate on a GPU — CPU training is the natural fit, not a
  compromise.
- If phase 2 introduces neural network components (hybrid architectures, MLPs,
  transformers), GPU training may become relevant at that point.

---

## 5. WebSocket Architecture

**Decision:** One shared agent game, server-authoritative. All game state
(agent and user) lives on the server.

**Details:**
- A single agent game runs continuously on the server. All connected visitors
  watch the same game via websocket, receiving state changes as they happen.
- When a user takes over to play manually, they get their own independent game
  with server-side state, identified by session. The agent's game continues in
  the background — it is not paused or interrupted.
- Server-side state for user games was chosen over client-stateful design for
  simplicity of implementation and smaller websocket payloads (send moves, not
  full board states).
- **Agent pacing:** The agent plays one move per configurable duration (e.g.
  one move per second), tuneable to whatever looks right visually. Without
  this, the agent would play at microsecond speed — unwatchable.
- **Session cleanup:** Each session has a last-activity timestamp. A background
  task sweeps periodically and drops sessions idle beyond a threshold (e.g. 5
  minutes). Websocket disconnection triggers immediate cleanup.
- **Load characteristics:** The server is extremely lightweight — no database,
  no disk I/O, no external API calls in the hot path. All in-memory compute
  completing in microseconds.
  - Agent watchers: just receiving broadcasts (~1 move/sec, few bytes). 10k+
    concurrent watchers is feasible.
  - Active players: each move is a single game engine call (microseconds).
    Thousands of moves per second on modest hardware.
  - Bottleneck under extreme load is websocket connection count (OS file
    descriptors, memory per connection), not CPU.
  - A max concurrent connections cap should be set so the server degrades
    gracefully rather than OOMing. No need for connection pooling, queuing, or
    horizontal scaling. Single async runtime (tokio) on a modest VPS is
    sufficient.

---

## 6. Model Inference & Artefact Packaging

**Decision:** Training produces a serialized artefact file. The server loads it
via a shared model-format crate, behind an abstract Agent trait.

**Details:**
- The trained model (weight tables + tuple pattern definitions for n-tuple
  networks) is serialized to a binary artefact file (e.g. `model.bin`).
- **Three-way separation:**
  - **Training crate** — produces the artefact. Depends on the model format
    crate. Not depended on by the server.
  - **Model format crate** — defines the serialization format, a read-only
    inference struct, and an `Agent` trait (e.g. `best_move(state) -> Move`,
    `evaluate(state) -> f64`). This is the only coupling point.
  - **Server crate** — loads the artefact at startup, uses the `Agent` trait
    for inference. Zero dependency on training code.
- The `Agent` trait is the stable interface. When phase 2 introduces new model
  architectures, they implement the same trait. The server doesn't change —
  just load a different artefact with a different `Agent` implementation.

---

## 7. WASM Frontend

**Decision:** The WASM frontend is purely a renderer. No inference runs
client-side.

**Rationale:**
- The agent game runs server-side and streams state over websocket. The
  frontend just receives board states and renders them.
- For user-initiated games, the frontend renders the board and sends user
  inputs to the server, which processes moves (using the game engine) and
  returns the new state.
- WASM is used for the rendering/UI layer (smooth animations, responsive
  interaction), not for computation. This keeps the client lightweight and
  avoids shipping model artefacts to the browser.

---

## 8. Model Evaluation During Training

**Decision:** Evaluate periodically using standard SOTA metrics. Checkpoint
artefacts with a retention policy. Training dashboard is a phase 1 deliverable.

**Details:**
- **Metrics:** Align with prior research (Wu et al.) — average score over N
  games, plus percentage of games reaching each tile milestone (2048, 4096,
  8192, 16384, 32768). This enables direct SOTA comparison.
- **Frequency:** Every 10k–100k training games (tunable). Cheap relative to
  training.
- **Reproducibility:** Fixed random seeds for evaluation game sets so that
  checkpoints can be compared fairly across runs. Seed configuration must be
  documented.
- **Checkpointing:** Save model artefacts at each evaluation point. N-tuple
  weight tables can be large (hundreds of MB to a few GB), so a retention
  policy is needed — keep the best K checkpoints and every Nth, discard the
  rest.
- **Dashboards:** See section 11 for full dashboard architecture.
- **Note:** User's wife (RL expert) joining at phase 2 — evaluation pipeline
  should be solid and well-documented by then.

---

## 9. Implementation Milestones (Phase 1)

**Milestone 1 — Play 2048 in the browser:**
Game engine crate → server with user sessions → WASM frontend. Proves out the
core game logic, websocket layer, and rendering end-to-end. Enables feedback on
game feel before moving to model work.

**Milestone 2 — Dummy model plays on live server:**
Model format crate + Agent trait → trivial agent (random/heuristic) → server
agent game loop with broadcast + configurable pacing. Proves the model-server
interface before real training. "Take over" button working.

**Milestone 3 — Training pipeline + dashboard:**
Research SOTA → n-tuple TD(0) training → evaluation pipeline → training
dashboard → swap dummy agent for trained model. User deploys to web in
parallel.

**Rationale:** End-to-end vertical slices over layer-by-layer. Catches
integration issues early and produces something tangible at each milestone.

---

## 11. Dashboard Architecture

**Decision:** Two distinct dashboard views serving different concerns, unified
by a shared eval data format.

### Model Results Dashboard (`/dashboard`) — user-facing
- Reads a `models.json` manifest listing all models with their eval stats.
- Each model entry has: name, type (heuristic / n-tuple / etc), description,
  and eval results (avg score, max score, tile-reach percentages).
- Heuristic agents (e.g. the dummy agent) appear as models with type
  "heuristic" and a single eval data point — no training curve, just
  performance. No special casing needed.
- Users can select models via dropdown or multi-select to compare side by side.
- Trained models include their full training curve (array of eval checkpoints)
  so the learning progression is visible.
- Static data — the server serves `models.json` and the page renders it.

### Live Training Dashboard (`/dashboard/training`) — internal or user-facing
- Reads the JSONL log from an active training run.
- Shows: model name, training config (total games, learning rate, patterns),
  progress (% complete based on games_trained / total_games).
- Auto-refreshes (tuneable interval, default 1s) until training is complete,
  then stops automatically.
- Training is considered complete when games_trained in the last log entry
  equals the configured total.

### Data format
- Both dashboards consume the same `EvalResult` JSON structure.
- For the model results dashboard, `models.json` wraps eval results with model
  metadata.
- For the live training dashboard, the JSONL file is the raw eval stream. A
  companion `training_config.json` provides the metadata (model name, total
  games, learning rate, etc.).

---

## 12. Performance Optimization

**Status:** Benchmark harness built; initial Hogwild scaling measured;
memory bandwidth identified as the current ceiling on training throughput.

**Approach:** Profile first, then fix. No guessing. The `benchmark` binary
and `BENCHMARKING.md` are the repeatable framework for this.

### Initial scaling measurement (4x6 patterns, AMD EPYC Rome, 16 physical cores)

Single-threaded serial vs. Hogwild at various thread counts, 5000 games:

| Config | Games/sec | Moves/sec | Speedup | Efficiency |
|---|---|---|---|---|
| serial-1 | 5,019 | 1.6M | 1.00× | 100% |
| hogwild-4 | 7,243 | 2.3M | 1.44× | 36% |
| hogwild-8 | 10,979 | 3.6M | 2.19× | 27% |
| hogwild-15 | 13,920 | 4.5M | 2.77× | 18% |

### Bottleneck identified: memory bandwidth, not CPU

`perf stat` confirms a memory-stall-dominated hot path, not a CPU-bound
one:

| Metric | Serial-1 | Hogwild-15 |
|---|---|---|
| IPC (instructions per cycle) | 1.24 | 0.33 |
| CPUs actually utilized | 0.998 | 6.7 / 15 |
| Cache miss rate | 20% | 30% |

IPC dropping from 1.24 to 0.33 is the textbook signature of memory-bound
workloads — cores spend 73% of cycles stalled waiting on DRAM. Root cause:
weight table is 268 MB, L3 cache is only 16 MiB, so every feature lookup
misses L3. At 15 threads the aggregate DRAM bandwidth demand plateaus.

**Implication:** adding more threads beyond ~8 yields diminishing returns
on this machine. The ceiling is hardware, not algorithm. 2.77× is the
realistic current limit without reducing bytes-per-move.

### Where the next wins live (ranked by expected impact)

1. **Reduce bytes per move** — halve memory traffic with f16 weights, or
   pack weights so one cache line serves more lookups. 2-3× potential.
   See `FUTURE.md` entry on memory-layout optimization.
2. **Reduce lookups per move** — algorithmic (e.g. skip evaluation on
   clearly-dominated afterstates). Smaller gain, harder to do cleanly.
3. **Faster-memory hardware** — would improve absolute numbers, won't
   change the shape of the curve.
4. **Prefetching hints** — PEXT-derived indices make software prefetch
   hard; hardware prefetcher already does what it can.
5. **Per-thread batching / delayed merge** — reduces coherence traffic
   but changes algorithm semantics. Probably <20% gain given the primary
   bottleneck is raw bandwidth, not coherence.

### Aspirational target

moporgic/TDL2048 reports ~100M moves/sec *inference* on consumer hardware.
Training moves/sec differs (writes + more computation per move), but the
gap (4.5M → 100M) is large enough that non-trivial optimization work is
required to close it. Not today's problem — logged as future work.

---

## 13. Hot Model Loading

**Status:** Implemented.

**Decision:** The server watches the models directory for new `.bin` files
using inotify (Linux file change notifications) and loads them automatically
without requiring a restart.

**Details:**
- Event-driven, not periodic scanning. Uses the `notify` crate (inotify on
  Linux) to watch the models directory for new files.
- When a new `.bin` appears: waits 1s for the `.meta.toml` sidecar to arrive,
  loads the model, registers it in the `ModelRegistry`, spins up a new agent
  game loop, and makes it available to connected clients on next `ListModels`.
- `ModelRegistry` uses `RwLock` for interior mutability, supporting runtime
  model additions without restarting the server.
- The training binary handles the full artefact lifecycle via `--models-dir`:
  train → save `.bin` to models directory → write `.meta.toml` → copy training
  logs → regenerate `models.json`. No manual file moves or restarts required.

---

## 14. Reproducibility

**Status:** In progress — CLI refactored, model retraining pending.

**Decision:** All existing models will be deleted and retrained from scratch to
ensure full reproducibility with the current codebase.

**Details:**
- Every trained model must be reproducible from a documented command.
- The training CLI now uses clap with named flags (`--games`, `--eval-interval`,
  `--model-name`, `--optimistic-init`, `--learning-rate`, `--models-dir`,
  `--description`). Commands are self-documenting and old experiments can be
  re-run exactly.
- Training parameters (learning rate, optimistic init value, pattern set,
  number of games, eval interval) are all captured in the output artefacts.

---

## 15. Deployment

**Decision:** Follow Linux FHS conventions with system-level installation.

**Layout:**
- `/opt/2048-solver/bin/` — server, training, generate_models, benchmark
- `/opt/2048-solver/frontend/` — static web assets (HTML, CSS, WASM)
- `/etc/2048-solver/config.toml` — server configuration
- `/var/lib/2048-solver/models/` — trained model `.bin` files + `.meta.toml`
- `/var/lib/2048-solver/training/` — per-run training logs

**Rationale:** This is a web server intended to run 24/7, not a desktop app.
System-level installation (`/opt`, `/etc`, `/var`) is conventional for this
use case. Requires `sudo` for deploy, which is acceptable.

The server reads a config file (first CLI arg, or `/etc/2048-solver/config.toml`,
or built-in defaults). Defaults point to relative dev paths so `cargo run` from
the server directory works without any config for local development.

---

## 16. Results Table Schema

**Decision:** Shared results views (table, comparisons) only display metrics
that generalize across agent types. Algorithm-specific training
hyperparameters live in per-model descriptions, not as table columns.

**Rationale:**
- Universal metrics — average score, max score, tile-reach percentages — are
  comparable across any agent, from heuristics to n-tuple TD to expectimax +
  endgame tablebases.
- N-tuple hyperparameters (learning rate, optimistic init, pattern set,
  multi-stage config, TC learning state) do not apply to other algorithms.
  Promoting them to columns would leave empty cells for heuristic and
  search-based agents, and force schema churn every time a new algorithm is
  added.
- Per-model descriptions (or a detail view) are the right place for
  algorithm-specific context — they scale with agent diversity without
  distorting the shared comparison surface.

---

## 17. SotA Dimensions

**Decision:** We pursue SotA on *multiple* independent axes. Training speed
is a first-class goal, not plumbing underneath the model work.

**Axes we care about:**
1. **Solver quality** — avg score, tile-reach rates (2048/4096/8192/16384/
   32768) at a given training budget. The headline metric.
2. **Training throughput** — games/sec, moves/sec during training, wall-clock
   time to reach a target performance bar. Iteration speed compounds: every
   doubling of training throughput doubles the experiments runnable per
   calendar day.
3. **Training sample efficiency** — games needed to reach a given
   performance bar. Orthogonal to throughput (algorithm vs. infrastructure).
   Improvements here (TC learning, multi-stage, better TD variants) reduce
   training cost more fundamentally than parallelism does.
4. **Inference throughput** — moves/sec during play. Reference:
   moporgic/TDL2048 reports 102M moves/sec on Ryzen 9.
5. **Memory footprint** — weight table size for a given pattern complexity.
   Affects deployment, portability, and cache behavior at inference time.

**Metrics we track:**
- Training: games/sec, moves/sec, wall-clock hours to N million games, core
  scaling efficiency (speedup ÷ cores used).
- Sample efficiency: games required to hit avg-score or tile-reach
  thresholds, across algorithm variants on fixed seeds.
- Inference: moves/sec single- and multi-threaded, p99 move latency.
- Quality: avg score over 1000 eval games (fixed seed), tile-reach %.
- Memory: RSS peak during training, artefact size on disk.

**Rationale:** A system that reaches SotA quality in 10× the wall-clock time
is not SotA — it is reproducing published results slowly. "Fastest-to-SotA"
is a real dimension of the problem. Treating parallelism as a boring
substrate hides the fact that training throughput gates every experiment we
can run; it deserves the same design rigor as model architecture.

---

## 18. Optimistic Weight Initialization

**Decision:** Use optimistic init for n-tuple TD with a small K — empirically
~0.25× the converged-per-pattern weight magnitude. The classical "K must
exceed E[return]" rule for tabular bandits does not transfer to n-tuple TD;
K should sit just above the *early* per-weight magnitude, not the converged
one.

**Empirical K-curve (4x6, 100K games, hogwild 14 threads, lr=0.0025):**

| K       | V_init    | Avg score | 2048%  | Notes               |
|---------|-----------|-----------|--------|---------------------|
| 0       | 0         | 12,725    | 4.0    | zero-init baseline  |
| 20      | 640       | 16,054    | 16.8   |                     |
| 50      | 1.6k      | 16,311    | 18.6   |                     |
| 100     | 3.2k      | 16,594    | 19.8   | **peak**            |
| 250     | 8k        | 15,921    | 16.3   |                     |
| 500     | 16k       | 15,300    | 13.9   |                     |
| 1000    | 32k       | 15,149    | 11.9   |                     |
| 2000    | 64k       | 14,653    | 10.5   |                     |
| 5000    | 160k      | 14,191    | 9.8    | shallow tail        |
| 380,000 | 12M       | 7,613     | 0.0    | catastrophic        |

K=100 wins by ~+30% on avg score and ~5× on 2048-rate vs zero init at the
same game count. Sample efficiency: **K=100 at 10K games matches zero-init
at 30K games** — ~3× early speedup.

**Curve shape is asymmetric:** steep ramp from K=0 (single biggest jump on
the curve is K=0→K=20: +26%), then a broad plateau K=20–250 within ~5% of
peak, then a gentle decline an order of magnitude past peak, then a cliff
somewhere between K=5000 and K=380000. The asymmetry matters for tuning:
overshooting K modestly is mostly free, undershooting near zero is not.

**Tuning rule:** for an n-tuple network with `P` total patterns (base × 8
symmetries) trained to expected converged avg score `S`:

```
K_optimal ≈ (S / P) × 0.25
```

(Calibrated from 4x6/100K: S=12.7k, P=32 → S/P=397 → K=100 = 0.25 × 397.)

**Why a small K — and not a big one as bandit literature suggests:**
- In tabular bandits, each arm has its own Q; K must exceed E[return] so
  unpulled arms stay attractive after one pull updates a competing Q
  downward.
- In n-tuple TD, V(board) = sum of `P` weight lookups. Even a small K sets
  every unvisited weight slightly above zero, which shifts unvisited
  *features* (not states) upward. Since most features are touched many
  times, the early per-weight magnitude is small — `O(lr × δ)` per touch,
  which is on the order of single digits in early games. K just needs to
  exceed *that* (a few units), not the eventual converged magnitude
  (hundreds to thousands).
- Going far above this baseline doesn't increase exploration further; it
  just slows the weights' descent toward truth.

**Worked examples (using S/P × 0.25):**

| Setup           | P  | Expected S | K_optimal |
|-----------------|----|------------|-----------|
| 4x6 / 100K      | 32 | ~13k       | ~100      |
| 4x6 / 1M        | 32 | ~36k       | ~280      |
| 4x6 / 10M       | 32 | ~110k      | ~860      |
| 8x6 / 100K      | 64 | ~14k       | ~55       |
| 8x6 / 1M        | 64 | ~41k       | ~160      |
| 8x6 / 10M       | 64 | ~125k      | ~490      |

Note: the K-curve is a broad plateau spanning roughly 0.2× to 2.5× of
K_optimal (within ~5% of peak), so the rule doesn't need to be precise.
Going an order of magnitude high costs ~10–15%; two orders catastrophic.
Going to K=0 costs ~30%.

**Failure mode (previously documented):** picking K orders of magnitude
above truth (we tried K=380,000 for 4x6 in early experiments) is not "more
optimism" — it cripples learning. With per-update step ≈ `lr × δ` where δ
scales with game returns (thousands), the time to drag weights from K down
toward truth scales as `K / (lr × S)` per weight; with K=380k, lr=0.0025,
S=110k it takes ~1400 updates per weight. Spread over 67M weight slots
touched ~once per 1000 games, that's many millions of games before weights
reach the right magnitude — even 10M games never recovers, and the run
underperforms zero-init by 5-10×.

**Why it works on n-tuple TD when random init does not:** optimistic init
shifts the *mean* of V (drives exploration of unvisited states); random
init shifts only the *variance* around zero (no exploration pressure,
nothing to break since each weight is already a unique
`(pattern, board_index)` lookup).

**How to apply:** pick K from the tuning rule above based on (a) total
pattern count and (b) the expected converged score for your target game
budget. If unsure, the K-curve is a plateau — anything within 0.5× to 5×
of the optimum is fine, and only orders-of-magnitude misses are
catastrophic. When training to a longer horizon than tested, scale K
upward proportionally to the expected converged score.

---

**Remaining polish items:**
- Tile sliding animations (requires client-side state diffing)
- Replace unicode arrow characters with SVG/icon font for consistent
  cross-device rendering (current padding hack is browser-specific)

---

## 19. Hogwild Micro-Optimization Attempts (Dead Ends)

An optimization session specifically targeted at extracting more Hogwild
throughput on the current hardware (AMD EPYC Rome / Zen 2 KVM guest, 16
vCPUs, 16 MiB L3, THP in madvise mode). The memory-bandwidth ceiling
identified in §12 turns out to be genuinely binding — every approach
below either traded for nothing or net-regressed, with the one caveat
that run-to-run variance on this shared VM is ~10–15%.

### Attempts and outcomes

| Attempt | Outcome | Why it failed |
|---|---|---|
| `panic="abort"` + `lto="fat"` | -7% hogwild-15, -14% serial | Aggressive inlining appears to hurt register allocation on this hot loop. The `lto=true` (thin) + default unwinding combination the repo already had is the faster spot on this codebase. |
| Explicit `_mm_prefetch(T0)` of all 8N offsets per eval | -11% serial, +3.5% hogwild-15 (within noise) | Forces a stack-array to hold offsets between prefetch-issue and read phases; the extra spills outweigh the DRAM latency hiding. Out-of-order execution already batches ~10 outstanding misses implicitly. |
| Force software PEXT fallback (nibble-by-nibble extraction loop) | 4.5× slower serial, 2× slower hogwild-15 | Despite Zen 2 microcoding PEXT at ~18 cycles, the `trailing_zeros` loop in the software version is worse. Hardware PEXT wins cleanly. |
| Single-orientation eval × 8 (algebraic shortcut assuming 8-orbit weight symmetry) | Convergence broken (late-game avg 2536 vs early 2861, i.e. negative learning) | **Math was wrong.** Sibling weights in a board's 8-orbit are NOT actually equal: the σ_k permutation on pattern-p index space depends on the full board, not just its pattern-p index. Two boards sharing a pattern-p index have different "orbit friends" for other patterns, so weights diverge with training. Verified empirically: 173% relative error after 200 games of training. |
| Thread pinning workers to distinct cores via `sched_setaffinity` | -7% hogwild-15, -8% hogwild-8 | KVM vCPU-to-physical-core mapping is opaque, so guest-side pinning restricts the scheduler without actually pinning physical cores. Default Linux scheduler already spreads load well on this environment. |
| Caching eval offsets in `MoveCandidate` and skipping 8N PEXTs in update | -12% to -27% serial/hogwild | The 256-byte `[u32; 64]` offset cache bloats MoveCandidate past the size where the compiler keeps it in registers. Every `current = next` becomes a large memory copy. Tried external ping-pong buffers; same problem — the scratch handoff costs more than the PEXTs saved. |
| THP madvise(`MADV_HUGEPAGE`) on the weight table | No measurable effect | THP is enabled in `madvise` mode on this kernel, but neither a Rust program nor a minimal C test produced any `AnonHugePages` after madvise + touch. Likely the KVM host refuses huge page backing. The madvise call is harmless (no-op when THP isn't available) so the hint was kept for portability. |

### What was actually kept

- `hint_huge_pages` madvise call (harmless on this VM, real win on hosts
  where THP works).
- `const TABLE_SIZE = 16^6` replacing the per-struct `table_size` field
  (tiny codegen win; mostly a tidiness change since the compiler could
  already constant-propagate the field).
- `libc` as an explicit dependency (supports the madvise hint).

No speedup beyond run-to-run noise was realized on this hardware. The
ceiling identified in §12 holds: with the weight table 16× larger than
L3, hogwild-15 is genuinely bandwidth-bound, and the only remaining
levers are (a) reducing bytes-per-lookup (f16/bfloat16 — see below), or
(b) running on a machine with a bigger L3 or HBM bandwidth.

### Why bfloat16 / f16 weights were not attempted

bfloat16 has a 7-bit mantissa. At weight magnitudes around 10²–10³ (the
typical converged scale) the representable resolution is ≈ weight·2⁻⁷ =
~8. TD(0) updates are α·δ ≈ 10⁻³ — three orders of magnitude below bf16
resolution, so updates round away to zero and learning stops. Standard
f16 has 10-bit mantissa but an 8-bit exponent too narrow for the full
weight range during optimistic-init runs.

Making half-precision work would require either stochastic rounding
(which preserves expected value under truncation but adds per-update
RNG cost) or INT16 fixed-point with a carefully chosen scale factor.
Both are real refactors that change the weight file format and require
fresh convergence validation — left for a future session.

### Aspirational gap

moporgic/TDL2048 reports ~11.5M moves/sec single-threaded for training
with 4-tuple patterns. Our serial-1 with 6-tuple patterns is ~1.7M.
The gap is primarily tuple-size × architecture: their 4-tuple tables
(~4 MiB each) fit comfortably in L2, ours don't fit in L3. Bridging
requires either switching to 4-tuple models (different convergence
characteristics — would forfeit the score plateau we've reached with
6-tuples) or quantized weights. Logged in FUTURE.md.

---

## 20. TC (Temporal Coherence) Learning

**Decision:** Implement TC(0) learning as an alternative to fixed-rate TD(0).
TC replaces the fixed learning rate α with per-weight adaptive rates based
on update coherence (Jaśkowski 2016, "Mastering 2048 with Delayed Temporal
Coherence Learning").

**Algorithm:** For each weight i touched during a TD update with raw error δ:
1. Compute coherence ratio: α_i = |E_i| / A_i (or 1.0 if A_i = 0)
2. Update weight: V_i += β × (α_i / m) × δ
3. Accumulate signed error: E_i += δ
4. Accumulate absolute error: A_i += |δ|

Where β is the meta-learning rate (default 1.0), m is total feature count
(base patterns × 8 symmetries). E and A persist across games, never reset.

**Rationale:**
- Paper reports 77% improvement over fixed-rate TD at 1-ply evaluation.
- Weights receiving consistent-sign updates (coherent) keep α_i near 1.0.
- Weights receiving oscillating updates (converged) see α_i → 0.0 automatically.
- No eligibility traces in TC(0) — simplest variant, same training loop structure.
- Compatible with Hogwild: E and A use same relaxed-atomic pattern as weights.

**Memory cost:** 3× weight storage (V + E + A tables of identical size).
4×6 patterns: 268 MB → 804 MB. 8×6 patterns: 537 MB → 1.6 GB.

**Design:** TcState is a separate struct (not embedded in NTupleNetwork).
The network is the inference artifact; TC accumulators are training-only state.
Save/load format unchanged — only V weights are persisted.

### Hogwild + TC beta tuning (8×6, hogwild-14, 1M games)

TC's coherence signal breaks down under Hogwild: concurrent threads push
different TD errors from different games into the same E/A accumulators,
making every weight look incoherent. Higher beta amplifies this, causing
weight divergence → Inf evaluations → NaN TD errors (Inf − Inf, IEEE 754).

Beta sweep results (1M games, 8×6 patterns, hogwild-14):

| β    | Final avg score | Status                |
|------|----------------:|:----------------------|
| 0.01 |          53,087 | Learning, but slow    |
| 0.05 |         106,320 | Good                  |
| 0.1  |         132,462 | Good                  |
| 0.3  |         163,034 | Good                  |
| 0.5  |         232,604 | **Best**              |
| 0.7  |          87,817 | Starting to diverge   |
| 1.0  |             744 | Diverged              |
| 1.5  |               0 | Completely diverged   |
| 2.0  |               0 | Completely diverged   |

**Conclusion:** β=0.5 is optimal for hogwild-14 with 8×6 patterns. The
paper's default β=1.0 works only in serial mode. The optimal β likely
scales inversely with thread count — more threads = more cross-game noise
in the coherence signal = lower β needed.

NaN guard added to `tc_update_orientation` and `TcState::accumulate` to
prevent catastrophic NaN poisoning, but the real fix is keeping β below
the divergence threshold.

---

