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

**Status:** Deferred — to be tackled immediately after dashboard work is complete.

**Approach:** Profile first, then fix. No guessing.
- Build a benchmarking/evaluation framework to measure games/sec and
  moves/sec.
- Aim to surpass existing tooling (moporgic/TDL2048 reports 102M moves/sec on
  Ryzen 9). We are building a best-in-class system.
- **Known likely bottlenecks** (to be confirmed by profiling):
  1. N-tuple index computation via repeated `get_tile` calls — should
     precompute tuple indices directly from 16-bit rows.
  2. Vec allocations in hot path (`empty_tiles` called every move).
  3. Naive transpose (16 get/set calls) — should be bitwise.
- Define a repeatable benchmark suite so we can measure before/after for every
  optimization.

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

**Remaining polish items:**
- Tile sliding animations (requires client-side state diffing)
- Replace unicode arrow characters with SVG/icon font for consistent
  cross-device rendering (current padding hack is browser-specific)

---

