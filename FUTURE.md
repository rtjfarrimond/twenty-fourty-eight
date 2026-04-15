# Future Ideas

### Experiments without deploy or persistence
Every K-sweep, ablation, or quick probe currently produces a permanent
model file: ~257MB for 4x6 or ~513MB for 8x6, auto-loaded into server RAM
and given a continuous game loop. After ~20 experiments the server is at
~7GB RSS holding mostly dead-weight models we only ever needed for a
single results-table data point. The data we actually want (final eval +
training curve) is a few KB; the .bin payload is irrelevant once captured.

Concrete options, pick or combine:
1. **Two-tier storage.** Submitter chooses `models/` (auto-loaded, played
   live) vs `experiments/` (catalogued, indexed in models.json, but never
   loaded by the server). Promotion is `mv` between dirs.
2. **No-persist mode.** `--ephemeral` flag on `training submit` writes
   only the eval log + config to a `runs/` dir; no .bin saved.
   Dashboard shows the curve, server never sees the model. Best for
   K-sweeps and ablations where the model artefact has zero downstream
   use.
3. **LRU eviction in server.** Server loads on demand (first user request
   for that model) and evicts after N minutes idle. Cheaper RAM, slower
   first-request latency. Orthogonal to (1) and (2).

Minimum viable: (2) — most of our recent experimentation produces no
artefact worth keeping. The K-sweep would have generated 8 ephemeral runs
and we'd have the same data with zero ongoing cost.

### Cancel running training job
The queue daemon can cancel pending jobs but not in-flight ones. Add
SIGTERM-the-child + cleanup-partial-artefacts (.log.jsonl, .bin in
progress) to make `training queue cancel <id>` work for any state.

### Quantify benchmark variance on the dashboard
The benchmarks dashboard currently shows a single run per config, regenerated
on every deploy. Numbers visibly drift between deploys because they include
run-to-run noise from: TD RNG, memory-bandwidth contention from other
processes at deploy time, scheduler decisions, and thermal state. A reader
can't tell whether "hogwild-15: 2.77× today, 3.1× tomorrow" reflects a real
trend or just stochastic variance.

Options, pick one or combine:
1. **Multiple runs per config, show mean ± stddev.** Extend bench-matrix
   to run each (algorithm, threads) N times, emit stats aggregated across
   runs. Dashboard plots means with error bars. Most rigorous; ~N× deploy
   time cost.
2. **Bigger games count so variance is negligible.** Law-of-large-numbers
   approach. Bump default `GAMES` in bench-matrix from 10k to 50-100k.
   Cheap, but doesn't eliminate memory-bandwidth contention variance.
3. **Caveat on the page.** Add a one-liner noting single-run data and
   expected ~10% variance. Minimum-effort band-aid if the real fix is
   deferred.

Minimum viable: (3) + (2). Ideal: (1) with N=3 runs per config.

### Parallel training (Hogwild-style)
Training currently uses a single core — we observed one core pegged at 100%
while the other 15 idle during the 8x6 10M run. Adopt Hogwild-style
shared-memory parallelism: N worker threads, each playing independent games,
reading and writing the shared `NTupleNetwork` weight tables directly via
relaxed atomics (or unsynchronized writes — TD(0) tolerates the races). No
locks, no message queue.

**Why Hogwild, not message-passing or map-reduce:**
- Workers need to *read* weights constantly (~64 feature lookups per move).
  A single-writer/queue design doesn't help the read path, which dominates.
- Updates are high-frequency and tiny (atomic `+=` on f32, ~1ns). A channel
  send is ~100ns — the queue itself becomes the bottleneck.
- Updates are commutative; staleness from race conditions is statistically
  negligible because the table is huge and updates are sparse.
- Empirically validated: Niu et al. 2011 (Hogwild paper); Wu et al. and
  moporgic/TDL2048 for 2048 specifically.

**Expected gain:** ~10-12× on 16 cores (sublinear — cache contention on the
shared table). A 10M 8x6 run drops from ~20 hours to ~2.

**Implementation sketch:**
- `rayon::scope` or `std::thread::scope` for the worker pool. Async/tokio
  is wrong — this is CPU-bound, not I/O.
- Weight table as `Vec<AtomicU32>` reinterpreted as f32, or `UnsafeCell`
  with relaxed ordering.
- Per-thread RNG and game state.
- Reserve 1-2 cores for the web server via `taskset` / CPU affinity or
  `nice -n 19` on the training process. Goal: training saturates N-2 cores,
  server never starves.
- Measure games/sec and core scaling efficiency (speedup ÷ cores) as part
  of the phase 1 performance work.

**Metrics this unlocks:** games/sec and wall-clock-to-N-million-games are
named SotA axes in `DESIGN_DECISIONS.md` §17 — this is the first big lever
on both.

**Potential follow-ups beyond Hogwild:**
- Per-thread local accumulators merged periodically (reduces already-small
  atomic contention, adds small real staleness between syncs). Measure
  before adopting.
- SIMD/PEXT-based feature indexing inside the hot move loop (orthogonal to
  parallelism, multiplies whatever single-thread perf we have).
- NUMA-aware thread pinning if we ever scale past one socket.

### Memory-layout optimization for training bandwidth ceiling
Hogwild on our current machine tops out at 2.77× on 15 threads — not
because of contention but because the weight table (268 MB for 4x6, 537 MB
for 8x6) vastly exceeds L3 cache (16 MiB), so every feature lookup is a
DRAM access. `perf stat` measured IPC collapsing from 1.24 (serial) to
0.33 (hogwild-15) — textbook memory-stall signature. See
`DESIGN_DECISIONS.md` §12 for full profiling results.

The realistic ceiling on *any* thread count is memory bandwidth. To push
past 2.77× we need to reduce bytes-per-move, not add cores. Ranked options:

1. **f16 weights.** Halves memory traffic on the read and write paths.
   Precision tradeoff: TD updates are small (~α·δ ≈ 1e-3 scale) so f16
   precision loss could accumulate. Needs empirical convergence
   comparison vs. f32. Expected gain: up to 2× if precision holds.
2. **Packed/cache-aligned layout.** Current layout: one large 16^6 table
   per base pattern, accessed at scattered PEXT-derived indices. Each
   lookup touches one 64-byte cache line and uses only 4 bytes of it —
   94% waste. Options:
   - Interleave small groups of adjacent indices across patterns so one
     cache line serves multiple lookups.
   - Pre-gather weights for hot index ranges into a separate small table
     that fits in L2.
   - Both ideas need access-pattern profiling first — 2048 play doesn't
     visit all 16.7M indices uniformly, so there may be exploitable
     skew.
3. **Fewer lookups per move.** E.g. early-exit on clearly-dominated
   afterstates, or reuse the value of unchanged boards across repeated
   evaluations. Algorithmic, harder to do correctly, probably smaller
   gain than #1 and #2.
4. **Prefetching.** PEXT-derived indices make software prefetch hard
   because each feature's next access depends on the current board's
   bits. Hardware prefetcher already does what it can.

**Why this is deferred:** it's deep work — each option needs a careful
convergence A/B to confirm it doesn't wreck learning, plus its own
throughput benchmark. Estimate: half a week of focused effort, not a
side-task. The 2.77× win from Hogwild alone is already useful for
near-term experiment iteration.

**How to apply:** when revisiting, use the benchmark harness to establish
the baseline (current numbers live in `DESIGN_DECISIONS.md` §12), try
one optimization at a time, run A/B with convergence parity check
(same-seed training to N games, compare avg score). Don't stack changes
without isolating each one's contribution.
