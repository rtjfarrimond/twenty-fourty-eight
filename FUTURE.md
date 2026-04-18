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
artefact worth keeping. The TC beta sweep generated ~10 useless 537MB
model files.

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

### Multi-stage training
Train separate value functions by max-tile threshold (e.g. stage 1
targets 2048, stage 2 targets 4096, etc.). Papers report +30-50%
improvement in convergence quality. Weight promotion between stages.
Pairs well with TC learning.

### Expectimax search at inference
Currently greedy 1-ply (pick max reward + V(afterstate)). Adding 2-3
ply expectimax at test time dramatically improves play quality without
changing learned weights. TDL2048+ jumps from ~412K avg (1-ply) to
~625K (6-ply). Inference-only change — no training impact.
