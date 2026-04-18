# Future Ideas

### Two-tier model storage
Submitter chooses `models/` (auto-loaded, played live) vs `experiments/`
(catalogued, indexed in models.json, but never loaded by the server).
Promotion is `mv` between dirs. Would reduce server RSS from models that
are deployed but rarely viewed.

### LRU model eviction in server
Server loads models on demand (first user request) and evicts after N
minutes idle. Cheaper RAM, slower first-request latency. Orthogonal to
two-tier storage.

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

### Rust toolchain & target-cpu investigation
Pinned to rustc 1.93.0 because 1.94.1 produces broken TC-hogwild codegen
(model doesn't learn at all). Additionally, `target-cpu=native` in
`.cargo/config.toml` may affect training quality — builds without it
produce weaker models (80K avg vs 232K avg at 1M games, same code).
Need to:
1. Test recent toolchain versions (1.93.0, 1.94.0, 1.94.1, nightly) with
   a short TC-hogwild run (100K games) and compare learning curves.
2. Confirm `target-cpu=native` effect — clean build + deploy + 100K run
   with and without, compare results.
3. If 1.94.x is confirmed broken, file a rustc bug with a minimal repro.

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
