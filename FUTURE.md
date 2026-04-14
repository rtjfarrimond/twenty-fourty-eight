# Future Ideas

### Proper user/group for services and data
Currently `/var/lib/2048-solver/` is root-owned and the server runs as root.
Rob has been chowned the data dirs as a workaround to avoid sudo for training.
Proper fix: create a `2048-solver` system user + group, run the server as
that user, add rob to the group, chmod g+w the data dirs. Update the systemd
unit and deploy script accordingly.

### Training job queue
Persistent queue of training runs that survives reboots, lets you inspect
what's pending, reorder, and kill jobs. Today we chain with `&&`, which is
fine for sequential overnight runs but can't be introspected or edited.

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
