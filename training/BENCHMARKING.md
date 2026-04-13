# Benchmarking & Profiling

## Hardware Requirements

The training crate uses BMI2 `PEXT` instructions for fast n-tuple index
extraction. This requires:
- **x86_64** architecture
- **Intel Haswell (2013)** or later, or **AMD Zen 3 (2020)** or later

Note: AMD Zen 1/2 have BMI2 but implement PEXT in microcode (~18 cycles vs
1 cycle on Intel/Zen 3+). Performance will be significantly worse on those
CPUs.

The build is configured with `target-cpu=native` (see `.cargo/config.toml`)
which enables the host CPU's full instruction set. Binaries are **not
portable** across different CPU generations.

To verify your CPU supports BMI2:
```sh
grep -o 'bmi2' /proc/cpuinfo | head -1
```

## Profiling Prerequisites

```sh
sudo apt install linux-tools-common linux-tools-$(uname -r)
cargo install flamegraph
```

Flamegraph uses `perf` which requires access to hardware performance counters.
Check your current setting:

```sh
cat /proc/sys/kernel/perf_event_paranoid
```

If it's above 1, lower it so your user can profile their own processes:

```sh
sudo sysctl kernel.perf_event_paranoid=1
```

To persist across reboots, add to `/etc/sysctl.conf`:

```
kernel.perf_event_paranoid = 1
```

Setting 1 allows userspace profiling of your own processes without exposing
kernel internals. Safe for a single-user dev machine.

## Running the benchmark

```sh
cargo build --release --bin benchmark
./target/release/benchmark 10000
```

The argument is the number of training games to run. Default is 10,000.

## Generating a flamegraph

```sh
cargo flamegraph --release --bin benchmark -- 5000
```

Opens or writes `flamegraph.svg` in the current directory.

## Results

Config: 4 base 6-tuple patterns x 8 symmetries = 32 patterns.

| Version | Games/sec | Moves/sec | Speedup |
|---|---|---|---|
| Baseline | 798 | 153K | 1.0x |
| + raw u64 bit shifts | 1,173 | 178K | 1.2x |
| + unrolled 6-tuple index | 1,400 | 281K | 1.8x |
| + value caching, no redundant evals | 1,697 | 343K | 2.2x |
| + flat weight array | 2,084 | 311K | 2.0x |
| + isomorphic eval (transform board, not patterns) | 2,666 | 734K | 4.8x |
| + BMI2 PEXT index extraction | 4,136 | 1.01M | 6.6x |
| + stack-allocated empty tiles | 4,514 | 1.20M | 7.8x |
| + batch afterstates (single transpose) | 5,132 | 1.26M | 8.2x |
| + LTO + codegen-units=1 | 5,103 | 1.29M | 8.4x |

Key optimizations (in order of impact):
1. **Isomorphic evaluation** — store 4 base patterns, transform the board 8
   ways with bitwise flip/transpose. 4x less memory = better cache.
2. **BMI2 PEXT** — single-instruction index extraction (note: microcoded
   ~18 cycles on AMD Zen 2, but still faster than software alternative).
3. **Value caching** — cache network evaluation alongside move candidates to
   eliminate redundant evaluations in TD update.
4. **Eliminated redundant game-over checks** — `best_afterstate` returning
   `None` already signals game over.
5. **Zero-allocation hot path** — stack-allocated empty tile list.
6. **Batch afterstates** — single transpose for both vertical moves.
7. **Bitwise board transpose** — delta-swap approach from reference impl.
8. **LTO** — link-time optimization for cross-crate inlining.

## Current bottleneck: memory bandwidth

IPC is 1.19 (vs theoretical 4+). Cache miss rate is 24.6%. The 256MB weight
table (4 patterns × 16M entries × 4 bytes) far exceeds the 16MB L3 cache on
this AMD EPYC Rome (Zen 2).

The reference (102M moves/sec) runs on a Ryzen 9 with:
- 64MB L3 cache (4x ours)
- Single-cycle PEXT (vs ~18 cycles on Zen 2)
- Higher single-thread clock speed

Expected performance on equivalent hardware (Zen 3+, 64MB L3): ~5-10x current
= 6-13M moves/sec. The remaining gap to 102M would be from C++ template
specialization and years of micro-optimization.

## Failed experiments

- **Row-based precomputed index tables** — 8MB of lookup tables thrashed L2
  cache. Slower than direct bit extraction.
- **Software prefetching** — batch-computing all 32 offsets then prefetching
  added loop overhead that outweighed the latency hiding benefit.

Hardware: AMD EPYC-Rome (Zen 2), 16 vCPUs, 16MB L3 cache, 30GB RAM

## Hardware upgrade path

A Hetzner dedicated server (AX series) with AMD Ryzen 5000/7000 would give:
- Single-cycle PEXT (vs ~18 cycles on current Zen 2)
- 32-64MB uncontested L3 cache (vs 16MB shared on current VPS)
- Estimated 5-10x speedup → 6-13M moves/sec

Options: AX41-NVMe or AX52 (~EUR 44-58/mo). Check current Hetzner lineup.

Hardware: (record your own)
