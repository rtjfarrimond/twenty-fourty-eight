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

Key optimizations (in order of impact):
1. **Isomorphic evaluation** — store 4 base patterns, transform the board 8
   ways with bitwise flip/transpose. 4x less memory = better cache.
2. **BMI2 PEXT** — single-instruction index extraction replaces multi-op
   bitmask approach.
3. **Value caching** — cache network evaluation alongside move candidates to
   eliminate redundant evaluations.
4. **Eliminated redundant game-over checks** — `best_afterstate` returning
   `None` already signals game over.

Target: 102M moves/sec (moporgic/TDL2048). Current gap: ~100x. Remaining
optimizations: compile-time pattern specialization, further loop unrolling,
multithreading, and possible SIMD.

Hardware: (record your own)
