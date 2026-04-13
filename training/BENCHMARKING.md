# Benchmarking & Profiling

## Prerequisites

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

The isomorphic approach was the breakthrough: store only 4 base patterns
instead of 32 expanded ones (4x less memory = better cache), transform the
board with fast bitwise flip/transpose, and use bitmask-based index extraction
(2 ops per pattern vs 6 shifts).

Target: 102M moves/sec (moporgic/TDL2048). Current gap: ~140x. Remaining
optimizations: compile-time pattern specialization, BMI2 pext64, further loop
unrolling, and possible SIMD.

Hardware: (record your own)
