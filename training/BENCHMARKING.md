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

## Baseline (pre-optimization)

- **798 games/sec**
- **~153K moves/sec**
- Config: 4 base 6-tuple patterns x 8 symmetries = 32 patterns
- Hardware: (record your own)
