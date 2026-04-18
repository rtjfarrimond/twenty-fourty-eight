---
name: training
description: Submit training jobs to the 2048 solver training queue
user_invocable: true
---

# 2048 Training Job Submission

Submit training jobs via the queue daemon. The daemon manages execution,
writes logs where the server expects them, and powers the live training
view. Never run the `training` binary directly with `training run` — always
use `training submit`.

## Usage

```bash
training submit [OPTIONS]
```

## Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | `serial` | `serial`, `hogwild`, `tc`, or `tc-hogwild` |
| `--threads` | `1` | Worker threads (must be 1 for serial/tc) |
| `--games` | `100000` | Total training games |
| `--eval-interval` | `10000` | Evaluate every N games |
| `--eval-games` | `1000` | Games per eval checkpoint |
| `--learning-rate` | `0.0025` | Alpha for TD, beta for TC algorithms |
| `--patterns` | `4x6` | `4x6` (4 base patterns) or `8x6` (8 base patterns) |
| `--model-name` | `ntuple-4x6-td0` | Output model name (used in filenames) |
| `--optimistic-init` | `0.0` | Constant weight init value |
| `--ephemeral` | `false` | Skip saving .bin — only logs + config deployed |
| `--description` | auto | Human-readable description for metadata |

## Algorithm guidance

- **TD algorithms** (`serial`, `hogwild`): use `--learning-rate 0.0025`
- **TC algorithms** (`tc`, `tc-hogwild`): `--learning-rate` is the beta
  meta-learning rate. Use `1.0` for serial TC, `0.5` for tc-hogwild with
  14 threads on 8x6 patterns. Higher beta diverges under hogwild.
- `8x6` patterns produce stronger models but use 2x memory (537 MB vs 257 MB
  per model, 1.6 GB vs 804 MB with TC state)

## When to use --ephemeral

Use for parameter sweeps, ablations, and quick experiments where the
training curve is the goal, not the model weights. The .log.jsonl and
.config.json are still deployed so results appear in the dashboard.
Follow up with a non-ephemeral run using the best parameters.

## Examples

Standard TD training:
```bash
training submit --algorithm hogwild --threads 14 --games 1000000 \
  --patterns 8x6 --model-name ntuple-8x6-td0-1M
```

TC learning sweep (ephemeral):
```bash
for beta in 0.1 0.3 0.5; do
  sleep 1
  training submit --algorithm tc-hogwild --threads 14 --games 1000000 \
    --learning-rate "$beta" --patterns 8x6 --ephemeral \
    --model-name "ntuple-8x6-tc-b${beta}-1M"
done
```

Production TC run (best beta, saved):
```bash
training submit --algorithm tc-hogwild --threads 14 --games 10000000 \
  --learning-rate 0.5 --patterns 8x6 \
  --model-name ntuple-8x6-tc-10M
```

## Queue management

```bash
training queue list          # Show all jobs by state
training queue cancel <id>   # Cancel a pending job
```

## Important

- Always add `sleep 1` between multiple `training submit` calls in loops
  — job IDs use unix timestamps for FIFO ordering and same-second
  submissions get random order.
- The daemon executes jobs sequentially (FIFO). Check queue status to
  see position.
