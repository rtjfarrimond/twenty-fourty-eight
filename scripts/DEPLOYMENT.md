# Deployment Runbook

## Scripts

| Script | What it deploys | Restarts |
|--------|----------------|----------|
| `deploy.sh` | Everything | Server + training daemon |
| `deploy-server.sh` | Server, frontend, config, models, skills | Server only |
| `deploy-training.sh` | Training daemon, CLI, generate_models, benchmark | Training daemon only |

## Dependency graph

```
queue  (standalone — no engine/model dependency)
  ↑           ↑
training    server
```

- **queue crate changes** affect both training and server — run `deploy.sh` (full deploy).
- **server-only changes** (server crate, frontend, config, models) — run `deploy-server.sh`. Training is unaffected.
- **training-only changes** (training crate, daemon behaviour) — run `deploy-training.sh`. Server is unaffected.

## Deploying while training is running

Use `deploy-server.sh`. It builds and restarts only the server and frontend.
The training queue daemon keeps running and the in-progress job completes
normally.

Do **not** run `deploy-training.sh` or `deploy.sh` while a job is running
unless you're willing to lose the current job. Both scripts stop the training
daemon, and the orphan-recovery sweep marks the interrupted job as failed on
next startup.

## Common scenarios

**Routine deploy (no training running):**
```bash
./scripts/deploy.sh
```

**Frontend/server fix while training is running:**
```bash
./scripts/deploy-server.sh
```

**Training code change (wait for current job to finish first):**
```bash
# Check if a job is running
ls /var/lib/2048-solver/queue/running/
# When empty:
./scripts/deploy-training.sh
```

**Skip benchmarks for a fast deploy:**
```bash
SKIP_BENCHMARKS=1 ./scripts/deploy-server.sh
```
