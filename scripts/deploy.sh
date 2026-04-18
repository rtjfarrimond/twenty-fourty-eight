#!/usr/bin/env bash
set -euo pipefail

# Full deploy: training stack + server stack.
#
# This restarts both services. If a training job is running it will be
# killed and marked failed. Use deploy-server.sh to deploy without
# interrupting training.
#
# See scripts/DEPLOYMENT.md for the deployment runbook.
#
# Usage: ./scripts/deploy.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/deploy-training.sh"
"$SCRIPT_DIR/deploy-server.sh"
