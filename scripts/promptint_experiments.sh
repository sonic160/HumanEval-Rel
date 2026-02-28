#!/bin/bash
#
# Run V0, V1, V1-RD, and V2 experiments with gpt-4.1-mini
#
# Usage:
#   ./scripts/run_experiments.sh          # Run all versions
#   ./scripts/run_experiments.sh v0       # Run only V0
#   ./scripts/run_experiments.sh v1       # Run only V1 (self-debugging)
#   ./scripts/run_experiments.sh v1rd     # Run only V1 with rubber duck
#   ./scripts/run_experiments.sh v2       # Run only V2 (alphacodium)

set -euo pipefail

# --- Configuration ---
MODEL="openrouter/openai/gpt-4.1-mini:nitro"
EPOCHS=50
MAX_TOKENS=10000
MAX_CONNECTIONS=100
REGISTRY="./he_rel/_registry.py@humanevalrel"
LOG_DIR="results/"

# --- Load environment ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "WARNING: No .env file found at $ENV_FILE"
fi

cd "$ROOT_DIR"

# --- Solver definitions ---
declare -A SOLVERS
SOLVERS[v0]="chain_of_prompts"
SOLVERS[v1]="self_debugging"
#SOLVERS[v1rd]="self_debugging_rd"
SOLVERS[v2]="alphacodium"

# --- Determine which versions to run ---
if [ $# -eq 0 ]; then
  VERSIONS=("v0" "v1" "v2")
else
  VERSIONS=("$@")
fi

# --- Run experiments ---
for version in "${VERSIONS[@]}"; do
  solver="${SOLVERS[$version]:-}"
  if [ -z "$solver" ]; then
    echo "ERROR: Unknown version '$version'. Valid: v0, v1, v2"
    exit 1
  fi

  echo ""
  echo "============================================================"
  echo "  Running $version — solver: $solver"
  echo "  Model: $MODEL"
  echo "  Epochs: $EPOCHS"
  echo "============================================================"
  echo ""

  uv run inspect eval "$REGISTRY" \
    --model "$MODEL" \
    --solver "$solver" \
    --max-tokens "$MAX_TOKENS" \
    --epochs "$EPOCHS" \
    --max-connections "$MAX_CONNECTIONS" \
    --log-dir "$LOG_DIR"

  echo ""
  echo "  ✓ $version completed"
  echo ""
done

echo ""
echo "============================================================"
echo "  All experiments completed."
echo "  Results saved to: $LOG_DIR"
echo "============================================================"
