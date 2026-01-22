#!/bin/bash

models=(
  "openrouter/openai/gpt-4.1-mini:nitro"
  "openrouter/google/gemini-2.5-flash-preview-09-2025:nitro"
  "openrouter/deepseek/deepseek-chat-v3-0324:nitro"
  "openrouter/qwen/qwen3-235b-a22b-2507:nitro"
  "openrouter/qwen/qwq-32b:nitro"
  "openrouter/qwen/qwen-2.5-coder-32b-instruct:nitro"
  "openrouter/meta-llama/llama-3-8b-instruct:nitro"
) # we use the :nitro suffix to select fastest provider available

# Load .env from repository root (one directory up from scripts/)
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
  echo "No .env file found at $ENV_FILE"
fi

# Default solver and option parsing
SOLVER="generate"

while getopts "c" opt; do
  case "${opt}" in
    c)
      SOLVER="chain_of_prompts"
      ;;
    *)
      echo "Usage: $0 [-c for chain_of_prompts solver]"
      exit 1
      ;;
  esac
done

echo "Using solver: $SOLVER"

for model in "${models[@]}"; do
  echo "Running evaluation for model: $model"
  uv run inspect eval ./he_rel/_registry.py@humanevalrel \
    --model "$model" \
    --solver "$SOLVER" \
    --max-tokens 10000 \
    --epochs 50 \
    --max-connections 100 \
    "
done
