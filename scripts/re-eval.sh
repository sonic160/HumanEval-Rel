#!/usr/bin/env bash
# Re-evaluate all .eval files in the results directory using @verify
# This is useful if the verification script has been updated and you want to
# re-score previous results.

set -euo pipefail
shopt -s nullglob
mkdir -p ./results/reeval

for file in ./results/*.eval; do
  echo "Processing $file"
  out="./results/reeval/$(basename "$file")"
  uv run inspect score "$file" \
    --scorer ./he_rel/scoring/code_tester.py@verify \
    --action overwrite \
    --output-file "$out"
done
