#!/usr/bin/env bash
# SHA-256 of the frozen artifacts recorded in analysis_plan.md Section 10.
# Run from anywhere: bash scripts/hash_frozen.sh
# Re-run after the second-pass label review and after any DECISION edit to
# task_contract.md or prompts/oracle.json, then update Section 10.
set -euo pipefail

cd "$(dirname "$0")/.."

files=(
  task_contract.md
  config/experiment.json
  data/feedback.jsonl
  data/selection.jsonl
  data/test_inputs.jsonl
  data/test_labels.jsonl
  data/generated_test_inputs.jsonl
  data/generated_test_labels.jsonl
  data/split_manifest.json
)

while IFS= read -r path; do
  files+=("$path")
done < <(find prompts -type f | LC_ALL=C sort)

# data/generated_test_*.jsonl are empty placeholders for the dropped Test-G
# extension; they are hashed so the full file set is covered.
LC_ALL=C sha256sum "${files[@]}"
