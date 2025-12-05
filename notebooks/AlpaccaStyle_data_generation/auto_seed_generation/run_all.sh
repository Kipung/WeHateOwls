#!/usr/bin/env bash
set -euo pipefail

# Run from inside auto_seed_generation directory

BASE_SEEDS="${1:-seed_tasks_original.jsonl}"
COMBINED_OUT="${2:-seed_tasks_combined.jsonl}"
TARGET_NEW="${3:-80}"
PARA_K="${4:-25}"                  # 0 = skip paraphrase
BACKEND="${5:-ollama}"             # ollama|openai
MODEL="${6:-mistral}"

GEN_O="gen_owl_seeds_ollama_only.py"
MERGE="merge_and_validate.py"
PARA="paraphrase_jsonl.py"
TEMPL="seed_templates.yaml"
OWL_JSONL="seed_tasks_owl.jsonl"

# 1) Generate owl-activating seeds
# generate
# inside run_all.sh generate section
python3 "$GEN_O" \
  --base "$BASE_SEEDS" \
  --out "$OWL_JSONL" \
  --target "$TARGET_NEW" \
  --similarity 0.75 \
  --owl_drop_rate 0.05 \
  --model "$MODEL" \
  --batch 12 \
  --max_calls 20



# 2) Optional paraphrase step
if [[ "$PARA_K" -gt 0 ]]; then
  if [[ "$BACKEND" == "ollama" ]]; then
    python3 "$PARA" "$OWL_JSONL" "$OWL_JSONL" "$PARA_K" ollama "$MODEL"
  elif [[ "$BACKEND" == "openai" ]]; then
    python3 "$PARA" "$OWL_JSONL" "$OWL_JSONL" "$PARA_K" openai "$MODEL"
  else
    echo "Unsupported backend: $BACKEND"; exit 1
  fi
fi

# 3) Merge and validate
python3 "$MERGE" "$BASE_SEEDS" "$OWL_JSONL" "$COMBINED_OUT"

echo "✅ Combined seeds written to: $COMBINED_OUT"
