# !/usr/bin/env bash
# set -euo pipefail

MODELS=(
  # "dlinear"
  # "patchtst"
  # "itransformer"
  # "timemixer"
  # "timellm"
  # "gpt4ts"
  # "chronos2_zero"
  "timesfm_zero"
  "dcrnn"
  "stgcn"
  "graph_wavenet"
  "acgrn"
  "mtgnn"
  "stid"
  "staeformer"
  "d2stgnn"
  "gc_moe"
  "testam"
)


DATASETS=(
  "chicago_crime"
)

for DATASET in "${DATASETS[@]}"; do
  echo "=== Dataset: ${DATASET} ==="

  for MODEL in "${MODELS[@]}"; do
    echo "Launching dataset=${DATASET}, model=${MODEL}"
    python scripts/run_one.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --device cuda
  done

  echo "Generating report for ${DATASET}..."
  python scripts/make_report.py \
    --runs "runs/${DATASET}/" \
    --out "runs/${DATASET}/report"
done

echo "All jobs finished."