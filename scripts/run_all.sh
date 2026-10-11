# !/usr/bin/env bash
# set -euo pipefail

MODELS=(
  "dlinear"
  "patchtst"
  "itransformer"
  "timemixer"
  "timellm"
  "gpt4ts"
  "chronos2_zero"
  "timesfm_zero"
  "staeformer"
  "stid"
  "agcrn"
  "mtgnn"
  "stgcn"
  "stsgcn"
  "graph_wavenet"
  "dcrnn"
  "d2stgnn"
  "gman"
  "astgcn"
)


DATASETS=(
  # "chicago_crime"
  "pems08"
  "redfin_metro_house"
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