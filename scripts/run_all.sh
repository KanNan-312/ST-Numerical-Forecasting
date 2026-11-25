# !/usr/bin/env bash
# set -euo pipefail

MODELS=(
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
  "testam"
  "gc_moe"
  "dlinear"
  "patchtst"
  "itransformer"
  "timemixer"
  "timellm"
  "gpt4ts"
  "chronos2_zero"
  "timesfm_zero"
)


DATASETS=(
  # "pems08"
  # "redfin_metro_house"
  # "chicago_crime"
  # "nyc_crime"
  # "nyc_bike"
  # chicago_crime_weekly
  chicago_crime_daily
  chicago_crime
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