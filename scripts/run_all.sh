#!/usr/bin/env bash
# set -euo pipefail
#
# Full sweep: every model in the registry x every dataset below, then (per
# dataset, since it's window-scoped and can't mix datasets) a case-library
# build, then (once, at the very end) one combined report over everything
# in runs/ -- make_report.py already groups its output by dataset x task
# on its own, so a single final call covers every dataset swept here.
#
# set -euo pipefail stays commented out on purpose: one model failing on
# one dataset (e.g. a requires_graph=True model on a dataset with no
# graph.npz) shouldn't stop the rest of the sweep -- failures show up in
# this script's own stdout/the run's logs, not as a hard stop.

MODELS=(
  "dcrnn"
  "stgcn"
  "stid"
  "agcrn"
  "graph_wavenet"
  "staeformer"
  "mtgnn"
  "gman"
  "d2stgnn"
  "dlinear"
  "itransformer"
  "patchtst"
  "timemixer"
  "gc_moe"
  "testam"
  "ensemble_st"
  "xgb_stack"
)

DATASETS=(
  "pems03"
  "pems04"
  "pems07"
  "pems08"
  "metr_la"
  "pems_bay"
  # "chicago_crime_weekly"
  "nyc_taxi"
  "nyc_bike"
  "nyc_crime"
)

# dataset -> window label (w{seq_len}_h{pred_len}, from each dataset's own
# configs/dataset/<name>.yaml -- NOT all the same: chicago_crime_weekly is
# w4_h4, nyc_crime is w12_h3, everything else here is w12_h12).
declare -A WINDOWS=(
  [pems03]="w12_h12"
  [pems04]="w12_h12"
  [pems07]="w12_h12"
  [pems08]="w12_h12"
  [metr_la]="w12_h12"
  [pems_bay]="w12_h12"
  [chicago_crime_weekly]="w4_h4"
  [nyc_taxi]="w12_h12"
  [nyc_bike]="w12_h12"
  [nyc_crime]="w12_h3"
)

for DATASET in "${DATASETS[@]}"; do
  echo "=== Dataset: ${DATASET} ==="
  WINDOW="${WINDOWS[$DATASET]}"

  for MODEL in "${MODELS[@]}"; do
    echo "Launching dataset=${DATASET}, model=${MODEL}"
    python scripts/run_one.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --device cuda
  done

  echo "Building case library for ${DATASET} (${WINDOW})..."
  python scripts/build_case_library.py \
    --dataset "${DATASET}" --window "${WINDOW}" \
    --out-dir "runs/${DATASET}/${WINDOW}/case_library"
done

echo "Generating combined report over runs/..."
python scripts/make_report.py \
  --runs "runs/" \
  --out "runs/report"

echo "All jobs finished."
