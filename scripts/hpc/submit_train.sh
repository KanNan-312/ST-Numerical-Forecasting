#!/bin/bash
# Submit one scripts/run_one.py training run as a GPU PBS job.
#
# The job itself (scripts/hpc/train.pbs):
#   1. copies the repo to a scratch directory, excluding runs/ and every
#      dataset folder except the one you asked for
#   2. brings up the Python environment (scripts/hpc/hpc.env's
#      setup_environment -- edit that once per cluster)
#   3. runs scripts/run_one.py there with --device cuda
#   4. every 60s (and once more at the end) mirrors the in-progress run
#      folder + live log back to this repo's runs//logs/ -- see "Tracking
#      progress" below
#   5. copies back the final result and frees the scratch directory
#
# Setup (once per cluster):
#   cp scripts/hpc/hpc.env.example scripts/hpc/hpc.env
#   <edit scripts/hpc/hpc.env: SCRATCH_BASE, queue, project code, setup_environment()>
#
# Usage:
#   scripts/hpc/submit_train.sh --dataset metr_la --model stgcn
#   scripts/hpc/submit_train.sh --dataset dc_house --model dlinear --task univariate \
#     --walltime 02:00:00 --ncpus 8 --mem 64gb \
#     -- --seq-len 12 --pred-len 12          # extra args, forwarded as-is to run_one.py
#
# Tracking progress (no need to just wait and hope):
#   qstat -u $USER                  # queued/running/finished, from your home/login shell
#   qstat -f <jobid>                # full detail: elapsed time, assigned node, resources
#   tail -f logs/<jobname>.o        # PBS's own stdout/stderr (via -k oe), updated live
#   tail -f logs/<PBS job ID>.log   # the same training log, mirrored back every 60s by
#                                    # train.pbs -- this is where per-epoch progress
#                                    # (tqdm/epoch_time_sec/train_mse lines) actually shows up
#   watch -n 30 'find runs/<dataset> -newer /tmp/.x -type f'   # see which run files have
#                                    # landed since the job started (any recent-mtime marker works)
#   The final checkpoint.pt/metrics.json for a given run only appear once that run's
#   fit()+eval() finishes (not written incrementally) -- the live log is what shows you
#   it's actually progressing before then, not stuck.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -f "$SCRIPT_DIR/hpc.env" ]]; then
  echo "scripts/hpc/hpc.env not found." >&2
  echo "Run: cp scripts/hpc/hpc.env.example scripts/hpc/hpc.env" >&2
  echo "then edit it for your cluster (queue, project code, scratch path, environment setup)." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hpc.env"

DATASET=""
MODEL=""
TASK="multivariate"
WALLTIME="${DEFAULT_WALLTIME:-04:00:00}"
NCPUS="${DEFAULT_NCPUS:-4}"
MEM="${DEFAULT_MEM:-32gb}"
GPUS="${DEFAULT_GPUS:-1}"
QUEUE="${DEFAULT_QUEUE:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --ncpus) NCPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1 (use -- to pass extra run_one.py args)" >&2; exit 1 ;;
  esac
done

if [[ -z "$DATASET" || -z "$MODEL" ]]; then
  echo "usage: $0 --dataset NAME --model NAME [--task multivariate|univariate]" >&2
  echo "          [--walltime HH:MM:SS] [--ncpus N] [--mem SIZEgb] [--gpus N] [--queue NAME]" >&2
  echo "          [-- extra run_one.py args, e.g. --seq-len 12 --pred-len 12]" >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR/data/$DATASET" ]]; then
  echo "data/$DATASET not found under $REPO_DIR -- nothing to copy to scratch" >&2
  exit 1
fi

mkdir -p "$REPO_DIR/logs"
JOB_NAME="stnf_${DATASET}_${MODEL}"

echo "Submitting: dataset=$DATASET model=$MODEL task=$TASK" \
     "resources=ncpus=${NCPUS},mem=${MEM},ngpus=${GPUS},walltime=${WALLTIME}" \
     "${QUEUE:+queue=$QUEUE}"

# cd to the repo root first so $PBS_O_WORKDIR (what train.pbs treats as the
# repo root) is this repo, regardless of where submit_train.sh itself was
# invoked from -- PBS sets it to qsub's own cwd at submission time, matching
# the facility's own example job script's use of $PBS_O_WORKDIR.
cd "$REPO_DIR"
qsub \
  -N "$JOB_NAME" \
  -l "ncpus=${NCPUS}" \
  -l "ngpus=${GPUS}" \
  -l "mem=${MEM}" \
  -l "walltime=${WALLTIME}" \
  ${QUEUE:+-q "$QUEUE"} \
  ${PROJECT_CODE:+-P "$PROJECT_CODE"} \
  -o "$REPO_DIR/logs/${JOB_NAME}.o" \
  -e "$REPO_DIR/logs/${JOB_NAME}.e" \
  -k oe \
  -v "DATASET=${DATASET},MODEL=${MODEL},TASK=${TASK},SCRATCH_BASE=${SCRATCH_BASE},RUN_ARGS=${EXTRA_ARGS[*]}" \
  "$SCRIPT_DIR/train.pbs"
