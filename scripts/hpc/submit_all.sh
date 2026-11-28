#!/bin/bash
# Submit scripts/hpc/train_all.pbs -- the GPU/scratch equivalent of
# scripts/run_all.sh. Edit the DATASETS/MODELS arrays directly in
# train_all.pbs (same as you'd edit them in run_all.sh), then just:
#
#   scripts/hpc/submit_all.sh                      # resources from hpc.env's defaults
#   scripts/hpc/submit_all.sh --walltime 48:00:00  # a bigger batch needs more time
#
# Unlike scripts/hpc/submit_train.sh (for one single one-off dataset+model
# run), this takes no --dataset/--model -- the batch to run is defined in
# train_all.pbs itself.
#
# Tracking progress: same as submit_train.sh -- qstat -u $USER, tail -f
# logs/stnf_train_all.o, tail -f logs/<jobid>.log (mirrored back every 60s).

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

TASK="multivariate"
WALLTIME="${DEFAULT_WALLTIME:-24:00:00}"
NCPUS="${DEFAULT_NCPUS:-4}"
MEM="${DEFAULT_MEM:-32gb}"
GPUS="${DEFAULT_GPUS:-1}"
QUEUE="${DEFAULT_QUEUE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --ncpus) NCPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $1 -- edit DATASETS/MODELS in train_all.pbs instead of passing them here" >&2; exit 1 ;;
  esac
done

mkdir -p "$REPO_DIR/logs"
JOB_NAME="stnf_train_all"

echo "Submitting batch job (edit scripts/hpc/train_all.pbs's DATASETS/MODELS arrays to change what runs)"
echo "resources: ncpus=${NCPUS}, mem=${MEM}, ngpus=${GPUS}, walltime=${WALLTIME}${QUEUE:+, queue=$QUEUE}"

# cd to the repo root first so $PBS_O_WORKDIR (what train_all.pbs treats as
# the repo root) is this repo, regardless of where this script was invoked
# from -- PBS sets it to qsub's own cwd at submission time.
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
  -v "TASK=${TASK},SCRATCH_BASE=${SCRATCH_BASE}" \
  "$SCRIPT_DIR/train_all.pbs"
