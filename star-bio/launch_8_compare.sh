#!/usr/bin/env bash
set -euo pipefail

# Launch eight independent four-fold experiments, one per GPU.
# Default mode is debug so this script never races eight git commits.
# Usage:
#   bash star-bio/launch_8_compare.sh
#   MODE=formal bash star-bio/launch_8_compare.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
# Force UTF-8 so Chinese Loguru/tqdm messages are readable in launcher.log.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
ENV_PYTHON="${PYTHON:-python}"
MODE="${MODE:-debug}"
EPOCHS="${EPOCHS:-300}"
WORKERS="${WORKERS:-4}"
BASE_ROOT="${BASE_ROOT:-star-bio/compare_runs}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-3407}"
LR="${LR:-0.0002}"
if [[ "$MODE" != "debug" && "$MODE" != "formal" ]]; then
  echo "MODE 必须是 debug 或 formal" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" ]]; then mkdir -p "$BASE_ROOT"; fi

if [[ "$MODE" == "formal" && "$DRY_RUN" == "1" ]]; then
  MODE_ARGS=(--commit-id DRY_RUN)
elif [[ "$MODE" == "formal" ]]; then
  git add star-bio
  git commit --allow-empty -m "star-bio eight experiment comparison $(date +%y-%m-%d@%H-%M-%S)"
  SHARED_COMMIT="$(git rev-parse --short HEAD)"
  MODE_ARGS=(--commit-id "$SHARED_COMMIT")
elif [[ "$MODE" == "debug" ]]; then
  MODE_ARGS=(--debug)
else
  echo "MODE 必须是 debug 或 formal" >&2
  exit 2
fi

# GPU, experiment name, fixed batch size, RevIN flag, pooling, drift flag, strength.
# GPUs 0..3 differ only in batch size. GPUs 4..7 compare to GPU 2.
configs=(
  "0 p64_bs4096       4096 --revin    mean      --no-batch-shift 0"
  "1 p64_bs1024       1024 --revin    mean      --no-batch-shift 0"
  "2 p64_bs256         256 --revin    mean      --no-batch-shift 0"
  "3 p64_bs64           64 --revin    mean      --no-batch-shift 0"
  "4 p64_bs256_norevin 256 --no-revin mean      --no-batch-shift 0"
  "5 p64_bs256_seg4    256 --revin    segments  --no-batch-shift 0"
  "6 p64_bs256_attn    256 --revin    attention --no-batch-shift 0"
  "7 p64_bs256_drift   256 --revin    mean      --batch-shift    0.25"
)

pids=()
for spec in "${configs[@]}"; do
  read -r gpu name batch revin pooling shift strength <<< "$spec"
  out="$BASE_ROOT/$name"
  log="$out/launcher.log"
  command=("$ENV_PYTHON" -u star-bio/train_patchtst.py
    --all-folds --epochs "$EPOCHS" --workers "$WORKERS"
    --gpu-id "$gpu" --device cuda --batch-size "$batch" --seed "$SEED" --lr "$LR"
    --sampler joint "$shift" --augmentation-strength "$strength"
    --patch-len 64 --stride 32 --d-model 128 --layers 3 --d-ff 256 --heads 8
    "$revin" --pooling "$pooling" --pool-segments 4
    "${MODE_ARGS[@]}" --output-root "$out")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  mkdir -p "$out"
  echo "launch GPU=$gpu name=$name output=$out"
  "${command[@]}" > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
