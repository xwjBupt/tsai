#!/usr/bin/env bash
set -euo pipefail

# Launch eight independent four-fold experiments, one per GPU.
# Default mode is debug so this script never races eight git commits.
# Usage:
#   bash star-bio/launch_8_compare.sh
#   MODE=formal bash star-bio/launch_8_compare.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
ENV_PYTHON="${PYTHON:-python}"
MODE="${MODE:-debug}"
EPOCHS="${EPOCHS:-300}"
WORKERS="${WORKERS:-4}"
BASE_ROOT="${BASE_ROOT:-star-bio/compare_runs}"
LOG_ROOT="${LOG_ROOT:-star-bio/compare_logs}"
mkdir -p "$BASE_ROOT" "$LOG_ROOT"

if [[ "$MODE" == "formal" ]]; then
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

# Each row: GPU, name, sampler, batch-shift option, augmentation strength,
# patch length, stride, transformer width, layers.
configs=(
  "0 baseline_joint  joint  --batch-shift       1.0 32 16 128 3"
  "1 no_shift        joint  --no-batch-shift    0.0 32 16 128 3"
  "2 class_sampler   class  --batch-shift       1.0 32 16 128 3"
  "3 uniform_sampler uniform --batch-shift      1.0 32 16 128 3"
  "4 aug_half        joint  --batch-shift       0.5 32 16 128 3"
  "5 aug_strong      joint  --batch-shift       1.5 32 16 128 3"
  "6 patch16         joint  --batch-shift       1.0 16 8  128 3"
  "7 patch64         joint  --batch-shift       1.0 64 32 128 3"
)

pids=()
for spec in "${configs[@]}"; do
  read -r gpu name sampler shift strength patch stride dmodel layers <<< "$spec"
  out="$BASE_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  echo "launch GPU=$gpu name=$name output=$out"
  "$ENV_PYTHON" star-bio/train_patchtst.py \
    --all-folds --epochs "$EPOCHS" --workers "$WORKERS" \
    --gpu-id "$gpu" --device cuda \
    --sampler "$sampler" $shift \
    --augmentation-strength "$strength" \
    --patch-len "$patch" --stride "$stride" \
    --d-model "$dmodel" --layers "$layers" --d-ff 256 \
    "${MODE_ARGS[@]}" --output-root "$out" \
    > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
