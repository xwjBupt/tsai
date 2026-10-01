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
mkdir -p "$BASE_ROOT"

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
  "0 patch64_joint       joint   --batch-shift       1.0 64 32 256 3 512"
  "1 patch64_no_shift    joint   --no-batch-shift    0.0 64 32 256 3 512"
  "2 patch64_aug025      joint   --batch-shift       0.25 64 32 256 3 512"
  "3 patch64_aug050      joint   --batch-shift       0.5 64 32 256 3 512"
  "4 patch64_class       class   --batch-shift       1.0 64 32 256 3 512"
  "5 patch64_uniform     uniform --batch-shift       1.0 64 32 256 3 512"
  "6 patch64_d384        joint   --batch-shift       0.5 64 32 384 4 768"
  "7 patch48_joint       joint   --batch-shift       0.5 48 24 256 3 512"
)

pids=()
for spec in "${configs[@]}"; do
  read -r gpu name sampler shift strength patch stride dmodel layers dff <<< "$spec"
  out="$BASE_ROOT/$name"
  mkdir -p "$out"
  log="$out/launcher.log"
  echo "launch GPU=$gpu name=$name output=$out"
  "$ENV_PYTHON" star-bio/train_patchtst.py \
    --all-folds --epochs "$EPOCHS" --workers "$WORKERS" \
    --gpu-id "$gpu" --device cuda \
    --sampler "$sampler" $shift \
    --augmentation-strength "$strength" \
    --patch-len "$patch" --stride "$stride" \
    --d-model "$dmodel" --layers "$layers" --d-ff "$dff" \
    "${MODE_ARGS[@]}" --output-root "$out" \
    > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
