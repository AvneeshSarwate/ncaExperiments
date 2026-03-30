#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

MODE_ARGS=("$@")
if [[ ${#MODE_ARGS[@]} -eq 0 ]]; then
  MODE_ARGS=("all")
fi

ROLLOUT_STEPS=10
idx=1
while [[ $idx -le ${#MODE_ARGS[@]} ]]; do
  arg="${MODE_ARGS[$idx]}"
  if [[ "$arg" == "--rollout-steps" ]]; then
    next_idx=$((idx + 1))
    ROLLOUT_STEPS="${MODE_ARGS[$next_idx]}"
    idx=$next_idx
  fi
  idx=$((idx + 1))
done

echo "== Regenerating PyTorch references =="
uv run python generate_reference.py --steps "$ROLLOUT_STEPS"
uv run python generate_op_reference.py

echo "== Rebuilding Metal library =="
xcrun -sdk macosx metal -c Sources/NCATrainer/nca_kernels.metal -o /tmp/nca_kernels.air
xcrun -sdk macosx metallib /tmp/nca_kernels.air -o nca_kernels.metallib

echo "== Building Swift target =="
swift build -c release

echo "== Running NCATrainer ${MODE_ARGS[*]} =="
./.build/release/NCATrainer "${MODE_ARGS[@]}"
