#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 SEED GPU OUTPUT_DIR" >&2
  exit 2
fi

seed="$1"
gpu="$2"
output_dir="$3"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${LCRSEG_PYTHON:-/root/.venvs/lcrseg-py310/bin/python}"

if [ ! -x "$python_bin" ]; then
  echo "Gate 0 Python interpreter is not executable: $python_bin" >&2
  exit 2
fi

mkdir -p "$output_dir"
cd "$repo_root"
set +e
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES="$gpu" \
"$python_bin" scripts/run_gate0_repaired.py \
  --config configs/gate0_repaired/fundus.yaml \
  --seed "$seed" \
  --output-dir "$output_dir" \
  --device cuda
status="$?"
set -e
printf '%s\n' "$status" > "$output_dir/.exit"
exit "$status"
