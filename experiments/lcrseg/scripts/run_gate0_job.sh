#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 C0|B0 SEED GPU OUTPUT_DIR" >&2
  exit 2
fi
variant="$1"
seed="$2"
gpu="$3"
output_dir="$4"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${LCRSEG_PYTHON:-/root/.venvs/lcrseg-py310/bin/python}"
evidence_dir="${GATE0_EVIDENCE_DIR:-$repo_root/docs/di_dmpa_jascl}"
case "$variant" in
  C0) config="configs/gate0_repaired_v2/fundus_lambda_u0.yaml" ;;
  B0) config="configs/gate0_repaired_v2/fundus_pas_probmse.yaml" ;;
  *) echo "invalid variant" >&2; exit 2 ;;
esac
case "$seed" in 0|1|2) ;; *) echo "invalid seed" >&2; exit 2 ;; esac
if [ -f "$output_dir/train.jsonl" ] || [ -f "$output_dir/last.pt" ] || [ -f "$output_dir/.complete" ]; then
  echo "refusing to overwrite existing training artifacts" >&2
  exit 2
fi
cd "$repo_root"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="$gpu"
"$python_bin" scripts/compile_gate0_reports.py --preflight --output-dir "$evidence_dir"
if [ "$seed" != 0 ]; then
  "$python_bin" scripts/compile_gate0_reports.py --seeds 0 --output-dir "$evidence_dir"
fi
mkdir -p "$output_dir"
set +e
"$python_bin" scripts/run_gate0_repaired.py --config "$config" --seed "$seed" --output-dir "$output_dir" --device cuda
status="$?"
set -e
printf '%s\n' "$status" > "$output_dir/.exit"
exit "$status"
