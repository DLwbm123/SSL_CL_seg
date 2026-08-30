#!/usr/bin/env bash
# Resource scheduling only; run from the frozen remote experiment root.
set -euo pipefail
cd /root/SSL_CL_gate0_v2/experiments/lcrseg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=.
export GATE0_EVIDENCE_DIR=/root/LCRSeg/runs/gate0_v2_audits
/root/.venvs/lcrseg-py310/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/root/LCRSeg/runs')
for path in [root/'gate0_v2_audits/SEED0_PAIR_GATE.json',
             root/'gate0_v2_resource_audit_threads1/RESOURCE_UTILIZATION_AUDIT.json']:
    assert json.loads(path.read_text())['status'] == 'PASS', path
for seed in (1,2):
    for prefix in ('lambda0','pas_probmse'):
        assert not (root/f'gate0_v2_{prefix}_fundus_seed{seed}').exists()
PY
for seed in 1 2; do
  for variant in C0 B0; do
    if [ "$variant" = C0 ]; then gpu=0; prefix=lambda0; else gpu=1; prefix=pas_probmse; fi
    run_dir="/root/LCRSeg/runs/gate0_v2_${prefix}_fundus_seed${seed}"
    mkdir "$run_dir"
    nohup bash scripts/run_gate0_job.sh "$variant" "$seed" "$gpu" "$run_dir" \
      >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" </dev/null &
    printf '%s seed=%s gpu=%s pid=%s output=%s\n' "$variant" "$seed" "$gpu" "$!" "$run_dir"
  done
done
