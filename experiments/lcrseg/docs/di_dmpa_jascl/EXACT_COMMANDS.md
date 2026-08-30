# Gate 0 v2 exact commands

Only branch `codex/gate0-pas-probability-mse` is authorized; do not merge main.
The frozen v1 source is `46e892960240543c946c570a9378d409b226384b`.
Archived v1 commands remain in `gate0_results_v1_zero_u_grad/EXACT_COMMANDS.md`.
All reported v2 training and prerequisite tests used source commit
`fb55e8022bc379e2515a46214c6fdf45ea818de6`. The later delivery commit adds only
reports and documentation-side aggregation/launch helpers. The remote
training checkout stays at the audited source commit; do not change its HEAD
while a run or resume is active. For reproducing the recorded source, use a
separate checkout at that commit and copy the documentation-only helpers from
the delivery branch as needed.

## Environment and official reference

Use the existing Python 3.10 environment; do not reinstall Torch.
The dedicated remote work copy is `/root/SSL_CL_gate0_v2/experiments/lcrseg`.
The official reference is a separate checkout at `third_party/JASCL_REFERENCE`
pinned to `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`.
Local Git bundles may be used to transfer these exact commits; the official
reference origin remains `https://github.com/prinshul/JASCL.git`.

## Required tests and read-only preflight audits

```bash
cd /root/SSL_CL_gate0_v2/experiments/lcrseg
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
/root/.venvs/lcrseg-py310/bin/python scripts/run_gate0_tests.py \
  --output-dir /root/LCRSeg/runs/gate0_v2_audits
/root/.venvs/lcrseg-py310/bin/python scripts/audit_gate0_v2.py \
  --mode all --device cuda \
  --output-dir /root/LCRSeg/runs/gate0_v2_audits
/root/.venvs/lcrseg-py310/bin/python scripts/compile_gate0_reports.py \
  --preflight --output-dir /root/LCRSeg/runs/gate0_v2_audits
```

Audit output is non-overwriting. If any domain has zero joint coverage, stop:
do not lower thresholds, search for a passing batch, or launch formal training.

## Conditional seed-0 pair, then seeds 1/2

Run these only after preflight is PASS. The launcher enforces evidence gates.

```bash
export GATE0_EVIDENCE_DIR=/root/LCRSeg/runs/gate0_v2_audits
bash scripts/run_gate0_job.sh C0 0 0 /root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed0
bash scripts/run_gate0_job.sh B0 0 1 /root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0
/root/.venvs/lcrseg-py310/bin/python scripts/compile_gate0_reports.py \
  --seeds 0 --output-dir "$GATE0_EVIDENCE_DIR"
# Only if both seed-0 runs and all evidence pass:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Also require RESOURCE_UTILIZATION_AUDIT.json PASS (commands below).
bash scripts/run_gate0_job.sh C0 1 0 /root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed1
bash scripts/run_gate0_job.sh B0 1 1 /root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed1
bash scripts/run_gate0_job.sh C0 2 0 /root/LCRSeg/runs/gate0_v2_lambda0_fundus_seed2
bash scripts/run_gate0_job.sh B0 2 1 /root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed2
/root/.venvs/lcrseg-py310/bin/python scripts/compile_gate0_reports.py \
  --output-dir "$GATE0_EVIDENCE_DIR"
```

When recompiling existing evidence from a later documentation-only delivery
checkout, additionally pass
`--source-commit fb55e8022bc379e2515a46214c6fdf45ea818de6`. The compiler checks
that runtime/config/test files do not differ from that audited source.

Seed 0 ran one process per GPU with the original default CPU threading.
Following the user's explicit request to improve GPU utilization, seeds 1/2
may run as two independent processes per GPU after the seed-0 and resource
equivalence gates. Each process is limited to one CPU compute thread: the
container exposes 112 host CPUs but has only a 16-core cgroup quota. This
changes resource scheduling only, not batch size, precision, model, data,
optimizer, or RNG seeding. No v1 directory may be reused.

The actual detached four-process launch, with prerequisite checks and refusal
to reuse output directories, is reproducible with:

```bash
bash docs/di_dmpa_jascl/launch_remaining_seeds.sh
```

## Resource equivalence gate

These are separate, non-overwriting diagnostic outputs. They do not replace
the original preflight reports. The three-domain repeat uses the same fixed
checkpoints and batches, not a search for higher coverage or better scores.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /root/.venvs/lcrseg-py310/bin/python scripts/run_gate0_tests.py \
  --output-dir /root/LCRSeg/runs/gate0_v2_resource_audit_threads1
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 \
  /root/.venvs/lcrseg-py310/bin/python scripts/audit_gate0_v2.py \
  --mode gradient --device cuda \
  --output-dir /root/LCRSeg/runs/gate0_v2_resource_audit_threads1
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. \
  /root/.venvs/lcrseg-py310/bin/python docs/di_dmpa_jascl/compare_cpu_thread_evidence.py \
  --reference-dir /root/LCRSeg/runs/gate0_v2_audits \
  --candidate-dir /root/LCRSeg/runs/gate0_v2_resource_audit_threads1 \
  --output /root/LCRSeg/runs/gate0_v2_resource_audit_threads1/RESOURCE_UTILIZATION_AUDIT.json
```

## Post-hoc diagnostics

Only after all six runs complete, aggregate existing logs, validation-only
diagnostics, matrices, and checkpoint finiteness/hashes. This reads no GT.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. \
  /root/.venvs/lcrseg-py310/bin/python docs/di_dmpa_jascl/compile_v2_diagnostics.py \
  --runs-root /root/LCRSeg/runs \
  --output /root/LCRSeg/runs/gate0_v2_audits/V2_TRAINING_DIAGNOSTICS.json
```

## Resume

```bash
/root/.venvs/lcrseg-py310/bin/python scripts/run_gate0_repaired.py \
  --config configs/gate0_repaired_v2/fundus_pas_probmse.yaml --seed 0 \
  --output-dir /root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0 --device cuda \
  --resume /root/LCRSeg/runs/gate0_v2_pas_probmse_fundus_seed0/last.pt
```

Checkpoints require identical config and source commit. V1 checkpoints are
read-only diagnostic inputs, not v2 resume checkpoints.
