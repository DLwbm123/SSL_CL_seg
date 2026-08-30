# Gate 1 prework: baseline freeze, compiler hardening, audit parity

Date: 2026-08-30. This is not a mechanism-admission result.
Branch: `codex/di-dmpa-gate1-diagnostics`, created from
`ea945382030e8eb2be070fa3d2ee20e5128f791d`, not from main.

## Completed independent work

- `BASELINE_FREEZE.json`: six original run/completion/log hashes, 24 freshly
  rehashed best/final checkpoints, C0/B0 config hashes, original report hashes,
  original source/report/upstream commits. No checkpoint was changed.
- Hardened `scripts/compile_gate0_reports.py`: actual per-batch zero coverage
  and zero raw consistency-gradient counts, per-domain minimum gradient,
  linearly interpolated p01 gradient, and minimum coverage. A single zero
  batch cannot hide behind an aggregate count or a positive maximum.
- Recompiled all six original logs with the freeze-verified historical mode.
  Result: PASS, errors=[], zero coverage and zero gradient counts both 0 for
  each C0/B0 run. Report is in `gate1_prework/compiler_hardened/`.
  This operation did not train anything or read final-test GT.
- Original `GATE0_STATUS.json`, `GATE0_V2_FINAL_REPORT.md`, `gate0_results_v2`,
  frozen configurations/protocol, and old LCR-Seg source remain unchanged.
- Added future-only `configs/diagnostics/future_gradient_audit.yaml` and a
  scheduling policy. Existing Gate 0 configurations retain every-batch audit
  by default. Disabled audit records a null gradient norm and an explicit
  false audit flag, not a fabricated successful gradient measurement.
- On/off exact-parity: actual UNet2D/JASCL 3x3, CUDA, synthetic hashed 16x16
  HDF5 inputs, 55 steps per path, positive PAS coverage and consistency loss.
  Student, teacher, optimizer, scheduler, GAS, prototypes, RNG, stage/sampler,
  best metric, matrices, deterministic logits and entire loss trajectory
  matched with maximum difference 0. Audit calls were 2 versus 0.
- Final focused suite: 28 passed, 0 failed/skipped, 6 preserved warnings,
  6.88 seconds. Source: `361e9ff1ed539e825071d86002fa37bdf0d9b875`.
  Exact parity report and pytest/JUnit are in `gate1_prework/attempt2/`.

The parity unit test necessarily updates two freshly initialized synthetic
model copies: 55 fixture optimizer steps per path, 110 total. It never loads
any frozen baseline checkpoint. Gate 1 diagnostic optimizer steps on the
frozen segmentation models remain 0, and transport fitting has not started.
Do not conflate test-fixture optimizer operations with authorized mechanism
diagnostics or claim that no optimizer ran anywhere in the test process.

## Failures and preserved upstream behavior

The first focused run failed before its state comparison: the test expected
step 55 to remain in PAS, while steps 53/54 are PAS and step 55 returns to
supervised. Inspection also found unmatched starting states caused by the
official classifier module's cold import resetting global RNG to 1024.
The constructor also reseeds Torch. First-step loss differed before either
path invoked the gradient audit, so that comparison did not isolate audit
overhead. The old transcript/JUnit remain in `gate1_prework/attempt1/`.

The corrected test uses one explicit initial model/optimizer/RNG checkpoint
for both synthetic paths and verifies nonzero PAS coverage/loss. No gate
threshold, formal baseline, or upstream source was changed. The cold-import
RNG side effect is an upstream limitation to record in later interpretation;
this task does not silently repair it or rerun accepted Gate 0 baselines.

Local system Python lacked pytest; the bundled document Python lacked Torch.
No packages were installed. All executed tests used the existing remote
Python 3.10.21 / Torch 2.2.1+cu121 environment. Preserved warnings concern
scheduler call order, scheduler epoch deprecation and CUDA NLL warn-only
determinism; no tolerance was relaxed.

## Protocol decision still required before preregistration

Gate 1A specifies 18 foreground class-domain-seed units but requests four
panels: C0/B0 crossed with student/EMA. The main admission panel and handling
of controls are not explicit. This affects selected feature source/K and
subsequent transport/reliability admission.

Proposed interpretation awaiting confirmation: B0-EMA is the primary panel;
C0 and student are separate controls, each reported on its own 18 units.
Do not pool 72 units or pick the most favorable panel after seeing results.
The alternative of requiring multiple panels to pass would define a different
gate and must be fixed before the first mechanism diagnostic.

No official preregistration has yet been committed. No geometry, transport,
reliability, gradient-conflict or theory diagnostic has run. No model is
registered and no Gate 2/pilot/full sweep is authorized.

## Exact commands used for completed prework

Remote checkout: `/root/SSL_CL_gate1/experiments/lcrseg`. The official ignored
reference is a symlink to the unchanged pinned checkout in the Gate 0 work
copy; data/outputs remain separate. Transfer used Git bundles; old Gate 0
work copy and runs were not checked out to a new revision.

```bash
cd /root/SSL_CL_gate1/experiments/lcrseg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=.
# Historical recompilation executed at 036fbe773dd88e3a6c60810628385c6b84b3892e:
/root/.venvs/lcrseg-py310/bin/python scripts/compile_gate0_reports.py \
  --frozen-baseline docs/di_dmpa_jascl/BASELINE_FREEZE.json \
  --evidence-dir docs/di_dmpa_jascl \
  --output-dir /root/LCRSeg/runs/gate1_gate0_recompiled
# Final focused tests executed at 361e9ff1ed539e825071d86002fa37bdf0d9b875:
env CUDA_VISIBLE_DEVICES=0 GATE1_PARITY_DEVICE=cuda \
  GATE1_PARITY_REPORT=/root/LCRSeg/runs/gate1_prework_audit_v2/AUDIT_ON_OFF_PARITY_REPORT.json \
  /root/.venvs/lcrseg-py310/bin/python -m pytest -q \
  tests/di_dmpa_gate1 tests/gate0/test_report_compiler.py \
  --junitxml /root/LCRSeg/runs/gate1_prework_audit_v2/pytest.xml \
  --basetemp /root/LCRSeg/runs/gate1_prework_audit_v2/pytest_artifacts
```

Use a new output namespace for a repeat; do not overwrite either attempt.
