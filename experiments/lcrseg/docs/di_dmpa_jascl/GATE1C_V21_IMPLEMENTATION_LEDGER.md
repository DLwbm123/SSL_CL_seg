# Gate1C v2.1 implementation and pre-publication checks

Preregistration: `9d8ecc65730bee5bec46a1f098c9fe96a67a59b9`.
Only the diagnosed missing legacy PAS input is amended. The original frozen
v2 code checkout, its incomplete attempt, and all B0 checkpoints remain intact.

## Minimal shared-engine changes

- `binding.py`: explicit `v2`/`v2.1` input contract; verify the published amendment,
  its authority and reconstruction proof. Default v2 still rejects `None`.
  Exactly `B0/seed1/stage1` can load the one hash-bound reconstructed tensor;
  all other banks still come from their original checkpoint payloads. There is
  no refit, checkpoint edit, second override or unconditional fallback.
- `runner.py`: CPU-check all nine required payloads before any real diagnostic
  forward. Propagate the input version to both workers and test/evidence gates;
  require a separate v21 output root. Recheck the external bank hash during
  immutability audits. Reuse the existing phase machine, workers and evaluator.
- `reporting.py`: versioned metadata, scope and report captions, with explicit
  V21 status/report aliases while retaining the existing internal file schema.
  Carry the missing historical-bank-hash limitation and prior 400 baseline
  recovery updates; new diagnostic optimizer updates remain zero.
- Exact publication is verified through GitHub's reachable HTTPS branch-ref
  API; no global network setting or remote origin is changed.

Relative to `01b3fd0`, there are **no changes** to Gate0 training/configs,
`gradients.py`, `reliability.py`, `metrics.py` or `execution.py`.
All nine original models, other eight legacy banks, K2 prototype banks, source
selection, case/pixel/pair/draw identities, formulas and C1-C8 thresholds remain
unchanged. The new binding test compares every other parent-contract field.

## Synthetic verification

Pre-publication v1: 114 passed, 1 failed, 13.65 s. The added v2.1 report test
found that the old decision object overwrote the authorized long-running
`next_action`. The repair preserves the versioned workflow action after merging
the unchanged scientific decision; it does not change any gate value.
The failed JUnit and synthetic artifacts are retained:
`/root/gate1c_v21_synthetic_v1.xml`, SHA256
`2d68de5c04f7b6953e22c4c1c3edf3c744c622bcc988de45bb960344adc74bca`.

Pre-publication v2: **116 passed**, 13.73 s; no real checkpoint/bank/GT tensor
read or diagnostic forward. Existing synthetic tests are retained, plus checks
for the sole input override, all-nine readiness, original-v2 rejection,
nonfinite/dtype/shape/gradient rejection, immutable proof and parent semantics,
mixed-version metadata rejection and versioned full-report compilation.
JUnit: `/root/gate1c_v21_synthetic_v2.xml`, SHA256
`622c1c4cb72a335923ddfbce04d8e6ac4b932f1abd6dc2e680f48240867e72b9`.

The tests ran on the existing Python with one CPU thread and no visible CUDA
device. They used `test_core.py`, `test_pipeline.py` and
`test_reconstructed_input.py` under `tests/di_dmpa_gate1c_v2`.

After exact-code publication, the complete suite must also pass the original
real integration plus the preregistered affected validation case and gradient
pair, both under strict no-update/model/bank guards. No v2.1 real diagnostic or
formal cache generation has run at this pre-publication milestone.
