# Gate 0 v2 resume equivalence report

Source commit: `fb55e8022bc379e2515a46214c6fdf45ea818de6`.
Device: CUDA / RTX 3090. Model: production LCRSeg UNet2D plus official JASCL 3x3 classifier.
Tolerance: `atol=rtol=1e-6`.

| Interruption | Result | Maximum absolute state/output difference |
|---|---|---:|
| Mid-supervised | PASS | 0.0 |
| Mid-unlabeled/PAS | PASS | 0.0 |
| Before loading stage-best checkpoint | PASS | 0.0 |
| After loading stage-best / before next-domain initialization | PASS | 0.0 |

Each trajectory compares student, EMA teacher, optimizer, scheduler, GAS,
prototype, Python/NumPy/CPU/CUDA RNG, sampler, stage state, matrices, best
metric, and deterministic evaluation logits. Exact reference/candidate
checkpoint paths and SHA-256 values are in `RESUME_EQUIVALENCE_REPORT.json`.

Scope: production model and stage machine on synthetic hashed HDF5 fixtures.
These tests cover the control-flow paths omitted by v1's six-step smoke test;
they do not assert that a separate full-data 100-epoch interruption was run.
Real-data gradient and formal C0/B0 evidence are separate gates.
