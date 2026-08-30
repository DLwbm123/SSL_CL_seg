# Gate 1C v2 deterministic supervised-reference repair

Prospective repair under the 2026-08-30 long-running authorization and its
current-method-only clarification (commit `b6d70699599bd89faafb4a1dd22223575c62bbb6`).
The frozen preregistration, all gate thresholds and all previous evidence remain
unchanged. This fixes an execution defect, not an unfavorable scientific result.

## Preserved failure

Exact code `68dedea7ccaa9144913dfc50a096364d7d55f2cf`, validation `attempt2`,
passed 94 synthetic tests. The real integration passed publication checks,
three stage-specific validation samples, the known-null sample and their exact
legacy PAS checks, then failed in `supervised_gradient` before consistency
gradient diagnostics. No formal run or optimizer update was performed.

- `pytest.xml` SHA256:
  `38221f77008a3946bfe4805d6fc163ddad6171811d54c811604c04c599c10675`.
- `pytest_output.txt` SHA256:
  `17231e8e998076a907311c47823a1418b233ad88e4a33aad0b7ab61924753ef0`.
- Attempt root:
  `/root/LCRSeg/runs/di_dmpa_gate1c_v2_validation/68dedea7ccaa9144913dfc50a096364d7d55f2cf/attempt2`.

## Root cause and minimal repair

`di_dmpa_gate1c_v2/gradients.py::supervised_gradient` directly requested the
CUDA NCHW cross-entropy mean. PyTorch 2.2.1 implements that fused NLLLoss2d
reduction using atomicAdd and rejects it under strict deterministic algorithms.
Its unreduced per-pixel forward and backward do not use that fused reduction.
See [the exact upstream implementation](https://github.com/pytorch/pytorch/blob/v2.2.1/aten/src/ATen/native/cuda/NLLLoss2d.cu).

Use the same cross entropy with `reduction='none'`, sum its per-pixel values and
divide by the count of labels not equal to 255. Ignored pixels have zero loss
and gradient; the existing all-ignored-input rejection remains. This is the
same unweighted mean CE over valid pixels, with a deterministic reduction order.
Floating-point roundoff may differ from an unguarded fused CUDA reduction; no
claim of bitwise parity to that nondeterministic kernel is made.

The shared function serves all four probe phases, so one repair covers draw0,
noise, posterior-mean and PoE. Strict determinism remains enabled. There is no
new loss, changed scaling, precision, batch, seed, label role, model parameter,
prototype or optimizer, and no Gate0/Gate1A/Gate1B source is modified.

## Required verification before a new real attempt

- CPU native mean CE value and parameter-gradient parity with no ignore,
  mixed ignore and one valid pixel; all-ignored input must fail closed.
- Bitwise repeated deterministic values and gradients; parameter.grad remains
  None; optimizer construction and backward invocation stay prohibited.
- Repeat the mixed-ignore and single-valid-pixel checks on both GPUs in the
  post-publication real integration, then execute its original fixed real cases
  and complete pair unchanged.
- Run all existing synthetic tests before publishing the exact repaired code;
  run the complete real integration only after its publication check succeeds.
- Numerical regression tolerances (value 2e-7 absolute / 1e-6 relative;
  gradient 1e-7 absolute / 1e-5 relative) apply only to independent CE
  implementation parity, never to C1-C8 or frozen admission thresholds.

All new outputs use a new exact-code namespace and immutable attempt directory.
Old attempt1 (network preflight) and attempt2 (deterministic CE) are preserved.

Pre-publication synthetic verification on the existing remote Python 3.10 /
PyTorch 2.2.1 environment: **98 passed in 8.31s**, no real checkpoint tensors
read. JUnit artifact: `/root/gate1c_v2_ce_repair_synthetic_v1.xml`. The complete
post-publication GPU integration is still required; this result does not
constitute Gate 1C admission or a formal scientific diagnostic result.
