# Gate1C v2.1: single-pair decomposition investigation

This prospective engineering investigation is limited to the first failed
registered pair, **B0/seed0/stage2/Drishti_GS/pair01**, under the current-method
long-running authorization. It is not a formal Gate1C retry or a threshold repair.
The companion JSON fixes all cases, seeds, checkpoints, counts and boundaries.

## Preserved failure

The original code `44a25254697fa535d2b48b64e27ecb226436f7d0` attempt exited 1
at 2026-08-30T16:06:46.458937+00:00. Its status remains
`BLOCKED_INCOMPLETE_EVIDENCE`: 9/9 validation units, 25/72 complete draw0 pairs,
no noise/posterior/PoE phase completion and no scientific C1-C8 verdict.
The first worker raised the original class-gradient decomposition tolerance
check; the other worker stopped at the next unit boundary. Both GPUs are idle.

A read-only postfailure audit rehashed all **648 formal files / 5,243,626,652
bytes** and all nine original B0 checkpoints. All 35 recorded model guards are
bitwise unchanged; 34 extractions completed and the failed pair did not. Public
metadata copies are in `gate1c_v21_failure/44a2525_attempt1/`; raw arrays remain
remote-only. Formal manifest SHA256:
`0d652551711e0a3ceff6ac8bdb0001355f4ec6083882460d740784ee837420d9`.

## Fixed investigation, before any new real forward

1. Publish this registration, then add a minimal inspector and synthetic tests;
   publish and verify its exact code before real probes.
2. Replay the **same failed pair once on GPU0 and once on GPU1**, sharing the
   original float32 engine, data, draw0 and student seeds. Preserve strict
   determinism, AMP off, TF32 off and original atol=1e-6/rtol=1e-4. No new pair,
   alternative draw, checkpoint, target, loss or tolerance is tried.
3. Let the original guard raise. Observe its traceback locals afterward, without
   changing any engine function, so candidate/normalization/block, unrounded
   residuals and parameter coordinates become inspectable. Do not swallow the
   error as a scientific PASS or silently resume the formal engine.
4. Only if that error is captured, inspect the same loss on detached probability
   leaves in float32 and float64. Use the original objective and class masks;
   compare each total with the three class components, with no new model forward
   or GT read. These checks diagnose where residuals enter; they do not replace
   the failed native parameter-gradient audit.
5. Verify model/bank/checkpoint immutability and publish both device outcomes.
   A non-reproduced or different failure is retained, not replaced with a retry.

The existing seeder explicitly disables both TF32 flags. A fresh interpreter's
cuDNN default is therefore not evidence of the actual diagnostic math mode.
Floating-point operations can disagree despite an algebraic identity, but that
is only a hypothesis here, not an established explanation for this failure.
[PyTorch 2.2.1 numerical-accuracy notes](https://raw.githubusercontent.com/pytorch/pytorch/v2.2.1/docs/source/notes/numerical_accuracy.rst).

No optimizer, backward, parameter.grad write, EMA/GAS/prototype update, hidden or
test GT, method registration, full retry or main merge is permitted by this
registration. Gate1B/original overall Gate1 remain transport FAIL; original
Gate1C v2 and v2.1 remain incomplete. A subsequent numerical/precision repair
requires a newly recorded finite plan after this evidence is reviewed; this
investigation preselects no such repair.
