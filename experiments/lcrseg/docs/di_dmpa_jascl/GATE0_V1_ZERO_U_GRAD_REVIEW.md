# Gate 0 v1: zero-unlabeled-gradient review

Frozen source: `46e892960240543c946c570a9378d409b226384b`.
Review date: 2026-08-30. The earlier overall PASS is withdrawn.

- Engineering checks completed.
- Leakage isolation passed.
- The tested six-step resume trajectory passed; it did not cover PAS or stage boundaries.
- The unlabeled objective gradient is zero: both unlabeled forwards and hard
  class-index MSE are outside autograd, and the consistency value is detached.
- These results are invalid for a semi-supervised JASCL claim.
- They may only be used as an inert-unlabeled / labeled-cycle control.
- Unlabeled forwards still consume classifier RNG. This is **not** simply a
  pure supervised baseline, nor the new compute/RNG-matched v2 C0 control.
- The old TinySegNet comparison is a deterministic supervised smoke test,
  not method off-switch parity. No DI-DMPA method exists.

`GATE0_STATUS_V1_ARCHIVED.json` preserves the original bytes, including its
historical PASS. `gate0_results_v1_zero_u_grad/` preserves the original reports,
protocol, and matrices; their historical labels do not override this review.
The original remote run directories and checkpoints are not changed.

The latest status remains blocked until all revised semantic, gradient,
deterministic-evaluation, resume, leakage, and C0/B0 three-seed gates pass.
