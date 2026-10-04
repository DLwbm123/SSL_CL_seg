# Implementation delta

Execution source: `77cf65db6802ebbeaf2cd6b7c097fc11f0ebc2cc`.
Baseline: `90eacbacca00d3c03e1296d5c7e6ac22a047df3b`.

## Reused unchanged

V4 native source trainer, stage Trainer/Provider, student optimizer and right projection, 40 feature equations, U loss, EMA, native supervised zero-U path, source receipts, full-horizon exporter and val evaluator core. No edits to historical engines or frozen data.

## Added

- Complete fork bookkeeping around the native snapshot: options, physical update count, teacher modes, Provider identity/roles and checked sets.
- Pure-softmax policy with reward-free last-layer softening and frozen historical normalization; no critic module/optimizer for GRPO.
- Four complete same-entry trajectories before any actor update; fixed per-domain calibration scale and verified complete-state U0 cache.
- PPO/instantaneous Bandit and group-relative STD/FS/reward-shuffle objectives; separate actor/critic gradients and Adam; 16 actor calls/group.
- Independent reward-permutation and minibatch RNG; action sampling independent of the native environment RNG.
- Global endpoint-freeze barriers before val; preregistered pilot, independent-controller confirmation and conditional Clip-Higher gates.
- Per-update policy diagnostics, private full state panels, separate evaluator label/GPU receipts and physical optimizer ledger.

## Setup correction, without student replay

The first setup passed 12 synthetic learning checks and legacy math tests, then failed before its first native optimizer call because the CuBLAS deterministic environment was absent. The failed prefix is retained. Its 65,884 synthetic optimizer calls are charged once and its successful evidence is reused. The corrected launch supplies `CUBLAS_WORKSPACE_CONFIG=:4096:8`; 690 native qualification calls remain the original allocation. The RNG correction adds a declared 16-call CPU-only check. No performance feedback was available or used.
