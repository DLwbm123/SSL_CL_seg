# Official JASCL GAS source audit

**Status:** `JASCL_OFFICIAL_SOURCE_AUDITED`  
**Repository:** `https://github.com/prinshul/JASCL.git`  
**Commit:** `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`

The repository was cloned read-only at depth 1 under
`third_party/JASCL_REFERENCE`. No JASCL training framework code was copied into
the LCR-Seg implementation.

## Active medical GAS path

- Classifier: `StochasticClassifier` / `ProbabilisticClassifier`.
- Weight: `mu.weight`, a bias-free `1x1x1` Conv3d classifier.
- Both input and classifier weight are channel-normalized; temperature is 10.
- `grad_update` is declared as an `nn.Parameter` but read and assigned through
  `.data`.
- After the ordinary backward and before `optimizer.step`, the trainer stores
  the squared classifier-weight gradient. That state is consumed by the next
  forward, not the same clean forward.
- The active scale is inverse sensitivity followed by the documented
  inverse-minmax transform. Gaussian noise is sampled with `randn_like`; no
  explicit variance multiplier is applied in the audited medical path.
- Classifier bias is disabled; perturbation scope is the classifier weight.
- The classifier has a `stochastic` argument, but the parent model invokes it
  without overriding the default `True`. Consequently `model.eval()` alone does
  not make the audited medical forward deterministic.
- A `sigma` Conv3d is declared but unused by the active GAS forward.

## Natural-image path

`LinearDecoder` instead stores squared head-weight gradients in a registered
buffer via a backward hook, uses a `1e-5` perturbation coefficient, and samples
noise unconditionally in its audited forward. Thus the official repository is
not behaviorally uniform across medical and natural branches.

## Paper/README/source comparison

The README states the qualitative GAS intent, but does not give an executable
update schedule. The actual medical and natural sources differ in state type,
noise magnitude, timing, and evaluation control. SR-GAS V0.1 therefore follows
the frozen experiment-plan contract, not an inferred merge of these variants:
same-step clean sensitivity, variance 0.1, explicit RNG, detached scale/noise,
no in-place master-weight mutation, and deterministic evaluation.

The repository root has no project-level license file at this commit. The only
located license is `Detection/LICENSE`, which is component-specific; this is
recorded in `third_party/JASCL_REFERENCE/LICENSE.txt` without attributing it to
the whole repository.
