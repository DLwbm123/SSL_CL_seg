# Gate1C v2.1: prospective same-pair float64 numerical reference

Registered 2026-08-30T16:33:56.274Z. Machine-readable contract:
[GATE1C_V21_FP64_REFERENCE_PREREGISTRATION.json](GATE1C_V21_FP64_REFERENCE_PREREGISTRATION.json).

This is a finite engineering reference under the user's current-method-only
long-running authorization. It is **not** a full Gate1C retry, method training,
an amended admission threshold, or a revision of either historical failure.

## Evidence motivating the reference

The published investigation/report commits are
`26121042817c0b7fc586ad44f869e86faca2c59b` /
`81afacd4ab5492059813a699e28a886af21520b3`.
The original float32 R2/class-balanced component-sum guard fails on the same
three of 483,152 coordinates on both GPUs, maximum error
3.1642557587474585e-06. Both detached probability-leaf gradient sums are exact.
All original guards, targets and weights remain unchanged.

Hypothesis: the discrepancy is introduced by finite-precision network VJP
accumulation. A float64 network reference with the **same random realization**
can test this explanation; merely reseeding a float64 random generator cannot.

## Finite protocol, fixed before implementation

1. Use exactly the failed registered pair
   `B0/seed0/stage2/Drishti_GS/pair01` once on each existing GPU. No new
   pairs, cases, seeds, candidate search, teacher draws or ground truth.
2. Observe the original three native float32 forwards (student unlabeled,
   teacher draw0, student labeled) and let the original guard raise. Require
   native probability/target/weight/prediction and all gradient hashes to match
   the prior device receipt. A temporary observer calls the original
   `torch.randn_like` unchanged and retains the three standard-Gaussian
   tensors, each float32 [3,16,3,3]; it does not modify returned values.
3. Make a deep, isolated float64 copy of the same frozen student. Perform one
   unlabeled forward using the official UNet and stochastic 3x3 classifier,
   replaying the captured **student-unlabeled** standard-Gaussian tensor cast
   to float64. No new draw, no forward-value override, and no original model
   precision change.
4. Keep the original failed teacher probability, R2 weights and teacher class
   strata detached and value-identical. Use the unchanged R2/class-balanced
   objective with float64 student probabilities. Compute total and each of
   three component gradients with four independent `autograd.grad` calls.
   Never define the total as the sum, correct residuals, drop coordinates or
   write `.grad`.
5. Report the unchanged `atol=1e-6, rtol=1e-4` componentwise predicate and
   unrounded residual statistics for all seven blocks. Report all native/
   reference forward, total-gradient and class-gradient differences, not only
   the failing coordinates.
6. Before and after the reference, run the original isolation helper on the
   exception's captured `probe_unit` state; independently compare legacy/
   current/history hashes, original and shadow model state groups, inactive
   gradients and all checkpoint/failure-manifest hashes. This adds evidence
   for this new probe; it does not fill the omitted receipt in the old run.

Maximum: **8 model forwards total** (3 native + 1 reference per GPU), four
float64 VJPs per GPU, at most 10 minutes per replica, two parallel workers,
zero optimizer updates, no new dependency. Create-only output root:
`/root/LCRSeg/runs/gate1c_fp64_reference/<preregistration-commit>/replica_gpu{0,1}`.
Implementation and synthetic tests must be committed/pushed and remotely
verified before either real replica runs.

## Interpretation fixed in advance

Numerical-reference support requires both devices to reproduce the native
failure and pass provenance/isolation, plus float64 decomposition passing the
**unchanged** componentwise predicate in all seven blocks and global relative
L2 residual <=1e-10. To interpret it as comparable to the native gradient,
native/reference global relative L2 difference must be <=1e-3 and cosine
>=0.9999 on both devices. These are new reference-interpretation bounds, not
relaxed Gate1C tolerances.
Relative difference is `norm(native-reference)/norm(reference)`; zero reference
norm is undefined and cannot satisfy comparability.

If decomposition passes but comparability does not, report
**HIGH_PRECISION_DECOMPOSITION_ONLY_NONCOMPARABLE**. If decomposition does not
pass, report **FP64_REFERENCE_NOT_SUPPORTING_HYPOTHESIS** and stop this finite
probe. Report both replicas and every discrepancy; no device selection or
automatic backend/precision search.

Even a supported reference does not rescue the old attempt, establish all
72 pairs, approve changed full-run precision, or constitute scientific
admission. Formal Gate1C v2.1 remains **BLOCKED_INCOMPLETE_EVIDENCE**; Gate1B and
overall Gate1 remain **FAIL_TRANSPORT_NOT_SUPPORTED**. Method registration and
new method/transport training remain off. Prior baseline recovery's 400
updates stay separately disclosed.

Analyze and publish both completed outcomes before any separate prospective
full-diagnostic or current-method implementation plan.
