# Frozen numerical contract

Diagnostic-only capture uses actual LowRankConv.delta() FP32 D and LowRankConv.effective() FP32 W_eff on the training backend. Only their diagnostic copies are converted to CPU FP64. Delta_add = W_eff.double() - W0.double(); E_add = Delta_add - D.double(). No ideal FP64 D replaces production D.

Parameter gates: ||D Q||/(||D||+1e-12) <=1e-5 for RAND; ||D V||/(||D||+1e-12) <=1e-5 for SRC_A/AB; ||U.T D||/(||D||+1e-12) <=1e-5 for AB. F_FULL/F_CONV/LR_FREE constraints are N/A. Exact zero D is ZERO_UPDATE; nonzero D quantized out of Delta_add is still nonzero. A and B remain trainable.

The independent elementwise addition budget, before formal recovery, is
b = eps32*(abs(W0)+abs(D)) + tiny32 + 8*eps64*(abs(W_eff)+abs(W0)+abs(D)).
eps32 and eps64 are full machine epsilons. Elementwise diagnostic slack is 8*eps64*(abs(Delta_add)+abs(D)+b). Require abs(E_add) <= b + slack. Report beta_add=||b||, ||E_add|| and minimum element margin. Neither b nor its constants are derived from measured E or validation scores.

For FP64 evaluation define gamma(n)=n*eps64/(1-n*eps64). Frobenius norms use torch.linalg.vector_norm; reduction margin gamma(2*numel+2)*norm is explicit. Basis spectral norms use the conservative upper bound ||P||F*(1+gamma(2*numel(P)+2)), not an unverified assumption of one. Runtime bases must exactly match the reference FP64-to-FP32 cast; record reference/runtime orthogonality, cast error and projector-error upper bound ||P32-P64||F*(||P32||F+||P64||F). Reference tensors are hash-checked by the original basis loader.

For each actual left/right product A@B, beta_dot=||gamma(inner_dimension)*(abs(A)@abs(B))||F. Delta subtraction bound is eps64*(abs(W_eff)+abs(W0)). beta_eval64 is beta_dot + basis_upper*||subtraction_bound|| + gamma(2*numel(product)+2)*||product|| + basis_upper*(gamma(2*numel(b)+2)*||b|| + ||element_slack||). The effective residual limit is 1e-5*(||D||+1e-12) + basis_upper*beta_add + beta_eval64. This is an intentionally conservative numerical engineering envelope, not an end-to-end formal machine proof. Independent exact FP64 sum/cast and scalar math.fsum dot products plus projection/source/basis/perturbation/nonfinite negatives qualify it. The FP32 addition budget is not a claim about TF32/FP16/BF16 matrix-product error.

Every audit records all fourteen rows, dtype/backend, finite/source/basis status, parameter residuals, legacy relative residuals, D and Delta norms, source-relative norms, quantized-away fraction and every applicable gate. Legacy values over1e-5 with all applicable gates passing are ROUNDOFF_LIMITED_REPRESENTATION, not zero leakage. Wrong projection, source or bases, nonfinite values and out-of-budget effective updates remain fatal.

At diagnostic epochs, optimizer and EMA finish and the log commits first. Atomic latest.pt is POST_UPDATE_PENDING_DIAGNOSTICS and includes exact model/buffers, EMA, Adam, LR position, RNG, data/order hash, operation counters and provenance. All layer evidence is persisted before a failure is raised; PASS is bound to checkpoint SHA256. Resume rechecks a pending checkpoint before any next update. Atomic interruption leaves the prior complete checkpoint intact. New399/new420 retain independent hardlinks; original unsaved420 is never represented as recovered.

Caller ambient RNG states are archived and equality reported. Keyed training streams use the frozen fork_rng/seed scheme and restore unseeded caller state, so unrelated cold-process ambient RNG bytes are not asserted equal. Complete states, all399 step records/losses, data positions and keyed stream identities must match; mismatches stop the rerun.
