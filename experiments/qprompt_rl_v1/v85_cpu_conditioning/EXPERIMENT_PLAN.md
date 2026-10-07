# V8.5 CPU target-sharpness and input-conditioning attribution

V8.3 and V8.3-R have been delivered with their original STOPs. V8.4 is not entered. This independent adaptive follow-up addresses two observed gaps: registered soft preferences themselves do not beat time-fixed expected reward, and the learned raw-input actors barely vary with context. Neither observation proves a cause of the endpoint failure.

Use a fixed 2×2 comparison: raw or training-fold standardized inputs, and target softmax temperature 1 or 0.25. Reuse the exact raw/T1 V8.3-R fits; train the other three arms for the same 64 Adam updates, with the same network, entropy, gradient clip, two seeds and context-held-out folds. This is a prespecified attribution experiment, not a hyperparameter search. Target temperature is separate from unchanged T=1 categorical deployment.

The combined standardized/T0.25 arm is primary; both seeds must beat all fold-trained fixed controls, and its two-seed mean gain over reused raw/T1 must be at least 0.0005 expected raw reward. Report the two component effects and interaction regardless of outcome. No better secondary arm can silently replace the primary. All new fits and one synthetic check per new arm cost at most 3,459 CPU actor updates, zero student updates and zero query-image accesses.

No new student training is authorized by this protocol itself. A passing result only triggers a separately frozen development confirmation within the already authorized D1 scope. This does not change any V8.3-R/V8.4 threshold, unlock hidden labels or real subsequent domains, or establish independent generalization. All results, including failures, will be published.
