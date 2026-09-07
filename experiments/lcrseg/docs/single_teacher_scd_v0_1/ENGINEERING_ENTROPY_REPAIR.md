# Engineering correction of zero-mass KL entropy

The first E stage1 attempt at source 057ce07 completed one optimizer update, then stopped on update 2 with a nonfinite loss. Its directory, step ledger, initial checkpoint, failure receipt and child exit remain unchanged. A diagnostic replay of those two fixed initial batches reproduced the failure; it made one separately counted successful reconstruction update and did not provide initialization for the repaired experiment.

The projected target can legitimately contain representational zero probabilities after FP64 exponential tilting. All inputs remained finite, the target was normalized and its constraint residual passed (maximum about 2.83e-14), but the original expression multiplied zero by log(0), producing NaN. This is a KL implementation defect, not a scientific failure or a failed capacity gate.

The correction evaluates the continuous entropy extension 0*log(0)=0 by replacing log(target) with zero only where target is exactly zero, before multiplication. It does not change the target, q smoothing, solver, iteration/bracket budget, constraint, reliable mask, reference label, normalization, threshold, lambda or update schedule. Positive-target arithmetic stays bit-identical, so S/A/B/C/D and the positive-support E terms retain their prior implementation semantics. No r=p fallback, conflict deletion or label substitution is introduced.

An independent synthetic confident-logit example reproduces target underflow without reading real cases. It verifies finite KL, agreement with xlogy, unchanged positive-target arithmetic, the p-r autograd gradient and the original solver residual contract. Complete CPU/GPU qualification and the fixed two-case stage0 smoke must pass on the new source before restarting E in a new create-only directory from the same common stage0.

The accepted fixed matrix still contains 39,800 updates. The discarded E attempt contributes one additional formal-attempt update (39,801 across all formal attempts); the diagnostic replay contributes one separately recorded engineering update. Neither is concealed inside the accepted budget or presented as zero cost. No failed stage produced val feedback and no method selection or hyperparameter tuning occurred.

The completed common/S/A/B/C/D stages remain at source 057ce07. The restarted E stages use the repaired source, with their own new stage1 as stage2 teacher. Final reporting must bind each stage to its exact implementation source and preserve both original parent outcomes and the repaired-stage receipt.
