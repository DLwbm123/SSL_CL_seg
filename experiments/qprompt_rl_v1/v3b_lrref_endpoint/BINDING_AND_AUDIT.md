# V3B-LRREF-ENDPOINT binding

User designation on 2026-09-30 explicitly selects audited LR_SRC_A / NativeLRParent.
Original KI historical identity is unverified. No QUERY substitution; no claim of original KI recovery.
Previous blocked audit remains unchanged in the dated Downloads delivery.

## Native reference

- Implementation: `experiments/lcrseg/five_frameworks_v1/native_parent.py`, originally introduced in `afcbfd9`; current source inherited from `c2d88ce` without edits.
- Historical execution: `cc21871a43e3cd105cddaf831f5c3ea2fef59b9e`, `native_lr_src_a_3domain_v1_20260914_01`.
- Historical P1 configuration selected in existing SELECT_PARENT and SELECT_B0_PARENT_LCTX receipts. No new selection by validation performance.
- Single LCRSegUNet2DJASCL, 14 projected convolutions, rank 8; only A/B trainable. BODY includes every trainable parameter. HEAD is frozen, so its feature block and validity flags are zero.
- Frozen trained weights plus `B @ (A - (A @ V) @ V.T)`; V is reconstructed by the original stage-entry SVD code. Projection is inside the forward parameterization. `apply_constraints` checks the residual, not an extra optimizer transform.
- Native Adam, coupled weight decay 4e-5, LR .001 * P1 multiplier .5, A/B ratio 1; native polynomial exponent .9, horizons 3200/2100. No timeline rescaling.
- Existing 8000-step REFUGE source checkpoints for seeds 161/162/163 are validated against their source identities and tensor fingerprints. Both target domains enter at stage 1 of their respective registered orders. New zero-update target entries reuse this legal flow. No full CL-trajectory claim.

## Bridge and controls

Native original is L-only, preserved as ORIGINAL. Other five arms use a registered bridge: unchanged native nonU objective and schedule, plus U consistency from relative step zero, coefficient .5, action-dependent confidence > .7 and valid-geometry denominator, dense effective-weight EMA .99. U uses the native collected LCTX forward and current-L donor. This added U rule is not claimed identical to the original recipe.

Action SKIP removes only U; the native nonU objective still updates. FINE and COARSE mathematics, BASE16 and EFFECT feature equations, policy/scaler/ridge code are inherited from V3A2. A local copy of the small pure-math modules removes the old QUERY executor's import-time environment dependency. Preview is implemented with `torch.optim.adam.adam`, matching actual Adam group flags and lazy states, retaining None gradients. Virtual weights are never installed in the student. Both BODY and frozen-empty HEAD remain in EFFECT40.

## Data and state

Canonical manifest and split hashes remain those asserted by the existing native loader. Frozen assets retain their existing loader hash checks. No test set. U accessor exposes images and geometry only. No old-domain training image read. Validation runs in a separate evaluator process and cannot feed policy training.

Per target domain, development seed 161 uses a fixed patient-sorted split: first two labeled patients F_online, next two F_audit, remaining L_fit (12 RIM / 6 Drishti). The exact private roles are saved before development. Pilot seeds 162/163 use all native current-L examples and the same current-U pool across methods. These are reused development patients, not independent patient generalization.

Features are extracted before feedback. Clean quality gradient uses the same L_fit source pair as that update, without augmentation. Readonly extraction restores all student buffers/modes, random streams and cursors; it does not step the optimizer or EMA. Checkpoints clone optimizer state to avoid aliasing the CPU step tensor. Main policies are frozen and predictions restore RNG, so policy initialization cannot perturb the student stream.

The check module covers Adam lazy/moment state and None versus zero. Real smoke covers feature immutability, preview versus real updates with the original A-side check, full-state repeated continuation, source/role binding and every method's full five-step cost. No efficacy gate. All physical calls including discarded smoke/branches are durable and budgeted.

## Window and lifecycle

Single new window starts 2026-09-30 10:30 UTC. Optimizer cutoff 2026-10-01 00:30 UTC; hard end 02:30 UTC. Integration, calibration, training and report share this ledger. Serial GPU 4; balanced five-step method queue per seed/domain. H is locked solely from measured full-block throughput with 25% margin, evaluation and report allowance. Source receipt reuse needs no new prefix updates. No automatic retries; errors preserve receipts and require a versioned repair within the same ledger/window. No push authorization.
