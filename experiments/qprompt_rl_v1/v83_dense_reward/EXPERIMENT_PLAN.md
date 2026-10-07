# V8.3 independent complete-reward supervision diagnostic

Authority: the user requested continued evidence-based improvements after weak experiments and approved image-level exploration. V8, V8.1 and V8.2 stay failed; this is an independently frozen question with smaller cost, not a retry past their stops.

## Evidence and question

Both GRPO adjustments failed: paired noise calibration reduced skipped groups but not utility; matching native 300-step episodes also failed. Policies did update and the qualified memory gate was active. Test whether the same 24→32→9 policy can learn useful selection from complete nine-action reward information. A positive result supports reward-supervised control only; it cannot be presented as GRPO efficacy.

## One intervention and fixed teacher information

Reuse the immutable eight V8.2 H300 native step-100 entries. Advance each with 100 NATIVE updates to obtain step-200 states; no Q_dev readout. At both decision steps 100/200, restore the full native entry for each of nine actions and each of the two original paired RNG streams; run exactly 100 native student updates and compute the unchanged Q_train gain/old-memory-reference penalty. Discard all candidate student states, never retain the best. This is 8×2×9×2=288 reward rows, 28,800 student branch updates. Keep the exact native optimizer, loss, teacher, roles, photo conditions and memory source.

For each of the 16 feature states, average the two reward streams for each action. Center and divide the nine rewards by max(population std, frozen paired floor 0.0007379373167760999), clamp logits to [-3,3], then softmax at temperature 1. These are fixed soft supervision targets; reward values are not policy input features. Train two zero-output actors 601/602 on the entire 16-row table for exactly 64 full-batch Adam .001 steps each, entropy coefficient .01, gradient clipping 1. Preserve both final checkpoints. No intermediate query evaluation or actor checkpoint selection; this is supervised fitting, not GRPO.

## Matched controls and gates

Give the fixed controller the exact same complete teacher information: choose one global action by mean reward over all 16 states, tie-breaking native then lower alpha/action ID, before development training. This avoids comparing a densely supervised actor against a fixed action selected from a different teacher budget. Native and uniform controls are unchanged. Complete and seal all 20 native/fixed/uniform/two-prior development endpoints before reading Q_dev. Development has been seen in earlier diagnostics; do not call it independent validation or patient generalization.

At least one fixed actor seed must exceed all three controls and its matching V8.2 actor by mean utility 0.0005, with new Dice within -0.0025 and old Dice within -0.005 of uniform. Otherwise seal STOP_DENSE_REWARD_NO_PRACTICAL_GAIN and retire this dense-supervision intervention. No target or low-label training follows automatically; positive diagnostics require separately frozen target qualification and evaluation. Do not expand seeds or change preferences, reward, conditions or gates after outcomes.

## Checks, cost and storage

Reuse the native paired-branch/full-state/query isolation qualification under H300 (eight student updates), and add one synthetic complete-table objective update that must lower its loss with finite gradients. Qualification actor cap nine covers native qualification plus this check. The stdlib check verifies reward translation invariance, flat preferences, all 288 required pairs, feature progress at 100/200, and negative gate behavior.

New student cap 34,008: qualification 8, decision entries 800, teacher branches 28,800, development entries 400, development training 4,000. New actor cap 137: qualification nine plus 128 fit steps. Root accounting carries the complete 190,845 student/704 actor prefix once; historical source 8,000 student remains separately disclosed. Source and original action audit are not repeated. GPU 5/6/7 and create-only NAS output with native wrapper and neutral commands only. Publish complete anonymous teacher rows, preferences, fit logs, endpoints and physical costs; keep images, roles and weights private.
