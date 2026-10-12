# Simple B1 confidence-only KL — fixed supplementary baseline

User authorization, 2026-10-12: “好的，你补一个更简单的基线”. This adds a transparently named simple baseline; it does not change B2, its published scores, V20 or the original method success criterion. No outcome-dependent weakening or configuration selection is performed.

## Single simplification

Reuse the existing `B1_PARENT_CONF_KL` family in the common `StageTrainer`: current EMA soft-label KL, eligibility `geometry & (max(q)>0.6)`. Relative to B2_C06, remove only the PAS prototype-similarity eligibility condition. Retain B2's confidence 0.6, lambdaU 1.0, EMA, warmup/ramp 0.2, Adam/weight decay 4e-5, parent A/B lr 0.0005 and polynomial schedule. The strict confidence inequality is the existing B1 definition. No coefficient search, reduced training, adverse initialization or selected checkpoint is allowed.

Same native parent, input-side hard projection and current-L LCTX CE+Dice. Main U updates the parent's trainable A/B; there is no R/Q feature sidecar, SWD, entry teacher, reference VJP, replay or additional inference component. The shared engine still computes current prototypes/PAS and then overrides the mask for B1. Those computations do not select B1 pixels and are disclosed; this is an algorithmic eligibility simplification, not a speed claim or a fully unconstrained finetuning baseline.

## Finite matrix and permissions

Seeds163/164, REFUGE→RIM→Drishti and REFUGE→Drishti→RIM. Reuse each corresponding sealed 8000-step source, source training0. Each trajectory inherits its own stage1 student; deployment transform remains identity. RIM3200 plus Drishti2100 gives21200 formal updates in4 trajectories/8 target stages. First163/O1 supplies engineering seals; then remaining three, without efficacy pruning. All8 stages and4 EXIT0 are required before20 unified seen-domain evaluations. Only current L labels and current U images/geometry enter training. Frozen splits/manifests and private/test boundaries stay intact.

Qualification caps96CPU/32generatedCUDA/4currentL; planned32/10/2, including a deliberately failed after-optimizer invocation. Reuse the existing masked-KL definition, but check a fixed two-case fit, empty/masked/detached target behavior, archived built-in B1 native full-state/loss/update parity, golden confidence mask despite forced rejected prototype mask, nonzero actual A/B U gradients, teacher freeze/EMA, true checkpoint continuation and changed-family rejection, deployment/own-prefix stage transition, durable failure counting, current-L zero-U smoke. No performance admission criterion: the baseline is completed and reported whether stronger or weaker.

The common engine's checkpoint family field now uses the actual model family. For every F5 method this remains `F5`; the new baseline is correctly labeled `B1_PARENT_CONF_KL`. Previously frozen/running source snapshots are unchanged. No training loop is duplicated.

## Comparisons and reporting

Report all seed/order/domain/rim/cup cells, Final=(2Old+Incoming)/3, Old, Incoming and Forget. Reuse frozen B0/B2 and matched F5 aggregates, and display V13/V17 as method context. Historical versus new comparisons remain labeled; this supplementary baseline is not evidence that B2 is invalid. Original practical gates are descriptive comparisons only and never determine whether to finish, retry or hide a baseline result.

This adds at most21200 formal calls to the existing method campaign allowance, tracked separately as supplementary-baseline cost; V20's21200 cap is unchanged and no unused candidate allowance transfers. No retry, new seed, parameter sweep or test-based selection. New create-only NAS directory, original storage wrapper, neutral parent/worker argv, fresh NFS write probe and GPU memory admission. Concurrent healthy jobs are preserved.

Once completed, publish source, this fixed protocol, qualification, full anonymous positive/negative results, costs and report via the required GitHub proxy. Keep data, patient rows, weights, checkpoints, private configuration and raw logs on NAS. Reused development patients and two optimization seeds do not establish independent generalization. A simpler baseline may perform better; do not assume or force a lower score.
