# V109 — supervised-only action margins

Complete training diagnostic, not a deployment result or an RL success. All 41 jobs and the coordinator exited 0. The added OFF action is the existing B0 supervised native loss: no U reads or U forward passes during updates. Actions 0–11 are unchanged. Feature extraction and behavior selection still use authorized U_adapt between updates.

## Complete result

Across all 80 retained training states, OFF exceeded the best original action in 35 states and was worse in 45. Mean OFF−best-original return was **−0.001188273192**, while OFF−original-action-mean was +0.002622005262. The former decomposes into +0.000244092499 new-task dense gain and −0.001432365691 old-task retention. Closing U therefore is not universally protective.

At step 100 the mean margin was −0.000963781308 (19/40 positive); at step 200 it was −0.001412765076 (16/40 positive). Both stream means were negative. All individual rows and context summaries, including the strongest positive and negative groups, are published in NEW_RETURN_TABLE, ACTION_MARGINS and MARGIN_SUMMARY. These were not selected by sign.

The mixed margins justify testing a conditional action policy, not replacing every update with OFF. Before fitting a 13-action policy, a separately frozen next collection will cover the 40 saved OFF-first step-200 prefixes; the current 80-state table does not contain those states. This is a development-informed design decision. Actual sequential deployment must include fixed OFF, matched CE and target-matched CE controls; no criterion is relaxed.

## Accounting and validation

12,012 native optimizer updates: 12 qualification plus 12,000 collection; all attempt/success pairs and root/child keys match. 8,008 OFF and 4,004 original-action updates; 960 Q_train image evaluations in 240 paired query calls; 120 behavior extractions. Zero actor updates, linear solves, stable probes, Q_dev evaluations or new annotations. Both OFF/native and original-action reference qualifications passed with full snapshots and RNG equal. All 80 gain/return/margin formulas were recomputed from saved scalars, all 960 original return entries were retained exactly, and dataset keys are unique and complete. Completion audit made zero model calls. No failed optimizer calls or recovery.

Cumulative campaign: 668,593 native updates excluding shared source 8,000 (676,593 including), 177,362 actor updates and 56 linear solves (52 data, 4 synthetic). Earlier negative rounds and V107 metadata recovery remain recorded.

## Evidence boundary

The first OFF return uses the old fixed 12-action behavior at its second decision. The appended second OFF return starts from the original retained prefix. Neither is the return of an optimized 13-action deployment policy. Training-return selection can overfit these same patients and streams. No hidden U labels, sealed test, new patients or actual later domains were accessed. Weights, snapshots, state features, raw logs and patient data remain private on NAS; public material contains source, protocol, scalar tables and accounting summaries only.
