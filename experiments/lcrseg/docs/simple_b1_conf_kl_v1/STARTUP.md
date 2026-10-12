# Simple B1 supplementary baseline: qualified and running

The user requested a simpler baseline on2026-10-12. The existing B1 family now has a separately frozen two-seed/two-order run. Frozen source `3ba19c9f6c982427d98aa242c7b09816ba3fe1b0`; root `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/simple_b1_20261012T071133Z`.

## Definition and comparisons

Same native input-projected low-rank parent, current-L LCTX CE+Dice and current EMA soft KL. B1 uses geometry plus confidence>0.6, without PAS prototype-similarity selection. It retains B2_C06 lambdaU1.0, A/B lr0.0005, Adam/decay/schedule and0.2 warmup/ramp. The common engine computes prototypes but does not use their mask for B1; no speed claim. No R/Q/SWD, entry teacher, replay or extra inference component is introduced. The whole result will be reported even if this simpler eligibility scores higher than B2.

Historical B0/B2 and matched F5 remain explicit references; V13/V17 provide method context. Neither B2's result nor the ongoing V20 protocol is changed. This is a supplementary baseline, not a substitute strong comparator or a new method iteration.

## Qualification

Native qualification PASS, QUALIFICATION_EXIT0. Actual costs32CPU fit/10generatedCUDA/2discarded current-L optimizer calls, including the deliberately injected after-optimizer failure. Archived built-in B1 full model/EMA/Adam/scheduler/cursor/prototype plus loss/update parity passed. Confidence/geometry boundary, detached teacher, empty graph-zero and two-case fit passed; maximum logit-gradient error2.77556e-17, fit loss0.378187→0.00290024. The generated native golden case accepted294912 pixels despite a forced all-rejected prototype mask, with nonzero actual A/B U gradients. Teacher freeze, checkpoint continuation and changed-family rejection, identity deployment, own-prefix transition, failed-state stop and real current-L zero-U smoke passed.

Two preparation failures are preserved separately with zero optimization: system Python lacked torch during plan import; shell quoting removed environment-key quotes before qualification entry. Registration moved to the existing py38 interpreter and the launcher was corrected. There was no failed mathematical/native qualification attempt and no reset of optimizer costs.

## Background run and first receipt

Four trajectories,8target stages,21200formal/source0. First163/O1 engineering seals, then the other three; all8seals/4EXIT0 before20unified seen-domain evaluations. No retry, coefficient search, new seed, best checkpoint or test access. Original source checkpoints and frozen manifests/splits are reused.

Started07:14:56UTC (15:14:56China time); controller1801710, first worker1801723. Fresh actual NFS/write/read probe passed with14.51TB free and GPU4–7 free6028/21483/21484/19098MiB. Existing jobs were preserved. Default native py38 and matching580.178.04 driver libraries are used through the original NAS wrapper.

One startup check at07:15:46UTC passed:181physical formal calls,0sealed targets, first-stage latest durable status160/3200, no job/controller exit. This is supervised warmup; empty module stats do not indicate an inactive baseline. Neutral parent/worker/GPU process strings passed. Worker CUDA992MiB is a startup usage snapshot, not peak memory. No patient efficacy result is available.

New baseline formal cap21200 is accounted separately from V20's21200. Method formal total before V20 was424673; with both fixed full matrices the combined ceiling is467073. This coincidental total does not transfer an unused literature-candidate allowance. Baseline qualification32/10/2 is additional to the prior1024/257/52, yielding cumulative1056/267/54. All earlier failures and rollback costs remain charged.

Existing hourly monitoring will track both V20 and this baseline, notifying on meaningful changes and publishing full anonymous results/costs after completion. Private data, patient rows, weights, checkpoints, configuration and raw logs remain NAS. Startup and qualification establish execution only; two seeds and reused development patients do not establish independent generalization.
