# AGMS_DS_HALF_V1 — one-factor development test

Status: code preparation; production requires fresh external review. No approval issued.

## Evidence and hypothesis

The complete AGMS P1 result is published at commit ef6888dc81519ce9e9ca13ea497bbf6397747d70. All ten models are VERIFIED. A5 mean Final=0.639210 versus A0=0.657466; both P_perf and P_joint failed. M-only A2 loses mean Old by 0.027910 without mean Incoming gain, H alone stays close to A0, and A5-A4 mean Final=+0.00003842. Saved A5 weighted DS/supervised gradient norms exceed H/supervised by roughly two orders of magnitude. These are single-seed development observations, not causal proof.

Hypothesis: reducing DS pressure will recover some Old/Final performance without a material Incoming penalty. Change exactly lambda_DS=0.25 to 0.125 in A5. This also affects auxiliary-head learning and is not a pure upstream-gradient intervention. Do not change head learning rate, risk temperature, H strength, thresholds, PAS, EMA, batch, randomness or training budget. Preserve the original head RNG namespace to compare identical initialization.

## Matrix and reuse

Seed163 only. O1: REFUGE→RIM→Drishti, train only Drishti stage2 for2100 updates. O2: REFUGE→Drishti→RIM, train only RIM stage2 for3200 updates. Two new nodes,5300 formal scientific and physical calls. Use the same independently accepted B2 stage1 prefix per order, not any AGMS/NKA endpoint. No new source, stage1, seed, sequence, baseline retrain or sweep.

Import A0×O1/O2 from original B2 public evidence and original A5×O1/O2 from the complete AGMS P1 publication. Both are matched by prefix binding/scores/timeline. Distinguish imported costs from new execution. Prefix payload hashes and actual models are verified before real access. Existing P0 26-L confidence-only screening is imported, with zero new P0 images/updates; no P0 claim is remeasured.

## Fixed budget and admission

New CPU: one attempt,21 planned physical optimizer calls,24 hard cap. Cases: native A5 gradient/EMA2, continuation5, baseline equivalence4, two expected after-optimizer failures2, two fixed-batch L-only fitting cases8. The unused3 calls are not retry authorization. Prior AGMS80 and earlier134 remain separate and immutable. Do not run the old suite.

Future native synthetic CUDA:11 calls (A5 continuation5, B2-off equivalence4, prescribed failure injection2), no real tensors. Future L-only smoke:4 unchanged early-warmup updates in each order,8 total; discard all state. Formal5300, real total5308. Diagnostic points8,4 extra VJPs each,32 total. No physical retry/replay after an unexpected engineering failure. Closed sessions and exact physical-to-committed alignment remain mandatory. Only GPUs5/6/7 with actual memory margin and the existing environment/NAS wrapper.

## Analysis and stopping

Evaluate only after both endpoints pass actual-model verification. Compare candidate minus original A5 and candidate minus A0 in each order, then average orders within seed. Report Final/Old/Incoming/Forget, rim/cup/disc_union, support, risk/alpha, diagnostic norms/cosines, actual calls, memory and worker time. Six final rows,18 domain rows,9 paired rows,8 new diagnostic points. Historical diagnostics remain referenced at their original publication, not charged or presented as new measurements.

A narrowly supported DS-pressure hypothesis requires mean Final improvement vs original A5≥0.003 AND mean Old improvement≥0.003 AND each-order Final difference≥−0.002 AND each-order Incoming difference≥−0.005. These are development decisions, not significance/publication gates. Regardless of outcome report the absolute A0 deficit, retain all outcomes and stop. No automatic follow-up, no original P1R/P2/P3.

Evaluation domains already informed this candidate: call it development tuning, never independent confirmation, patient generalization, original KI reproduction or SOTA. Evaluation labels never enter training, risk or masks.

## Provenance and authority

The user's September18 autonomous-method-tuning delegation authorizes preparation and subsequent bounded research; it is not a fabricated external approval. Production authorization continues to require a genuine independent approval bound to the new commit, code tree, plan, original B2 prefixes, control imports, environment and execution plan, then a current delegation receipt bound to that approval digest. Old AGMS R2 is refused. Current stage: STOP_AWAITING_EXTERNAL_CODE_REVIEW after CPU preparation succeeds.

An isolated branch specializes the existing agms_cl_v0_1 package for the new study identity. It reuses the training loop and resume implementation. The approved 89dc765 checkout, original docs/CPU/ledger/results and source data are unchanged. This branch must never be used to resume the original AGMS run.
