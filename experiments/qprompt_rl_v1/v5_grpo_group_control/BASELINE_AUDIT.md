# Baseline audit

Resolved baseline: `90eacbacca00d3c03e1296d5c7e6ac22a047df3b`. V4 experiment execution snapshot: `24b5df091565f5d570025699e9cc9ee050365fcb`.

- Read V4 plan, controller, worker, protocol and FINAL_INTERPRETATION; followed engine/runner, native_parent, train_stage, native_data, evaluator and NAS wrapper references.
- NativeLRParent semantic identity is NATIVE_LR_SRC_A_3DOMAIN_V1. Both A and B train, rank 8; effective W0+B@(A−(A@V)@V.T). The frozen readout and remaining body weights do not train. No A-only conversion and no BA dual projection.
- Two independent stage1 domains from REFUGE; native Adam group lrs .0005 for A/B, weight decay .00004, polynomial scheduler exponent .9; 3200/2100 updates; FP32 disabled AMP scaler.
- Supervision, native LCTX warmup/augmentation, soft U loss and threshold .7, dense effective-weight EMA .99 retained. lambda=0 executes native supervised path and does not read U for its training update. Feature extraction still probes current U read-only.
- V4 Provider batches/geometry and LCTX/UL generators are stateless functions of segmentation seed/order/stage/cursor/stream, not method or controller seed. No worker dataloader cursor exists. U0 does not force extra U forwards for common randomness.
- Existing snapshot already owns student/teacher/Adam/scheduler/scaler, RNG, step/cursor/epoch, gradients, buffers, telemetry and read counters. V5 wrapper completes bookkeeping: options, physical update count, teacher module modes, Provider identity and checked sets. Empty prototype container is native bookkeeping, not old-domain memory.
- V4 fit/online/audit identities and full-L/U deployment split reused unchanged. Online and audit use different training patients; neither is an independent external cohort. Val used historically and for V5 pilot promotion; test sealed. Old REFUGE labels only evaluator.
- NAS config was read from V4 CONFIG.private.json. Historical dev sources resolve to V3E source receipts; offline actors and fixed panel/scaler resolve to V4 fit/panel jobs. No guessed dataset paths.
- Physical GPUs 5/6/7 identified live; 5 has an existing job with >23 GiB free, 6/7 have >24 GiB free. Existing processes untouched. /data_nas is NFS with 18 TiB available at audit.
- History lookup for source183–187 returned no matching source folder in bounded protocols search, but one historical directory was permission-denied. Claim limited to not used in V5 development.
- Original KI historical identity remains unverified. V5 is an incremental experiment on the V4 base. No original KI or paper SOTA reproduction claim.

Runtime complete-state/cache equality and data-role invariance must pass P0 and reference admission checks; this static audit alone is not qualification.
