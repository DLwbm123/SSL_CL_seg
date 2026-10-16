# M&Ms V2 user-interrupted closeout

On2026-10-16 at13:04:51UTC the user requested stopping the current project
experiment and using GPU0 for a separate independent source baseline. This
supersedes the earlier instruction to finish the current V2 round. The
coordinator was frozen before terminating its two workers; all three owned
processes were absent at13:05:21UTC. No unrelated process was signalled.
The existing m-ms-v2 heartbeat was paused to prevent further scheduled work.
There is no V2 restart, retry, next stage, or next-round authorization.

## Actual formal consumption and missing endpoints

| Seed | Stage | Physical Adam updates | Frozen budget | Outcome |
|---|---|---:|---:|---|
|163|Philips|53300|53300|COMPLETE, final validation available|
|163|GE|1419|25100|INTERRUPTED_BY_USER; no committed periodic checkpoint in this stage; final metrics NA|
|163|Canon|0|26500|CANCELLED_NOT_STARTED; NA|
|164|Philips|50867|53300|INTERRUPTED_BY_USER; latest committed checkpoint at50000; final metrics NA|
|164|GE|0|25100|CANCELLED_NOT_STARTED; NA|
|164|Canon|0|26500|CANCELLED_NOT_STARTED; NA|

Total formal physical updates105586 of209800. Ledger counts include the1419
and867 updates after each interrupted branch's last durable stage checkpoint;
they must not be dropped from cost accounting. No attempt was made to replay
or evaluate these interrupted states. Existing artifacts remain private and
preserved. The six-target study did not finish; final study metrics and
two-seed comparisons are NA.

Separate costs:87200 inherited V1 source updates, zero new source Adam calls,
112 pre-launch engineering Adam calls. Source summary/bank preparation used
66 additional labeled image exposures per seed (132total); the completed
Philips final history collector used64 more. Completed Philips training
recorded106602 L and85280 U image exposures; precise complete exposure/forward
totals for interrupted workers are unavailable in final cost receipts and
are not inferred from final checkpoint counters. Stop/audit added zero model
forwards, gradients, optimizer calls or independent-test access.

## Available fixed-final result, seed163 only

| Endpoint Dice (%) | V2 | Same-seed V1 | V2 minus V1 (pp) |
|---|---:|---:|---:|
|Old Siemens|42.618857|35.307609|+7.311248|
|Incoming Philips|78.189961|78.122784|+0.067178|
|Mean over seen sites|60.404409|56.715196|+3.689213|

The source Siemens acquisition was65.817577%, so its signed acquisition-to-
Philips decrease is23.198720pp. This is reduced forgetting relative to V1,
not adequate preservation or evidence from two seeds. Siemens class Dice:
LV48.205062%, MYO46.540432%, RV33.111078%. Philips class Dice:
LV82.702241%, MYO76.905486%, RV74.962157%.

The completed stage recorded zero zero-displacement updates out of53300,
44187 corrected-call counts, and144620 accumulated residual-conflict counts.
Corrected calls include floating-point rounding, so this is not a meaningful
intervention fraction. Final student Siemens foreground proxies were
approximately.032743/.034448/.041206, with background.038569. Earlier-to-final
proxy increase plus Dice loss is descriptive concordance with insufficient
protection, not a calibrated proxy or causal effect. At the final step the
proposal-before and proposal-accepted differences were zero at recorded
floating-point precision. A nonzero displacement or orthogonal transform
does not establish beneficial learning. No post-stop inference was performed.

## Delivery boundary

This report contains anonymous aggregate evidence only. Source checkpoint
files, per-patient metrics, raw logs and private server metadata are not public.
The completed-stage V1 reference is commit19f30387. Runtime algorithm source
was cd8bf767. Delivery SHA and anonymous-access verification are recorded
separately after push; writing this report alone does not establish delivery.
