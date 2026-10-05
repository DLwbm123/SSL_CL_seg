# V6 R0 stopped by user

Verified stop: 2026-10-05T12:19:04.610539+08:00.

The user stopped the loss-weight GRPO experiments and requested investigation of unlabeled-image coreset selection. The main coordinator and its three training workers exited; no owned GPU process remained. The hourly continuation automation is PAUSED. Existing checkpoints, source snapshots, data, ledgers and completed results are preserved. Resumption and R1/R2/R3 continuation are not authorized after this stop.

## Scientific status

- 62 of 126 preparation/training jobs completed; three jobs were interrupted and 61 remain pending. This is a user stop, with zero recorded technical failures before stopping.
- All 18 endpoints at 20% labels were evaluated and publicly delivered in [results/R0_interim20](results/R0_interim20/RESULT_REPORT.md), payload commit `5f11699120ff119a807ce28a5a906a02fb31dcd1`.
- No 10% or 5% validation scores were read. Their low-label success gates remain unevaluated. The campaign is incomplete; it is not a complete negative low-label result.
- The early 20% timing amendment remains part of the record. No full-campaign completion receipt is fabricated.

## Retained cost and recovery boundary

480,645 student optimizer calls were charged across completed jobs and interrupted attempts. Interrupted attempts recorded 64,497 development calls, 56 actor calls and 16 critic calls; attempt and success entries match, with no unparseable ledger lines. CPU optimizer costs are listed separately in USER_STOP_RECEIPT.json. The 96 synthetic qualification actor calls and original fixed-source training are separate from the student total. Evaluation resources are retained in the published 20% report.

The last durable student checkpoints for all three interrupted workers exist. They can lag the last logged optimizer update; no exact continuation from the stop instant is claimed. No costs or failed/interrupted artifacts may be discarded if a future, explicitly authorized resume occurs.

The proposed research direction is currently design only. No coreset training, new data access, additional label budget, or replacement automation was launched.
