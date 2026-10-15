# Main experiment launch receipt — 2026-10-14

Implementation completed; engineering admission passed; formal experiment **RUNNING**, not scientifically completed. Execution source commit `27901f3753ead8c8fbcc32b3917332e0b056cf8d`, branch `codex/f5-module-pilots`.

Run ID: `main_style_ssl_v1_20261014T013808Z`.
Remote NAS root: `/mnt/source-nas/account/LCR-Seg/SSL_CL_seg/runs/main_style_ssl_v1_20261014T013808Z`.
Coordinator PID1990018 detached from SSH/session; GPU2–7 each runs one source job. Startup receipt confirmed all six native source jobs RUNNING with 200–300 committed and matching physical updates, CUDA resident, neutral main/worker argv. Controller log had no immediate error. This is a startup snapshot, not continuing monitoring.

| Benchmark | Frozen optimizer updates | Startup |
|---|---:|---|
| Fundus, seeds163/164, two orders sharing each seed's source | 37200 | two source workers RUNNING |
| Prostate, seeds163/164, one six-site order | 177400 | two source workers RUNNING |
| M&Ms canonical320, seeds163/164, one four-vendor order | 297000 | two source workers RUNNING |
| Total, 30 source/target DAG nodes | 511600 | six roots running; 24 targets depend on sealed predecessors |

## Engineering evidence and negative findings

88 relevant shared-engine regression tests passed. The original CPU invocation failed because the neutral working directory was not the source directory. A subsequent selected suite had 88 passed and one error from an unrelated legacy F2/CWMI external-source fixture, because its dependency path was absent. This main method does not use CWMI. Both original CPU logs and the scope admission remain on NAS; we do not claim the whole selected historical suite passed. Current main golden assertions additionally passed all2/3/4 classes, foreground no-PAS admission, conflict/taper denominator, background veto, RGB basis equality and mono-channel projection rank.

All three real-data qualifications passed source/target backward, EMA update, frozen-entry checks, exact source/target checkpoint replay, deployment merge, stage transition, seen-val full-volume metrics, and fixed96-update two-patient fitting. These are engineering evidence, not an estimate of held-out generalization or method improvement.

| Dataset | Initial two-case loss | Final two-case loss | Merged A/B/R gradient norms |
|---|---:|---:|---|
| Fundus | 1.813574 | 0.075712 | 0.018904 / 1.098333 / 0.511217 |
| Prostate | 1.384504 | 0.032297 | 0.006765 / 0.376001 / 0.190127 |
| M&Ms | 1.822624 | 0.189510 | 0.006043 / 0.510580 / 0.485496 |

Preserved real-GPU engineering attempts: attempt1 failed at device-object conversion before any optimizer call; attempt2 failed exact target replay after8 calls; attempt3 passed fundus but failed the legacy RGB-only first-convolution admission for prostate after108 calls; attempt4 passed all benchmarks after312 calls. Total engineering native invocations428, separately from formal budget. Failed snapshots/logs/checkpoints and the zero-optimizer stride diagnostic remain on NAS. CPU synthetic optimizer calls were not separately instrumented and are not included in428.

The fixes convert the device object, make normal/resume Q layout contiguous, and use numerical full rank min(m,d) for the registered mono-channel first convolution. Same r8 and k=min(m//2,d-r) are retained: first-layer protection k1 for MRI, k8 for RGB. No confidence/style/loss weights, dataset split, horizon, seed or validation selection changed.

## Result retention and evidence boundary

Frozen method and orders are in [PREREG.md](PREREG.md). No ablations or independent-test image/label access. U loaders have no GT fields; only fixed seen-domain val is evaluated, final checkpoint per stage. MRI uses stored normalized FP32 input and processed ZYX mm spacing; M&Ms ED/ES is averaged per patient. The auxiliary25 patients are excluded.

Each job retains physical-call ledger, status/exit/cost receipts, logs, final checkpoint and sealed deployment; private per-phase/case metrics flush as JSONL and partial domain aggregates persist on failures. One-empty distance Infinity is retained with explicit counts. Final `SUMMARY.json` retains every available class/site/seed/order matrix and signed Old/Incoming/Final/Forgetting changes; failed branches block descendants without retry while independent authorized branches continue. No scientific gain is assumed. Patient rows/checkpoints stay private on NAS. Completed anonymous aggregate tables and report/source require GitHub delivery later under global completion policy; this ongoing run is not yet publicly delivered.

Main status `STATUS.json`; fixed DAG `PLAN.json`; startup `STARTUP_HEALTH.json`; qualified evidence `engineering_04/RESULT.json`; main coordinator log `controller.log`; worker logs `configs/<node>.log`; job outputs `jobs/<node>/`. No automatic periodic monitoring or new experiment follows this launch.

## Current resource amendment

On2026-10-14 the user restricted execution toGPU4–7 and authorized multiple processes/card. Checkpoint-boundary migration completed with no optimizer replay; GPU2/3 released. Current coordinatorPID2008159 and policy2 slots/card supersede the startup allocation above. Training snapshot, all30 nodes and511600-update budget remain unchanged. See `../main_style_ssl_v1_ops/REPORT.md` and the NAS `resource_change_20261014T021117Z/` receipts for current placement and logs.
