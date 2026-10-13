# B1 three missing trajectories: startup confirmed

User authorized only163/O2,164/O1,164/O2, original configuration and15900 new formal updates. Prior163/O1 remains completed/read-only, including its5300 updates and five readouts. No V27 reduction or new hierarchy variant is authorized.

Frozen execution source `40b2e282cd7e028879fc88347563e26dcbe807eb`; plan `89255b5a941c9b1ac6eab0ead58b88e33acc1262c1107116f5bb3f845c83b089`. New root `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/simple_b1_complement_20261013T034900Z`; neutral controller3143803. Started2026-10-13T03:48:56Z (11:48:56Shanghai). Source/PREREG were pushed and anonymous access plus remote SHA verified before launch.

## Actual startup evidence

One check at03:50:06Z (11:50:06Shanghai) confirmed three live workers and1299 actual optimizer calls across them, no exit/failure, zero stage seals yet. Counts are a sequentially collected startup snapshot, not a synchronized efficacy result.

| Job | GPU | Observed optimizer calls |
|---|---:|---:|
| SIMPLE_B1_S163_O2 | 4 | 448 |
| SIMPLE_B1_S164_O1 | 5 | 432 |
| SIMPLE_B1_S164_O2 | 6 | 419 |

Full parent/worker command lines and nvidia-smi process names passed the neutral naming rule. Actual GPU usage at the check was992–2156MiB; this is not peak memory. NAS mount plus actual write/read probe passed, approximately14.49TB available. Existing unrelated jobs were preserved.

## Qualification and initial launch correction

Original B1 training/constructor/model/data/update code and settings are unchanged from qualified source3ba19c9. Reused native qualification32CPU/10generatedCUDA/2real-L is explicitly marked inherited; there are zero new qualification optimizer updates. New scope/merge/gate checks passed. After correcting the launcher, a native constructor check instantiated source/adapters/EMA successfully with zero optimizer or model-forward calls.

The first preparation root ending034300Z omitted `CUBLAS_WORKSPACE_CONFIG`; three workers stopped during deterministic teacher construction before any optimizer counter or evaluation. All failure receipts and that root remain preserved. Restoring the original `:4096:8` setting and binding it before torch imports corrected the launch. The current execution is the first optimization of these three cells; this is not a retry after optimization or a changed experiment. See PREPARATION_FAILURE.json.

## Completion and delivery scope

Exactly15900 new formal updates, six new stages and three EXIT0 are required before fifteen new seen-domain evaluations. Reuse prior163/O1 scores without training or evaluation; full four-trajectory matrix then totals21200 B1 updates. The original F5/B2 gates, all24 paired class cells, and both new-only and full-matrix rows will be reported, including failures. The observed prior cell and reused development patients are not fresh independent confirmation.

Method formal total594273 remains fixed. Existing combined599573 plus15900 gives615473; qualification cumulative including B1 remains1280CPU/337generatedCUDA/68real-L, historical673 rollback and five NA retained. Startup physical calls are not additional to the15900 cap.

A completion-only follow-up is scheduled hourly for this already launched batch. It stays quiet while running, delivers public anonymous results/report/source/costs after completion, and deletes itself after delivery or a reported training failure; no automatic next experiment. No V27 weight reduction, new hierarchy variant, extra seed or tuning. Private weights, images, patient rows, configurations and raw logs stay NAS.
