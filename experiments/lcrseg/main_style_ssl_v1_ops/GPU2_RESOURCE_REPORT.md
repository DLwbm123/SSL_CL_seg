# GPU2 resource handoff admission

Status on 2026-10-14 12:37 UTC: armed and waiting for new committed checkpoints. No study worker has migrated yet and no throughput gain is claimed.

The user authorized GPU2 resource optimization for this study only. The target permits at most two study workers per card. Existing workers on GPU4/6 continue until their own checkpoint boundary; unrelated projects remain untouched. A pre-existing non-study GPU2 client is preserved with PID/birth matching. Additional unknown clients or less than 10 GiB free VRAM block new admission.

The external coordinator reuses the frozen worker and restore loop. It inherits completed nodes and live worker identities, reserves target slots while waiting for checkpoints, admits each actual boundary with zero optimizer calls, and retires the old study process only after confirmed continuation. It preserves the 30-node, 511600-update protocol, seeds 163/164, data order, FP32 settings, and both admitted algorithm snapshots. There is no source retraining, ablation, test-set access, optimizer replay, or automatic training retry.

Validation: five CPU admission/regression checks pass. Three representative native checkpoints pass exact student/EMA/optimizer/scheduler/scaler/prototype, RNG, cursor and provider-read restoration with finite backward gradients. All engineering optimizer calls are zero. The operative coordinator completed its first loop, two independent checkpoint watchers are waiting, all four inherited workers grow during a 25-second sample, and visible command lines are neutral.

Negative engineering records are retained: initial strict idle-card rejection after a non-study client arrived; a preflight rejected because the new launcher omitted the original deterministic CuBLAS workspace environment; a startup verifier treated an exited zombie as live; and the first CPU coordinator exited before scheduling because an inherited resume cursor was `None`. The environment was restored to the original value, the CPU field handling was fixed with a regression check, and only the CPU coordinator was recovered. None of these attempts restarted or updated a study model. Private receipts and full operational logs remain on the server and are excluded from Git.

Actual target throughput and the effect of GPU2 sharing remain unmeasured until boundary handoff. This report establishes engineering admission, not improved segmentation metrics or a scientific success gate.
