# Exact checkpoint handoff to target-gpu

The authorized migration completed on 2026-10-14 at approximately 13:26 UTC. The four study workers continue on A100 GPUs 0 and 1, with at most two formal workers per card. The former study workers, controller and watchers on bridge-gpu were retired after verified target progress; unrelated projects were preserved. Training is ongoing, so this is an engineering handoff report, not an experiment completion or scientific success claim.

| Chain | Domain | Exact handoff checkpoint |
| --- | --- | ---: |
| Prostate seed 163 | BIDMC | 8000 |
| Prostate seed 164 | UCL | 6000 |
| M&Ms seed 163 | Philips | 36000 |
| M&Ms seed 164 | Philips | 32000 |

The paused live workers first continued only to their next committed checkpoint. Each source ledger equaled the checkpoint cursor before the worker was frozen and its prefix copied. Each source and target passed exact student, EMA, optimizer buffers, scheduler, scaler, prototypes, RNG, probe, telemetry and provider-read restoration, with finite backward gradients. The next training batches, transformed tensors, labels, patient sampling and stream order matched exactly. These private batch traces and receipts stay on authorized remote storage. Both source and target engineering optimizer calls were zero. No updates were replayed and source checkpoints/history were retained.

The frozen 30-node, 511600-update protocol, seeds, configuration and data order remain unchanged. Original workers retain snapshot `27901f3753ead8c8fbcc32b3917332e0b056cf8d`; only the previously authorized Prostate seed 164 target chain uses `c234bcd23e0fc26737eb7833624434686c0846e6`. No source retraining, independent-test access, ablation, tuning or RL resumption occurred. Existing admitted future nodes remain under the same frozen controller.

The target uses Torch 2.6/cu124 instead of Torch 2.7.1/cu128. External operations normalize disabled Adam semantic metadata and the extra Torch 2.6 `LambdaLR.verbose=False` print-state field, and restore authenticated stage-entry tensors. These operations do not change the learning-rate equation or model state. FP32, disabled AMP/TF32 and deterministic settings remain in place. Cross-backend bitwise trajectories are not claimed. At the handoff, the largest absolute source/target labeled-loss difference was about 7.1e-8 and the largest unlabeled-loss difference about 2.3e-8.

Ten CPU operation checks pass. In the final 25.8-second sample, all four verified PID/birth identities advanced: Prostate seeds 163/164 each gained 35 updates; M&Ms seeds 163/164 gained 26/29 updates. Aggregate throughput was 4.84 updates/s. Instantaneous GPU utilization was 98%/77%, with approximately 36/31 GiB free VRAM. This short sample is not a full-run throughput guarantee.

Negative engineering records are preserved: missing neutral-entry parent directory; NFS extended-attribute copy failure, corrected by writing the small metadata bytes directly; a strict scheduler-state rejection; an initial compatibility hook applied to the base scheduler rather than the overriding LambdaLR serializer; an initial check loop referring to a not-yet-created configuration; and an audit collector merging worker/watcher labels with matching names. The corrected final audit checks all eight distinct source process identities and absence of all four former CUDA clients. No failure invoked an optimizer. Original failed/blocked attempts, prior host migrations and all pause records were copied and retained.

Thirty-minute monitoring has resumed against target-gpu only, staying quiet unless actionable. Full private evidence is under the current run's `host_migration_my_gpu_20261014T125300Z` directory, including `origin_audit`, native checks, handoff receipts, final health and retirement verification. Current operational metadata identifies the live controller; historical PIDs and copied prefix ledgers are not live progress or extra physical updates. Experiment completion, full positive/negative scientific results and public GitHub delivery remain pending the frozen run's actual terminal audit.
