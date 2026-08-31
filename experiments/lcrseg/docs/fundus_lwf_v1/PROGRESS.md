# Fundus LwF-style baseline V1: engineering passed, formal runs active

Snapshot: **2026-09-01 02:00:10 Asia/Shanghai** (2026-08-31 18:00:10 UTC). Scientific outcome: **pending**. Registration commit `6d89f39446840365cf709b414ed3c9d26ba5a297`; exact execution commit `4d4c2e4333fd0c75733b58d0c44227b15beedc6b`. Both were published and verified before the new real-data checks. Runtime source remains pinned to the execution commit; later reporting commits must not be checked out into the running source directory.

The sole existing training-code change preserves the concrete method class name in the shared Sequential-SSL constructor. It fixes Uniform-KD's checkpoint identity without changing the training loss or loop. Historical checkpoint files and results were not modified or resumed.

Engineering admission is **PASS_ENGINEERING**. The first focused suite passed 20 tests, with zero failures, errors or skips, including both arms' actual two-domain backward/checkpoint/resume tests. The one permitted real-batch check passed with exactly 24 forwards and two FP32 optimizer updates; golden repeats agreed and the independent old clone stayed frozen and gradient-free. The fixed two-case, 2,000-step overfit check passed: mean foreground soft Dice 0.999992073, minimum class soft Dice 0.999989033 and checkpoint-logit error 0. These are training-fixture engineering results, not held-out performance. The existing overfit utility's internal checkpoint Git sentinel is unchanged; its actual source is established by the pinned checkout and parent-observed job receipt.

All 1,119 training/validation-role input assets, totaling 157,158,028 bytes, matched frozen hashes, and were verified unchanged again before admission. Three manifest/split pairs and the checksum inventory also matched. Unlabeled training views had no label paths. Both overfit cases were verified against the frozen `train_labeled` role; dataset roles are taken from that manifest, not inferred from original filename text. No frozen test-role data was requested by the new checks or training.

Three queues were dispatched at 2026-08-31 17:56 UTC. Each runs its matched pair serially on the same GPU; a failed child or artifact gate prevents the next child from starting.

| Seed | GPU | Sequential-SSL | Uniform-KD T=2 |
| --- | --- | --- | --- |
| 0 | 4 | running, 2,500 persisted steps | queued |
| 1 | 5 | running, 2,500 persisted steps | queued |
| 2 | 6 | running, 2,500 persisted steps | queued |

These counters are checkpoint/log observations, not final completion claims. All three live resolved configs matched their frozen command plans. Each run has 13,400 registered steps, for six runs and 80,400 formal updates in total. GPU7 was used for engineering and is now free. Other GPU processes were not interrupted.

All new runtime files are under `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/fundus_lwf_v1_20260901`. The NAS mount and actual write/read probe passed. `operations/ENGINEERING_ADMISSION.json`, per-job `launch.json`/`exit.json`, `FORMAL_PLAN.json` and queue scripts record admission, source/config hashes and parent-observed process exits. Raw inputs, arrays, patient identifiers, checkpoint contents and the overfit montage are not published. The public admission file contains only aggregate engineering evidence and private-artifact hashes.

Next: observe the existing queues without duplicate launch. After all six `training_verified.json` receipts and actual child/queue exits are valid, recheck frozen input and checkpoint integrity, then implement/verify and execute the **single separate** test evaluation already specified in the protocol: all 36 lower-triangular test cells, case-mean Dice, all three paired seeds and unchanged feasibility bounds. No test evaluation has started. Validation scores cannot change any setting. No direct comparison with pooled repaired-B0 scores is admissible.

PMGC remains closed with G4/G5 failed. Its execution checkout was reverified unchanged, and the prototype-derived new-method line remains ended. This study does not reopen it. The 30-minute follow-up continues; neither engineering success nor a running job is scientific success.
