# R1_12H_SUSTAINED_V1 completion

## Execution

**COMPLETED:** 28/28 tasks, 24/24 final endpoints, 40,000/40,000 unique committed updates. No missing endpoints or failed formal tasks.

The session ran from 2026-09-25 19:36:10 to 20:20:28 Asia/Shanghai (44 min 18 sec), including qualification. The full queue finished before the 12-hour cap; no additional training or R2/R3 was started. The final queue contains 28 COMPLETED entries.

| Accounting | Count |
|---|---:|
| Planned / actual formal commits | 40,000 / 40,000 |
| Formal optimizer attempts / successful calls | 40,000 / 40,000 |
| Failed formal optimizer attempts | 0 |
| Recovery / replay attempts | 0 |
| Synthetic qualification calls | 14 |
| Real L-only smoke calls | 8 |

Four shared query prefixes contribute 8,000 updates, four D0 controls contribute 12,000, and twenty suffixes contribute 20,000. All 24 final students are at global step 3,000. Six synthetic calls were consumed during failed pre-formal qualification before the E1 repair and remain charged.

## Results

Scores below are equal-domain means of per-image macro foreground Dice (0–1). Each domain is weighted equally, not by its validation set size. Full rim/cup/disc_union scores and support counts are in RESULTS.csv; domain means are in DOMAIN_MEANS.csv.

| Backbone | D0 | Q0 | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|---:|---:|
| UNET_QUERY_128 | 0.718000 | 0.730832 | 0.731319 | 0.726788 | 0.724586 | 0.726599 |
| DINOV2_VITS14_QUERY | 0.844045 | 0.837334 | 0.837746 | 0.837658 | 0.837809 | 0.837467 |

| Backbone | Q0−D0 | Q3−Q0 | Q3−Q1 | Q3−Q2 | Q3−Q4 |
|---|---:|---:|---:|---:|---:|
| UNET_QUERY_128 | +0.012832 | -0.006246 | -0.006733 | -0.002202 | -0.002013 |
| DINOV2_VITS14_QUERY | -0.006711 | +0.000476 | +0.000063 | +0.000151 | +0.000342 |

Directions are not consistent across backbones: Q0 exceeds D0 for UNet but not DINOv2; Q3 is below Q0/Q1/Q2/Q4 for UNet and slightly above them for DINOv2. Neither backbone meets the original Q3 progression criterion (equal-domain improvement at least 0.005 and no domain loss exceeding 0.01). UNet Q3−Q0 is −0.000014 on RIM-ONE and −0.012477 on Drishti; DINOv2 is −0.004909 and +0.005861. Thresholds were observe-only and did not stop the queue.

## Provenance, repair, and verification

All formal training and endpoint evaluation bind source commit `41dfa243f15b184d728c19422a9f23ddcb346de9`; E0 was `edee1ab84175e73c9e055ce904a93831ef7f2a9e`. The report publication commit is separate from the training commit. Historical R0 ledgers and reviewed results are unchanged.

The single engineering repair relaxed unsupported strict CUDA deterministic kernels to warnings before any formal updates. Fixed seed 261, paired initialization/data/augmentation, losses and optimizer configuration were preserved. Exact bitwise reproducibility across reruns is not claimed. See PATCH_LOG.jsonl.

Environment: Python 3.12.7, PyTorch 2.6.0+cu124, NumPy 1.26.4, CUDA build 12.4; A100 GPUs shared with unrelated work. BF16 autocast with FP32 master parameters; TF32 disabled. D0 uses canonical gathered pixelwise cross entropy; frozen project augmentation is an implementation choice, not a paper claim.

Preparation passed three CPU contract checks; both backbones passed CUDA checkpoint/RNG restore and deployment equality plus real-L smoke. Final student prediction equality after removing training-only state passed 24/24. TASK_LEDGER.jsonl and STATE_AUDIT.json cover physical update counts, final checkpoint readability, finite student tensors, aligned scheduler/data cursor and shared-prefix bindings. DIAGNOSTICS_SUMMARY.json summarizes the 100-update diagnostic records without publishing raw logs.

RESOURCE_REPORT.csv includes per-task training time, updates/sec, memory peaks, trainable parameters, checkpoint bytes and batch-1 384×384 inference latency. Checkpoint size includes recovery/training state. Latency was measured on shared GPUs and is not a controlled speed comparison.

## Scope and retained artifacts

This is one development-seed R1 run on exposed validation, not independent patient confirmation, statistical significance, or final paper evidence. All negative outcomes are retained. Private data, per-case schedules, receipts, raw logs and checkpoints remain in the server run directory and are excluded from GitHub. No R2/R3 or performance-driven tuning was performed.
