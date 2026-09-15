# Completed five-framework study: reduced single-seed protocol

**VERIFIED COMPLETE.** Finished 2026-09-16 00:45:36 China Standard Time (2026-09-15 16:45:36 UTC). The original launch-to-completion elapsed time was 37.94 hours. All experiment workers and the finite controller have exited. No new experiments or Phase E were started.

## Completion and provenance

- 328/328 retained DAG receipts: four source models, 280 development target stages, 32 replication target stages, and 12 selections.
- Exactly 826,800 target updates plus 32,000 source updates = **858,800 formal optimizer updates**.
- 112 cancelled target stages were never launched; 296,800 originally planned updates were avoided. Cancelled nodes are not counted as completed.
- Zero formal failure records. Final physical ledger entries and recorded optimizer attempt/completion counts match each stage cap; all 316 retained student files are nonempty. Models were not rehashed or loaded for this publication check.
- Worker implementation: `cc21871a43e3cd105cddaf831f5c3ea2fef59b9e`; controller: `da3a950bfe4eb3c97abb9124c5ddcadf7ee613b4`. Reference: **NATIVE_LR_SRC_A_3DOMAIN_V1**, not original KI.
- The [user-authorized amendment](SCHEDULE_REDUCTION_20260915.md) was applied after some development results were available. Historical source models, selections, checkpoints and old protocol files were preserved.

## Replication results

Seed **162** only, equal average over the two orders REFUGE → RIM_ONE_r3 → Drishti_GS and REFUGE → Drishti_GS → RIM_ONE_r3. Final is the equal average over three final seen-domain macro Dice values; each macro uses rim and cup. Old averages the first two seen domains, Incoming is the final incoming domain. Forget averages the source and first-target performance drop from their earlier evaluation to the final stage; lower is better. Values below are on the 0–1 scale; delta is percentage points.

| Method | Final ↑ | Old ↑ | Incoming ↑ | Forget ↓ | Δ Final vs B0 (pp) | D worker hours |
|---|---:|---:|---:|---:|---:|---:|
| B0_PARENT_LCTX | 0.624712 | 0.583535 | 0.707067 | 0.181595 | +0.000 | 1.011 |
| B3_PARENT_LCTX_DENSEG | 0.614322 | 0.573149 | 0.696667 | 0.193485 | -1.039 | 0.998 |
| B4_PARENT_PAS_KL_DENSEG_JOINT | 0.615337 | 0.568715 | 0.708582 | 0.193504 | -0.938 | 1.345 |
| F1 | 0.607755 | 0.558151 | 0.706963 | 0.215069 | -1.696 | 1.307 |
| F2 | 0.616492 | 0.566719 | 0.716038 | 0.197968 | -0.822 | 4.042 |
| F3 | 0.626762 | 0.584679 | 0.710927 | 0.184202 | +0.205 | 1.110 |
| F4 | 0.615525 | 0.564896 | 0.716783 | 0.202929 | -0.919 | 0.923 |
| F5 | 0.638665 | 0.596259 | 0.723476 | 0.170281 | +1.395 | 0.981 |

F5 has the highest Final score in this single-seed run. Its Final difference from B0 is +1.395 percentage points, with order-specific differences of +2.211 and +0.579 points. Its Old and Incoming means are higher and its Forget mean is lower than B0. F3 has a smaller positive average difference (+0.205 points), with opposite signs across the two orders. F1, F2 and F4 have lower mean Final than B0. These are descriptive results, not evidence of across-seed superiority.

All 15 framework-versus-B0/B3/B4 comparisons, including negative and mixed results, are in [PAIRED_COMPARISONS_REDUCED.csv](PAIRED_COMPARISONS_REDUCED.csv). These are matched seed/order aggregate comparisons. No patient-level confidence interval or multi-seed confidence interval is reported.

## Development selection and limits

The complete 70-candidate development table is in [DEVELOPMENT_CANDIDATES_REDUCED.csv](DEVELOPMENT_CANDIDATES_REDUCED.csv). B0 reused the chosen parent trajectory. F1–F4 each used eight development configurations; F5 used only C01 and C02 because those candidates had already started when the user reduced the schedule. F5 selected C02. Therefore the frameworks do not have equal tuning budgets. B1/B2 development outcomes are retained, while their unstarted D repetitions were cancelled.

Only one replication seed is available. No claim of multi-seed stability, independent-patient generalization, original KI reproduction, SOTA, or an official external-method reproduction is supported. The same development patients were used for tuning and seed replication. B3 and B4 were the previously frozen development references; they are not the strongest baselines in this new seed, where B0 has the highest baseline Final.

## Cost

| Scope | Training stages | Optimizer updates | Summed worker hours |
|---|---:|---:|---:|
| SOURCE | 4 | 32,000 | 1.805 |
| C | 280 | 742,000 | 95.989 |
| D | 32 | 84,800 | 11.717 |
| ALL | 316 | 858,800 | 109.511 |

Summed worker time includes stage training and evaluation and counts all completed development configurations and all four sources, including sources not used by the reduced D schedule. It is not exclusive GPU compute or wall-clock elapsed time. GPU sharing affects timing. Qualification and smoke are separate costs in the historical [COST_SUMMARY.json](COST_SUMMARY.json): cumulative CUDA qualification 60 optimizer calls and current-L smoke 24; this amendment added zero real qualification updates. Full per-stage telemetry and operation counts are published in the JSON export; expensive VJPs/readouts are not equated to optimizer calls.

## Public evidence and reproducibility

- [FINAL_RESULTS_REDUCED.json](FINAL_RESULTS_REDUCED.json): all retained training-stage aggregates, scores by domain and order, timelines, resolved options, selection results, cost counters, completion checks, and explicit cancelled IDs.
- [FINAL_METRICS_REDUCED.csv](FINAL_METRICS_REDUCED.csv): eight-method final/class metrics and replication costs.
- [STAGE_INDEX_REDUCED.csv](STAGE_INDEX_REDUCED.csv): all 316 executed source/target stages and their costs.
- [PAIRED_COMPARISONS_REDUCED.csv](PAIRED_COMPARISONS_REDUCED.csv): all 15 framework/reference comparisons and both order differences.
- [DEVELOPMENT_CANDIDATES_REDUCED.csv](DEVELOPMENT_CANDIDATES_REDUCED.csv): all 70 unique development configurations, including unselected outcomes.

The stdlib-only `experiments/lcrseg/scripts/export_reduced_native_results.py` validates completed artifacts and regenerates the public aggregate JSON from the private run root. It does not launch training, access patient-level files, or read model tensors. Recomputed final metrics and pair differences agree with the completion summary.

Patient identifiers, per-case validation, images, labels, raw/private logs, credentials and model weights remain private on the server. This report publishes aggregates only. Monitoring is paused after verified public delivery; the reduced matrix is closed and no further experiment is queued.
