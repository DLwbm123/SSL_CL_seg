# F5_CONFIRMATION_V1 — completed P1 results

**COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW.** All 28 new target stages completed and passed actual model-file integrity acceptance. Formal scientific and physical updates are exactly 74,200. Training completed on 2026-09-16 at 18:27 China Standard Time. No further experiments or P2 were launched.

Runtime code: [667f178c3b80183fd80809760ff31ec5f9a14e25](https://github.com/DLwbm123/SSL_CL_seg/commit/667f178c3b80183fd80809760ff31ec5f9a14e25). This independent results branch does not change the clean runtime checkout.

## Main results

Primary inference uses seeds 163 and 164, first averaging the two orders within each seed. The table reports F5 minus each reference, on a 0–1 Dice scale.

| Reference | seed 163 Final delta | seed 164 Final delta | Primary mean Final delta |
|---|---:|---:|---:|
| B0_PARENT_LCTX | +0.015899 | -0.003762 | +0.006069 |
| B2_PARENT_PAS_KL | +0.006743 | -0.056546 | -0.024902 |

G1, G2 and G3 all fail the frozen budget-decision criteria; these are not statistical significance tests. G1 fails because seed 164 does not improve over B0. G2 fails on order O2 (mean Old delta -0.006546 and Forget delta +0.008668). G3 fails against B2. Both positive and negative outcomes are retained. The decision is **STOP_NO_ADDITIONAL_EXPERIMENTS**.

Seeds 162–164 are supplementary descriptions. B0/F5 seed 162 are historical imports, not retrained. These are development-patient results; they do not establish independent-patient generalization, original KI reproduction or SOTA.

## Qualification and costs

- CPU synthetic: 13/13 groups PASS, cumulative 30/40 optimizer calls and 6/8 test invocations.
- Native synthetic CUDA: 12/12 cases PASS, 21 calls including 3 planned after_optimizer failures. Prior fixture failure had zero optimizer calls; its failed cost session is retained.
- Three reused source models passed actual weight identity/hash/schema and synthetic-forward acceptance. New source training: zero.
- Real smoke: B0/B2/F5 each 8 current-L-only updates, total 24, U reads zero; disposable states were not inherited.
- Real-data optimizer calls: 74,224 = 74,200 formal + 24 smoke. Synthetic CPU and CUDA costs are separate.
- Formal failed sessions, restarted sessions and uncommitted physical calls: zero. Formal measured worker sessions total 5.061111 hours, including evaluation and shared-resource waiting; not exclusive GPU compute time.
- Integrity acceptance and source verification costs, operation counts, session histories, environment and CUDA peaks are retained in the JSON reports.

## Files

- [FINAL_REPORT.md](FINAL_REPORT.md): unchanged exporter report, including all final trajectories.
- [FINAL_METRICS.csv](FINAL_METRICS.csv): 18 trajectories, with explicit historical_import/new_execution.
- [STAGE_METRICS.csv](STAGE_METRICS.csv): 90 stage/domain records.
- [PAIRED_COMPARISONS.csv](PAIRED_COMPARISONS.csv): 34 paired records.
- [COST_AND_COMPLETION.json](COST_AND_COMPLETION.json): completion, costs, qualification and integrity evidence.
- [PUBLIC_RESULTS.json](PUBLIC_RESULTS.json): full aggregates and reproducibility metadata.
- [VERIFICATION.json](VERIFICATION.json): fresh aggregate/schema, cost and model-proof freshness check.

The original exporter records LOCAL_REPORT_READY because it does not push or verify GitHub. Those original reports are retained unchanged; publication verification is performed separately for this results commit. No weights, patient identifiers, per-patient results, images, labels, private paths or credentials are published.

## Approval provenance

External R2 was independent AI review of commit 2070845b679da42ea7a4df4268f492f2943deb6e, not human signoff or proof of CUDA/source/smoke success. An initial qualification-only bias assumption failed on the native bias-free classifier. The user subsequently explicitly authorized repair and continuation. Runtime commit 667f178 fixes only the generated fixture, adds narrowly scoped user-repair authority binding, and updates regression/evidence metadata. It is not misrepresented as the original externally approved commit. The original approval, launch receipts and failed attempt remain preserved outside the runtime checkout.

PLAN/source reuse/parent binding and all scientific options remain frozen. Parent identity is NATIVE_LR_SRC_A_3DOMAIN_V1. See the runtime code branch for the full plan and definitions.
