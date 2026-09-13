# Evidence–claim matrix

Paths are relative to `experiments/lcrseg/docs/`. PENDING cells are not results.

| Claim/question | Evidence | Setting/status | Counterevidence or limit | Permitted wording / excluded claim |
|---|---|---|---|---|
| Historical LCTX improves over clean supervision on average | `ams_seq_transfer_v0_1/FINAL_REPORT.md`, commit 671d0e57 | Full parameters, seeds 61–63, repeated development cohort | Different parameter space and source states from new tasks; selected retrospectively | Historical average development gain; not new matched attribution |
| SIGN old positive control mean | `dpr_sign_replication_v0_1/post_hoc_analysis/PAIRED_EFFECTS.csv`, `PATIENT_INTERVALS.csv` | Post-hoc, old seeds, +0.00473157 | CI includes zero; never the old primary | Exploratory control finding, not confirmation or replacement of primary |
| SIGN new-seed replication | `dpr_sign_replication_v0_1/public_results/NEW_SEED_PRIMARY_RESULTS.json`, commit e2ac3943 | Seeds 71–73, -0.0001835866; CI [-0.0065045695,+0.0060014891] | Same development patients; positive estimate not reproduced | No positive replication estimate; not proof of no effect/equivalence |
| Matched source-scoring contribution | `lctx_paper_closeout_v1/PROTOCOL.json`; future `public_results/comparisons/PAIRED_EFFECTS.csv` | C_LCTX−C_FULLMIX, PENDING | Donor supervision and Dice grouping change together | Local recipe increment only if observed; no isolated mechanism proof |
| Practical recipe differences | Same fixed output, C_LCTX−C_CE/C_CED and C_FULLMIX−C_CED | Same sources/F_CONV space; PENDING | Comparisons include augmentation/loss changes | Matched practical recipe differences, not an all-method leaderboard |
| Target annotation sensitivity | `SUBSET_MANIFEST.json`; future low-arm contrasts and RUN_LEDGER | Target L16→8 / L10→5; fixed source, PENDING | No low-target CE/CED; same update budget means extra patient exposure | Fixed-source target-label sensitivity; not all-sequence 10% labeling |
| Identical mixed inputs | `lctx_paper_closeout_v1/core.py`, qualification receipt | Original native generator used inside same step | Supervision differs intentionally | Matched images/noise/masks; not identical gradients |
| FULLMIX supervised normalization | Same core and qualification | Raw multiplicity 2; valid-pixel CE; synthetic-image Dice | Ignore pixels and nonlinear Dice grouping | Matched CutMix-style segmentation control; not official CutMix/BCP reproduction |
| No new U/response computation | Operation accounting and access qualification | New 30 targets only, PENDING actual totals | Current EMA remains in training; historical SIGN has VJP cost | No new U or VJP if verified; not EMA-free training |
| Cost/efficiency | Future RESOURCE_ACCOUNTING and historical separate costs | 79,500 new updates fixed; wall time not yet observed | Shared GPU contention; updates do not establish speed | Report measured costs, no unmeasured speed advantage |
| Independent patients / third domain / SOTA | `SUBMISSION_SCOPE.md` | NOT_ESTABLISHED / NOT_INCLUDED | Absent independent provenance audit and comparisons | Explicit limitations; no independent generalization, 3+ domain retention or SOTA claim |
