# Frozen SIGN replication V0.1 completion

Run `dpr_sign_replication_v0_1_20260913_01` completed at **2026-09-13 18:10:17 +08:00**. The live closeout at 18:18:42 confirmed controller `COMPLETE`, 18 training receipts, 18 evaluation receipts, all 36 task exits zero, and no GPU67 controller failure receipt. The controller process had exited; GPUs 6 and 7 each reported 0% utilization and 1 MiB used at 18:17. The original parent's exit 1 remains the historical GPU relocation event.

## Fixed matrix and storage

- Six fresh source trainings and twelve paired target trainings completed: 47,700 formal optimizer/backward/EMA updates.
- SIGN recorded 25,440 response VJPs and 38,160 response forwards. GPU relocation's 15 additional physical updates are disclosed separately (47,715 physical training updates); qualification's 40 updates remain separate.
- All 18 nonempty deployment files and 36 epoch checkpoints are present on the established NAS run root. The run's archive receipt records `ARCHIVED_ON_NAS`; private weights, recovery artifacts and patient-level scores remain there.
- The archive receipt was emitted before the controller appended relocation accounting to `RESOURCE_ACCOUNTING.json` and `TERMINAL.json` and created `GPU_RELOCATION_COST.json`. Its file sizes describe that earlier snapshot, not the final files; it is preserved unchanged. No fresh hashes or byte-equivalence claims were added at closeout.

## New-seed result

| Metric | F_CONV | SIGN | SIGN minus F_CONV |
| --- | ---: | ---: | ---: |
| Final | 0.5957328953 | 0.5955493087 | -0.0001835866 |
| Incoming | 0.6990033101 | 0.7011606642 | +0.0021573541 |
| Old | 0.4924624805 | 0.4899379532 | -0.0025245274 |
| Forget | 0.1760217797 | 0.1785463071 | +0.0025245274 |

The primary averages orders within each seed, then equally averages seeds 71/72/73. Final's paired patient 95% interval is [-0.0065045695, +0.0060014891], crossing zero. The result is **NO_POSITIVE_REPLICATION_ESTIMATE**; the separate +0.005 practical reference was not met. Seed effects are +0.0113283910, -0.0025150934 and -0.0093640575. Reused development patients do not provide independent patient confirmation.

Old-seed post-hoc A remains separate, and the old primary `NO_PRIMARY_ACCURACY_GAIN` is unchanged. No additional experiments, tuning or main merge were performed. See `public_results/` for aggregate tables, domain/class costs, resource accounting and frozen source provenance. Public delivery includes these small aggregate artifacts and operational status; private data, patient-level records, weights and credentials are excluded.
