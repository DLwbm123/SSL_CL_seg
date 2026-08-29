# SR-GAS V0.1a Hard-Stop Report

Status: `SRGAS_PILOT_GATE_FAILED`

The implementation and engineering contracts passed, but the preregistered 1000-step pilot safety gate failed. The protocol therefore stops before all seed-0 full runs, A5 spatial-shuffle, additional seeds, external baselines, and Prostate experiments.

## Passed evidence

- V0.1a class-space bridge audit passed with class-semantics SHA256 `5c52655356b11831820433035dad0adfe919219a4da2a9f70d2b18d784010200`.
- SR-GAS contract tests passed `50/50`; the complete regression suite passed `148/148`.
- A1 two-case foreground Dice was `0.991204`.
- The A1 REFUGE common parent completed exactly 8000 steps and passed its gate.
- All A1-A6 pilots completed exactly 1000 new optimizer steps from parent SHA256 `8f188ba27074ecb09a689377982774e6cf59e8c1c652d3927be54fd7c377bf55`.
- NaN/Inf, AMP skip, hidden-GT usage, old-model gradient, and historical-anchor changes were all zero.
- A5 had valid, finite, nonzero R2C sensitivity; its noise scale differed from A4; projection-head proxy gradient and the R2C coefficient in the total objective were exactly zero.

## Failed safety thresholds

The fixed maximum A5-vs-A1 trajectory drop was `0.015`.

| Evaluation site | Worst step | A1 Dice | A5 Dice | A5 drop | Gate |
|---|---:|---:|---:|---:|---:|
| REFUGE | 350 | 0.730065 | 0.698619 | 0.031446 | 0.015 |
| RIM-ONE-r3 | 200 | 0.718829 | 0.701899 | 0.016931 | 0.015 |

At step 1000, A5 had recovered: its REFUGE result was better than A1 by `0.032163`, and its RIM-ONE-r3 drop was `0.002011`. This recovery does not override a trajectory-level preregistered gate.

No hyperparameter was changed, no failed run was rerun, and no downstream full experiment was started. The authoritative detailed gate is `reports/experiment_status/SRGAS_PILOT_REPORT.json`; all completed artifacts are mirrored without overwrite under `/data_nas/jiangsuiyang/LCR-Seg/srgas_v0_1a`.
