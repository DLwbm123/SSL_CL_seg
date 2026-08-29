# SR-GAS V0.2 Hard-Stop Report

Status: `SRGAS_V0_2_SEED0_PILOT_FAILED`

## Frozen basis

- The V0.1a hard stop remains correct and immutable.
- No V0.1a full runs exist.
- V0.2 did not relax the original `0.015` worst-trajectory threshold.
- V0.2 changed only sensitivity timing and noise onset; architecture and R2C formula remained unchanged.
- The byte-identical seed-0 REFUGE parent SHA-256 was `8f188ba27074ecb09a689377982774e6cf59e8c1c652d3927be54fd7c377bf55`.
- Manifest SHA-256 was `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`; split SHA-256 was `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`.

The read-only historical audit reproduced the V0.1a A5 worst drops (`0.031446` REFUGE, `0.016931` RIM-ONE-r3). Its normalized AUC deltas versus A1 were `+0.018982` and `-0.000393`, confirming that the preregistered failure was a worst-point transient rather than an AUC or endpoint collapse.

## Implementation and tests

Implemented the frozen lagged-sensitivity buffer, successful-step 20% linear warm-start, stateless shared raw-noise stream, L0-L4/D1/D2 configurations, runner integration, audit/gate/analysis scripts, and checkpoint state.

- V0.2 contract tests: `23/23` passed.
- Full regression suite: `171/171` passed.
- All seven pilots used identical parent, batches, augmentation seed, and—for L1-L4/D1/D2—byte-identical raw-noise checksums at every step.
- All runs completed exactly 1000 successful incremental steps with NaN/Inf=0, AMP skip=0, hidden-GT training usage=0, old-model gradient=0, and historical-anchor change=0.

## Seed-0 pilot endpoints

| Variant | Final | BWT | Previous | Incoming |
|---|---:|---:|---:|---:|
| L0 | 0.585319 | -0.218052 | 0.614094 | 0.770310 |
| L1 | 0.594601 | -0.204895 | 0.627251 | 0.773288 |
| L2 | 0.580288 | -0.229745 | 0.602401 | 0.768875 |
| L3 | 0.590788 | -0.194827 | 0.637319 | 0.770491 |
| L4 | 0.595322 | -0.190298 | 0.641847 | 0.772988 |
| D1 | 0.584510 | -0.215200 | 0.616946 | 0.768106 |
| D2 | 0.588954 | -0.208259 | 0.623887 | 0.770415 |

L4 passed every endpoint/source gate:

- vs L0: Final `+0.010003`, BWT/Previous `+0.027753`, Incoming `+0.002678`;
- vs L3: Final `+0.004533`, BWT/Previous `+0.004528`, Incoming `+0.002498`;
- vs L2: Final `+0.015034`, BWT/Previous `+0.039447`, Incoming `+0.004113`.

## Gate failure

| Site | L4-vs-L0 worst drop | Worst step | Threshold | Normalized AUC delta | Step-1000 delta |
|---|---:|---:|---:|---:|---:|
| REFUGE | 0.046378 | 350 | 0.015 | +0.015873 | +0.027753 |
| RIM-ONE-r3 | 0.000685 | 350 | 0.015 | +0.006368 | +0.005355 |

The REFUGE worst-point gate failed. Recovery occurred by step 400, but the recovery, positive AUC, and positive endpoint cannot override the original safety gate.

Gate outcomes:

| Gate | Outcome |
|---|---|
| Engineering | PASS |
| Original worst-point safety | **FAIL** |
| AUC safety | PASS |
| Endpoint | PASS |
| L4 vs L3 relation-conditioning | PASS |
| L3 vs L1 adaptive | PASS |
| L4 vs L2 total-GAS | PASS |

## Failure interpretation

The evidence does not support the hypothesis that the V0.1a transient was caused only by site-start full-amplitude noise:

1. The L4 failure occurred at step 350, after the 20% warm-start had reached full amplitude at step 201.
2. L3 (lagged supervised sensitivity + warm-start) nearly passed but still reached a REFUGE drop of `0.016948` at step 350.
3. L4 reached `0.046378` at the same step, indicating that lagged R2C conditioning amplified the transient even though it improved endpoints.
4. At step 350, L4's logged lagged-to-current sensitivity L1 difference was `0.770266`, compared with `0.000696` for L3; L4 perturbation/weight ratio was `0.687878` versus `0.374371` for L3. This supports an inference that stale, rapidly changing R2C sensitivity geometry remains unstable after warm-up.
5. D1 (same-step + warm) still failed REFUGE at `0.024040`; D2 (lagged + no warm) failed REFUGE at `0.038451` and RIM-ONE at `0.029971`. Neither timing change alone satisfied safety, and their L4 combination did not eliminate the REFUGE transient.

Exact `cos(S_t,S_{t-1})` is unavailable because full sensitivity tensors were not persisted at the 50-step evaluation grid. The report marks this unavailable rather than reconstructing or rerunning completed pilots.

## Protocol stop

The following were not executed:

- seed-1 L0 parent and independent L0/L3/L4 pilot;
- seed-0 L0-L4 full matrix;
- L4 spatial-shuffle full control;
- seeds 1/2 full runs and multi-seed analysis;
- external Sequential-SSL, Uniform-KD/LwF, and SS-EWC baselines;
- Prostate RUNMC→BMC experiments.

No failed run was rerun, no completed run was overwritten, no hyperparameter was changed, and no downstream experiment was started.

## Evidence paths

- Detailed gate: `reports/experiment_status/SRGAS_V0_2_SEED0_PILOT.json`
- Historical audit: `reports/analysis/srgas_v0_2/v01a_trajectory_summary.json`
- V0.2 trajectory/noise/sensitivity/cost tables: `reports/analysis/srgas_v0_2/`
- Remote runs: `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_srgas_v02_*_pilot1000`
- Immutable NAS mirror: `/data_nas/jiangsuiyang/LCR-Seg/srgas_v0_2`
