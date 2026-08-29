# LCR-Seg V0.3 final report

**Final status:** `FUNDUS_V0_3_INTERNAL_GATE_FAILED`

The preregistered internal Fundus gate failed. The protocol therefore hard-stopped before strong-baseline seeds 1/2, P0 seeds 1/2, the external Fundus gate, bootstrap expansion, and the Prostate pilot.

## 1. V0.2a freeze boundary

- `reports/experiment_status/V0_2A_FREEZE_AND_V0_3_PREREGISTRATION.md`
- `reports/experiment_status/V0_2A_FREEZE_AND_V0_3_PREREGISTRATION.json`
- Frozen conclusion: V0.2a engineering gate passed but research gate failed; V0.3 validates only the R1 positive signal and does not re-enable teacher rejection.

## 2. Implementation and verification artifacts

Added or updated protocol code includes:

- `lcrseg/methods/lcrseg_v0_3.py`
- `lcrseg/methods/lcrseg_v0_2a.py`
- `lcrseg/methods/__init__.py`
- `lcrseg/engine/continual_runner.py`
- `lcrseg/analysis/v0_3.py`
- `configs/experiments/lcrseg_v0_3_r0.yaml`
- `configs/experiments/lcrseg_v0_3_r1.yaml`
- `configs/experiments/lcrseg_v0_3_p0.yaml`
- `scripts/golden_v0_3_p0_site1_bridge.py`
- `scripts/run_v0_3_experiment.py`
- `scripts/run_v0_3_p0.py`
- `scripts/evaluate_v0_3_p0.py`
- `scripts/posthoc_v0_3_admission.py`
- `scripts/compile_v0_3_internal_gate.py`

Eleven V0.3-specific tests cover semantic equivalence, P0 zero relation loss/backward, parent lineage, frozen schedule and split hashes, prohibited teacher rejection and hidden-GT training use, aggregation, and patient-level bootstrap units.

## 3. Test result

- Full remote suite: `70 passed in 15.60s`.
- All formal runs were also audited for 13,400 optimizer steps, finite logged losses, zero AMP skips, zero hidden-GT training use, zero old-model gradient detection, zero historical-anchor mutation, zero teacher rejection, frozen manifest/split hashes, complete site matrices, and complete checkpoints.

## 4. P0 exact bridge

- Report: `reports/experiment_status/V0_3_P0_SITE1_BRIDGE.json`
- Status: `PASSED` on physical GPU 4.
- Frozen parent checkpoint SHA-256: `9bdadf34a5a32d936b14cfff3f4c9ffa2ee62c5f24142ca12b4a3b9815c46b32`.
- Logits, relation distributions, learnability, admission masks, pseudo labels, anchors, losses, counts, optimizer state, and scheduler comparisons all had exact zero error.

## 5. P0 seed-0 result

Run: `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_3_p0_progressive_norelation_full200e`

| Final Dice | BWT | Incoming Dice | Previous-site Dice |
|---:|---:|---:|---:|
| 0.610189 | -0.196932 | 0.741477 | 0.602428 |

The continuation used 5,400 new optimizer steps after the frozen 8,000-step R1 REFUGE parent, for a full-equivalent budget of 13,400. All relation-loss and declared relation-backward quantities were exactly zero. The maximum aggregated schedule error was 0.015361, below 0.05.

## 6. Relation component gate

Status: `PASSED`.

R1 minus P0 at seed 0:

| Final | BWT | Incoming | Previous |
|---:|---:|---:|---:|
| +0.051460 | +0.080640 | -0.002300 | +0.080214 |

All four preregistered relation-component thresholds passed. This result only supports the relation component at seed 0; it does not override the later multi-seed internal gate.

## 7. R0/R1 seedwise results

| Seed | Variant | Final | BWT | Incoming | Previous |
|---:|---|---:|---:|---:|---:|
| 0 | R0 | 0.655105 | -0.118462 | 0.734080 | 0.675946 |
| 0 | R1 | 0.661649 | -0.116292 | 0.739177 | 0.682642 |
| 1 | R0 | 0.674048 | -0.117624 | 0.752464 | 0.652893 |
| 1 | R1 | 0.675104 | -0.137910 | 0.767044 | 0.642121 |
| 2 | R0 | 0.717026 | -0.083722 | 0.772841 | 0.690355 |
| 2 | R1 | 0.702788 | -0.093437 | 0.765080 | 0.677812 |

## 8. Paired differences and cross-seed statistics

R1 minus R0 per seed:

| Seed | Delta Final | Delta BWT | Delta Incoming | Delta Previous |
|---:|---:|---:|---:|---:|
| 0 | +0.006544 | +0.002170 | +0.005097 | +0.006696 |
| 1 | +0.001056 | -0.020286 | +0.014580 | -0.010772 |
| 2 | -0.014238 | -0.009715 | -0.007761 | -0.012543 |
| Mean | -0.002213 | -0.009277 | +0.003972 | -0.005540 |
| Sample std | 0.010769 | 0.011235 | 0.011213 | 0.010633 |

Direction counts were 2/3 positive for Final, 1/3 nonnegative for BWT, 2/3 positive for Incoming, and 1/3 positive for Previous.

## 9. Internal Fundus gate

Status: `FUNDUS_V0_3_INTERNAL_GATE_FAILED`.

Failed conditions:

- Mean Delta Final was -0.002213, below +0.003.
- Mean Delta BWT was -0.009277, below 0.
- Mean Delta Previous was -0.005540, below +0.003.
- Only 1/3 seeds had nonnegative Delta BWT and only 1/3 had positive Delta Previous.
- REFUGE optic-disc-rim cross-seed mean Delta Dice was -0.025358, exceeding the permitted 0.015 drop.

Passed conditions:

- Mean Delta Incoming was +0.003972, above -0.005.
- Final improved in 2/3 seeds.
- Overall mean class deltas were +0.002501 for optic-disc-rim and -0.006927 for optic-cup, both above -0.010.
- Maximum admission-schedule errors for seeds 0/1/2 were 0.022192, 0.003544, and 0.022553, all below 0.05.
- Admission mechanism passed: admitted-minus-candidate pseudo-label accuracy gaps were positive in 3/3 seeds for both foreground classes, with cross-seed means +0.044923 and +0.047993.

Authoritative gate report: `reports/experiment_status/V0_3_FUNDUS_INTERNAL_GATE.json`.

## 10. Conditional strong baselines

Not executed. Sequential-SSL and Uniform-KD seeds 1/2 were prohibited after the internal gate failed. Their expected run directories were checked and are absent.

## 11. External Fundus gate

Not evaluated because the conditional strong baselines were not permitted to run.

## 12. Conditional P0 multi-seed

Not executed. Although the seed-0 relation component gate passed, P0 seeds 1/2 also required the internal and external Fundus gates; those conditions were not met.

## 13. Admission, anchor, relation, and gradient analyses

- Frozen hidden-GT admission analysis completed independently on physical GPU 7 and is stored in `reports/analysis/v0_3/fundus_admission_analysis.csv` and `.json`.
- Hidden GT was not imported into training; all formal training logs report zero hidden-GT usage.
- Historical-anchor mutation and old-model gradient detection were zero in every formal V0.3 training run.
- P0 relation loss and backward contribution were exactly zero throughout its 5,400 new steps.
- A separate expanded relation/gradient comparison was not run after the hard stop; no unsupported result is claimed.
- The initial GPU7 console header contained an incorrect UUID; a correction was immediately appended. The actual queried GPU7 UUID was `GPU-885f8cbc-d6bd-3ba9-f65d-a34373a93c0c`.

## 14. Bootstrap

Not executed after the internal hard stop. The patient-level bootstrap implementation and unit test exist, but a tested implementation is not reported as an experimental result.

## 15. Prostate

Not executed. The required internal and external Fundus gates did not both pass, so the RUNMC to BMC pilot was prohibited.

## 16. Formal run paths

- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_uniform_relation_kd_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed0_lcrseg_v0_3_p0_progressive_norelation_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed1_lcrseg_v0_3_r1_progressive_uniform_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e`
- `/home/jiangsuiyang/SSL_CL/runs/fundus_seed2_lcrseg_v0_3_r1_progressive_uniform_full200e`

## 17. Explicitly unexecuted items

- Sequential-SSL seeds 1/2.
- Uniform-KD seeds 1/2.
- External Fundus gate.
- P0 seeds 1/2.
- Expanded patient-level bootstrap report.
- Prostate RUNMC to BMC pilot.

No result is claimed for any unexecuted item.
