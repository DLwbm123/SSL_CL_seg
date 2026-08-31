# Fundus LwF-style baseline V1: final negative result

**Status: FAIL_BASELINE_FEASIBILITY. This registration is closed.** Six formal runs completed, all engineering and artifact gates passed, and the sole test readout produced all 36 registered cells. Uniform-KD improved final Dice in all three seeds and improved retention, but failed the fixed incoming-learning bound. No threshold, seed, coefficient, temperature, denominator or budget was changed.

The mean paired difference in incoming Dice was **-0.010064513822073073**, below the preregistered minimum **-0.01** by **0.000064513822073073**. In percentage points these are -1.0064513822073073 versus an allowed -1.0. Rounding to two decimal places would hide the failure; the decision uses the full values. This difference is much larger than the 1e-12 arithmetic audit tolerance. The result is a valid scientific negative, not an execution failure.

## Fixed comparison and scope

See [PROTOCOL.md](PROTOCOL.md) and [registration.json](registration.json). The question was whether fixed uniform output distillation, T=2 and coefficient 1, improves a matched legacy weak/strong SSL baseline on REFUGE -> RIM_ONE_r3 -> Drishti_GS. Both arms were freshly trained with seeds 0/1/2 and their respective frozen splits, FP32, 200 epochs/domain and 13,400 updates/run. There was no checkpoint selection, parameter search, hidden-training-label access or test-driven tuning. Validation outputs never changed the protocol.

This is an **LwF-style fixed-class Fundus adaptation**, not a faithful reproduction of the original classification paper. It uses the legacy shared runner and patient-mean hard foreground Dice. It cannot be directly compared with repaired B0's different training framework and pooled Dice. Neither this result nor its engineering checks establish clinical significance, independent peer review, a new method or overall project success.

## Complete paired evidence

All values below are Dice-scale fractions. F is the mean of the final row's three domain scores; I is the mean of the three diagonal scores; BWT averages final-minus-diagonal for the first two domains. Higher is better for all three. The evaluator averages the two foreground-class Dice values per patient and then patients per domain; both-empty class Dice is 1. All registered required Dice values are finite.

| Seed | Arm | F | I | BWT |
| --- | --- | ---: | ---: | ---: |
| 0 | sequential_ssl | 0.639840292 | 0.779868530 | -0.210042356 |
| 0 | uniform_kd | 0.662270134 | 0.746318265 | -0.126072198 |
| 1 | sequential_ssl | 0.638073024 | 0.758496132 | -0.180634662 |
| 1 | uniform_kd | 0.693841227 | 0.765441765 | -0.107400807 |
| 2 | sequential_ssl | 0.617942191 | 0.762227390 | -0.216427799 |
| 2 | uniform_kd | 0.711204723 | 0.758638480 | -0.071150635 |

Differences are Uniform-KD minus Sequential-SSL within the same seed:

| Seed | Delta F | Delta I | Delta BWT |
| --- | ---: | ---: | ---: |
| 0 | 0.022429842 | -0.033550264 | 0.083970159 |
| 1 | 0.055768203 | 0.006945633 | 0.073233855 |
| 2 | 0.093262533 | -0.003588910 | 0.145277164 |

The mean F was 0.631951835612 for Sequential-SSL and 0.689105361203 for Uniform-KD, an increase of 0.057153525591. The mean BWT improved by 0.100827059120; the mean I decreased by 0.010064513822. These are descriptive paired results from three seeds; no statistical-significance claim is made.

| Registered criterion | Observed | Minimum | Decision |
| --- | ---: | ---: | --- |
| mean_final_dice_improvement | 0.057153525591 | 0.01 | PASS |
| positive_final_dice_seeds | 3.000000000000 | 2 | PASS |
| per_seed_final_dice_improvement | 0.022429841503 | -0.01 | PASS |
| mean_bwt_improvement | 0.100827059120 | 0.01 | PASS |
| mean_incoming_dice_improvement | -0.010064513822 | -0.01 | FAIL |

All five conditions were required. Four passed and one failed. Full precision is retained in [RESULT.json](RESULT.json); all 36 cells, both foreground classes and each patient denominator are in [TEST_CELLS.csv](TEST_CELLS.csv). No seed or cell has been omitted.

## Execution and independent arithmetic check

The initial synthetic suite passed 20 tests. The readout suite passed 29 tests (20 repeated shared checks plus nine readout checks), with zero failures, errors or skips; two of at most four registered synthetic-suite invocations were used. The single real-batch check and single fixed 2,000-step overfit check passed earlier and were not repeated. Their diagnostic scores are not scientific performance.

All six original training children and all three queues exited 0. Each run logged exactly 8,000 + 3,200 + 2,200 contiguous steps: **80,400 formal optimizer updates total**, with finite losses and zero skipped updates. The admission verified frozen planned/resolved configs, all 18 final-stage checkpoints, the unchanged previous-model tensors, metadata and previously audited training/validation bytes before requesting a test-role view. The last queue ended at 2026-08-31 18:38:21 UTC.

The sole test child ran on GPU7, completed at 2026-08-31 18:54:29 UTC and exited 0 with empty stderr. It made **612 model forwards and 2,430 case predictions**, with zero optimizer steps. Forward hooks saved the same predictions used by the evaluator, without additional model calls. The follow-up auditor used only saved arrays and frozen test labels: it recomputed every class Dice, case/domain mean, F/I/BWT, pairing and gate within 1e-12, checked exact case order/coverage and file hashes, and made **zero model forwards and zero optimizer steps**. All 1,307 protected paths retained their hashes. See [ARTIFACT_AUDIT.json](ARTIFACT_AUDIT.json).

This is a separately implemented **artifact and arithmetic audit by the same agent**, not review by an independent researcher. Ancillary ASD/HD95 can be undefined for empty surfaces in the unchanged evaluator and remain in the private table; they are not substituted for, or used to filter, required Dice evidence.

## Provenance and archive

- Preregistration: `6d89f39446840365cf709b414ed3c9d26ba5a297`; registration SHA-256 `70dc562b87c8d49740253d9c91b5497cdce637dcd1320f2edaf5341b27b07c09`.
- Training/engineering source: `4d4c2e4333fd0c75733b58d0c44227b15beedc6b`. Its NAS checkout remains unchanged.
- Sole readout source: `3ff05c22877436a8c6d18d02a6abfb40051e37da`. Published and Git/GitHub-verified before the test; executed in a separate unchanged NAS checkout.
- Artifact auditor source: `1452fe5a612ce60954317732211d4bb07dd6e574`, script SHA-256 `5352e1212117c9a0bd500e202ed85ee103a5582aef0618cf761cb9c05dcdf52c`. The exact published script used unchanged standard-library archive helpers from the readout checkout; no training/evaluation source was updated.
- Raw result SHA-256: `556adabdb85830b66f23d811861cdc8c01def096a8576a95cbd0f99e4368d568`. The public result and cell CSV preserve their corresponding NAS bytes exactly.

The additive private NAS archive is `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/fundus_lwf_v1_20260901/archives/5b7227dab85f9329cda903f622c23034c390f96eb7ae38983320e062daa2c9a0`. Its `evidence.tar` contains **9,191 regular source/evidence files, 1,506,661,608 logical file bytes, 11 preserved symlinks and 14 preserved hardlink groups**. All archive members and original-source bytes were checked, and the bundle was sealed and atomically promoted. The four-file sealed bundle totals **1,432,315,964 bytes**; see [ARCHIVE_RECEIPT.json](ARCHIVE_RECEIPT.json). This distinguishes underlying evidence files from the tar/container file count.

The archive includes training and readout source, exact checkpoint/resume artifacts, engineering fixtures, completed run/queue/evaluation/audit logs and receipts, raw predictions and case tables. Its own active archive operation is excluded from the recursive payload; that completed operation and receipt remain in `operations/archive_1`. Final public-report verification is an additive `publication/` supplement, so there is no circular self-hash requirement. Every original source remains present. This is a verified copy **on the same NAS**, not an independent-device disaster backup. No HOME duplicate or frozen input was deleted or changed.

## Closure and finite next step

No more training, model forwards, test attempts, alternative KD settings or rescue arms are admitted under this registration. PMGC remains closed with `FAIL_PMGC_FEASIBILITY`, and the prototype-derived new-method line remains ended; previous Gate1B/Gate1C/MMPR negative evidence is unchanged.

The 30-minute heartbeat may next perform a bounded source/comparability audit for a different external, non-prototype method. Any new computation requires a separately recorded prospective question, fixed implementation/evaluation contract, seeds, budget, success rule and failure exit. It cannot relabel this comparison as passed, turn its test outcomes into tuning targets, or treat this study's closure as overall project success.
