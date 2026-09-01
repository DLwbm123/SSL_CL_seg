# SHOR-JASCL V0.3.1 Active-Support Fix recovery protocol

## Frozen predecessor

This recovery starts from review commit `311c631476ba20bfce9a46ce2ff254db96cc82b9` on a new branch and does not modify or replace any SHOR-JASCL V0.3 artifact. V0.3 remains `BLOCKED_BEFORE_VALIDATION_EVALUATION` with engineering blocker `BLOCKED_NUMERICAL_FAILURE`, error `nonfinite ridge alpha`, and null scientific status. H1-H5 remain unevaluated. Its execution counters remain zero for model construction, model forward, autograd, optimizers, training, validation-GT reads, and test-GT reads.

The frozen read-only input remains exactly 183 files and 4,386,018,614 bytes with content SHA256 `05c9008ad4496ccbdc51df6103638024d49fae4b3b4cdc2a9f829c5f3ab165bb`.

## Root cause and sole repair

Bootstrap OOF reconstruction intentionally initializes the full alpha array with NaN sentinels and fills only held rows whose bootstrap multiplicity is positive. Rows with zero multiplicity are outside the bootstrap empirical support and correctly retain NaN. V0.3 passed the full array to threshold selection, where the inactive sentinels triggered the numerical blocker before the multiplicity mask was applied.

V0.3.1 adds one active-support operation: for `alpha`, `labels`, and `multiplicity`, select only rows with `multiplicity > 0`, require at least one active row, and require every active alpha value to be finite. Both public calibration and threshold selection perform this compression before top-1, log-odds scoring, candidate generation, or calibration. Inactive rows remain untouched and may retain NaN. An active NaN or infinity remains `BLOCKED_NUMERICAL_FAILURE`.

This change does not use `nan_to_num`, fill inactive probabilities, recompute inactive rows, alter bootstrap draws, delete or resample cases, or change any threshold rule. With all multiplicities equal to one, threshold, calibration, and route results must be exactly unchanged.

## Frozen scientific protocol

Descriptors, ridge equations, lambda and temperature grids, folds, bootstrap draws, SHOR log-odds, top-1 tie handling, candidate thresholds, calibration cutoffs, current-expert fallback, S0-S4, H1-H6, validation-only evaluation, expert-probability caches, and private inputs are unchanged. Validation domain and segmentation GT remain evaluator-only after the candidate seal. Model construction, forward, autograd, backward, optimizers, training, and test-GT access remain forbidden.

## Admission and preflight

The existing 188 related tests remain required with zero failures, errors, or skips. Add only the registered active-support tests: inactive-NaN success, invariance to inactive values, active nonfinite blocking, all-one equivalence, and one synthetic bootstrap-to-route chain.

After the implementation commit is pushed and remotely verified, one read-only private preflight may execute only seed 0, stage 1, bootstrap replicate 0. It must confirm positive zero-multiplicity count, finite active OOF rows, unchanged inactive NaN sentinels, completed threshold selection, zero validation-GT reads, and zero model forwards. It publishes `SHOR_V0_3_1_PREFLIGHT.json`. No independent 30-unit preflight is allowed.

## Authorization and hard stop

Exactly one new create-only V0.3.1 `formal_01` is authorized under a new NAS protocol root derived from the implementation short SHA. It reads the original 183-file bundle without mutation and runs the complete registered SHOR validation flow. It may yield only a registered scientific status or an engineering blocker. No previous V0.3 partial OOF is reusable.

After publication, stop for independent review. No test evaluation, second attempt, threshold modification, validation refit, C0 regeneration, model training, LoRA, adapter, Prostate, MnMS, sweep, or main merge is authorized. A scientific failure permanently stops the image-only domain-agnostic snapshot-routing line.
