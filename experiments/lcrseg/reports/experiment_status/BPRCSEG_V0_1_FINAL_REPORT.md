# BPRC-Seg V0.1 final report

**Final status:** `BPRC_GRADIENT_SCALE_NOT_SUPPORTED`  
**Protocol consequence:** hard stop before Part B method implementation or training  
**Part B authorized:** `false`  
**Optimizer steps executed by BPRC:** `0`

## 1. Frozen starting point

`TARC_V0_1_FROZEN_FOR_BPRC` was recorded before BPRC work. It preserves the exact TARC result `TARC_RELATION_FIDELITY_NOT_SUPPORTED`: TARC Gate A passed, Gates B/C/D failed, and no TARC method or training configuration was promoted.

BPRC therefore changed only the historical relation representation/reduction in a feasibility audit. It did not use anchor transport, feature mapping, a new temperature, a new relation weight, hidden-GT class balancing, or boundary-GT training. The R0 supervised loss, SSL path, native relation path, anchor lifecycle, data, checkpoints, and schedules remained frozen.

## 2. Exact metric reuse and descriptive postmortem

`BPRC_METRIC_REUSE_AUDIT_PASSED` verified direct reuse of the frozen TARC implementations for relation margin, previous-site fidelity, current-site safety, exact R0 supervised validation loss, baseline loss, and stateless functional validation loss. The frozen TARC source-module SHA-256 values matched before the feasibility run.

The descriptive TARC postmortem completed without optimizer steps or formula changes:

- mean class Gram distortion: `0.313718`;
- static disc-rim margin agreement: `0.773684`;
- class-transport disc-rim margin agreement: `0.764289`;
- mean disc-rim delta: `-0.009394`;
- Gram-distortion/delta correlation: `-0.205786`;
- exact reproduction of the six canonical TARC disc-rim margin deltas: `6/6`.

This postmortem was descriptive only and did not select or tune the BPRC formula.

## 3. Audit-only primitives

Implemented only the pre-feasibility loss primitive and audit infrastructure:

- B0: frozen R0 categorical pixel mean;
- B1: categorical loss balanced equally over classes present in the detached old-winner map;
- B2: top-2 Bernoulli pairwise loss with the same class balance;
- B3: all-competitor Bernoulli pairwise loss with the same class balance.

The winner always came from detached frozen-old relation scores. Pair probabilities used the existing relation logits and inherited R0 temperature; Bernoulli probabilities used epsilon `1e-6`. There was no GT grouping, foreground reweighting, rescaling, new lambda, optimizer update, or method registration.

For each of seeds 0/1/2 and both transitions (`REFUGE->RIM_ONE_r3`, `RIM_ONE_r3->Drishti_GS`), the audit used exactly 32 fixed current-site update batches, 16 previous-site validation batches, and 16 current-site validation batches. This produced:

- gradient rows: `768` total, `192` per B0/B1/B2/B3 variant;
- virtual-step rows: `768` total, `192` per variant;
- margin rows: `6912` total, `1728` per variant;
- virtual-step norm: exactly `1e-3`;
- B0 versus exact frozen R0 maximum loss error: `1.862645149230957e-09`;
- old-model gradient count: `0`;
- checkpoint/model mutations: `0`;
- optimizer steps: `0`;
- non-finite values: `0`.

All three fixed-batch manifest hashes recomputed exactly, and all before/after current and old model hashes were identical.

## 4. Gradient-scale gate

**Result: FAIL. This is the first failed preregistered gate and determines the final status.**

For B3 relative to B0 over 192 paired comparisons:

- median gradient-norm ratio: `3.530838` (required within `[0.5, 2.0]`);
- p10 ratio: `1.956723` (passes required `>=0.25`);
- p90 ratio: `6.808760` (fails required `<=4.0`);
- non-finite count: `0`.

The all-competitor reduction was therefore materially larger than the frozen B0 scale. The protocol explicitly forbids post-hoc rescaling, so this cannot be repaired inside V0.1.

## 5. Previous-site utility gate

**Result: FAIL independently.**

- B3 previous-val loss delta better than B0: `51.5625%` (required `>=60%`);
- median B0 previous loss delta: `-0.0004639141`;
- median B3 previous loss delta: `-0.0003655581`;
- required B3 bound: at most B0 minus `1e-4`, i.e. `<=-0.0005639141`.

B3 did not demonstrate the required historical utility.

## 6. Current-site safety gate

**Result: PASS.**

- median current loss delta B0/B3: `-0.0001479425` / `-0.0006766897`;
- B3 satisfies the B0 plus 2%-absolute-loss bound;
- median current Dice delta B0/B3: `+0.0000869483` / `+0.0002924316`;
- B3 satisfies the B0 minus `0.002` Dice bound.

This safety result does not override earlier failed gates.

## 7. Disc-rim margin gate

**Result: FAIL independently.**

- B3 improved disc-rim margin agreement versus B0 in `42.7083%` of comparisons (required `>=60%`);
- median disc-rim B3-minus-B0 margin delta: `-0.000058060` (required `>=+0.005`);
- classwise median B3-minus-B0 deltas: background `-0.000022007`, disc-rim `-0.000058060`, cup `+0.000197487`.

The class-safety floor itself passed, but the intended disc-rim improvement did not.

## 8. Pairwise-specific gates

**B3 beyond B1 class balance: FAIL.**

- median previous loss delta B1/B3: `-0.0005124900` / `-0.0003655581`; B3 did not satisfy B1 minus `5e-5`;
- median disc-rim agreement B3-minus-B1: `+0.000101067` (required `>=+0.003`).

**B3 all competitors beyond B2 top-2: FAIL.**

- median previous loss delta B2/B3: `-0.0006722845` / `-0.0003655581`; B3 did not satisfy B2 plus `5e-5`;
- classwise median B3-minus-B2 margins were `+0.000005629`, `+0.000084005`, and `-0.000046303`; the class-safety floor passed, but the previous-site utility criterion failed.

Thus neither pairwise structure beyond ordinary class balancing nor all competitors beyond top-2 was supported.

## 9. Engineering verification

- all 18 engineering checks in the canonical compiler: passed;
- primitive and inherited project regression: `196 passed in 19.58s`;
- hidden GT training usage: none; boundary/interior and semantic-class diagnostics were post-hoc only;
- TARC/R0 checkpoints and data: not modified;
- BPRC method files and training configs: absent;
- completed prior runs: not rerun or overwritten.

An initial unit-test fixture used an invalid tensor-constructor form and failed before entering the loss. The failure was preserved under `reports/failures/bprc_primitive_test_fixture_20260828T140700Z.{md,json}`. Only the fixture construction was corrected; the primitive, formulas, thresholds, data, and audit results were unchanged.

## 10. Canonical evidence and SHA-256

| Artifact | SHA-256 |
|---|---|
| `BPRC_FEASIBILITY_AUDIT.json` | `c1c98bfc16412a42360cf696f3d57d56645cb2d911da41c04c9c60e788695677` |
| `BPRC_FEASIBILITY_AUDIT.md` | `6e747f72b64a27049bdca2c6ba32f9f16003ddbf869608917399936a1153a01f` |
| `STOP_NEW_RELATION_METHODS.json` | `5f5ac4bfc1baa85f261e410d8d50c3735ee87ebfe0febabbcc1ce53c9d85f383` |
| `STOP_NEW_RELATION_METHODS.md` | `e7ed033774d28cff1d9ebae7dd8137ab4e810f4a4eeceb9d8beee2cd732b4454` |
| `feasibility_gradient_scale.csv` | `f2051b02f547ae4ebc511d53c33c23677f83f6673fefab2e1b9d2fe06d6178e6` |
| `feasibility_virtual_steps.csv` | `e567855871cb325b0c4bb6d33fe20ff2b8a7b2069755b9cb972341401baed32e` |
| `feasibility_margin_analysis.csv` | `bb7b405e0e186ab4662bcf1e30efe143c71bcc565aea486ad56ff1fa0cc4f0b5` |
| `BPRC_AUDIT_FULL_TESTS.log` | `4e0ffae17a6f7ba522be0c0b31b93530af3d3346ed59794f6fe2eb3ca741e988` |

## 11. Explicitly unexecuted

Because the Part-A status is not `BPRC_FEASIBILITY_SUPPORTED`, the following were prohibited and not executed:

- `bprcseg_v0_1` method registration;
- B0/B1/B2/B3/pair-shuffle training configurations;
- B0 one-batch golden or 500-step equivalence bridge;
- seed-0 pilots or full runs;
- pair-shuffle, seeds 1/2, frozen post-hoc, external baselines, Prostate, or M&Ms experiments.

`STOP_NEW_RELATION_METHODS` is binding for this protocol. Future work may only proceed under a new protocol and is limited to source-faithful DC2T/JASCL-PAS reproduction or preregistered strong baselines; no further relation-coordinate or relation-loss variant is authorized.
