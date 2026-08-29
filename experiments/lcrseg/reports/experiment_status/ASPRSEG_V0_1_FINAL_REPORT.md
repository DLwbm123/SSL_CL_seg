# ASPR-Seg V0.1 final report

**Final status:** `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`  
**Protocol consequence:** hard stop before Part B method implementation or training  
**Optimizer steps executed by ASPR:** `0`

## 1. Previous-method freeze

`PREVIOUS_METHOD_LINES_FROZEN_FOR_ASPR` was recorded before ASPR work:

- LCR routing/SRA = frozen;
- SR-GAS = frozen;
- RAGM = not executed;
- ASPR = a new prototype-memory feasibility hypothesis;
- R0 network, losses, optimizer, scheduler, data, manifests, splits, and checkpoints = frozen;
- no architecture change and no third training network were authorized.

Evidence: `reports/experiment_status/PREVIOUS_METHOD_LINES_FREEZE_FOR_ASPR.{md,json}`.

## 2. Relation-space audit

The first audit produced a false-positive hard stop because its checker required
three descriptive axis strings to be literally equal. The frozen semantics
instead define the relation axis by reference to the anchor-bank axis. The
original failed audit was preserved, a failure bundle was written, and a new
non-overwriting corrected audit passed.

Corrected status: `ASPR_RELATION_SPACE_AUDIT_PASSED`.

- Relation source: existing `UNet2D dec3 -> ProjectionHead`.
- Feature dimension: 128, read from model/checkpoint rather than hard-coded in memory logic.
- Relation grid: existing quarter-resolution grid.
- Class order: background 0, optic-disc-rim 1, optic-cup 2.
- ASPR memory foreground IDs: 1 and 2; background excluded.
- Relation temperature: 0.1.
- Valid mask: existing strict full-cell relation-grid validity.
- All 9 seed/site R0 checkpoints were present with valid normalized anchors.
- All 6 consecutive-site old/current model pairs loaded strictly and produced
  compatible finite normalized relation tensors.

Evidence:

- `reports/experiment_status/ASPR_RELATION_SPACE_AUDIT_CORRECTED_20260828T123600Z.{md,json}`
- `reports/failures/aspr_relation_space_audit_false_positive_20260828T123600Z.{md,json}`

## 3. Three-seed memory reconstruction

Seeds 0/1/2 were executed independently on physical GPUs 5/6/7. The memory
loader used frozen evaluation batch size 4, deterministic order, no color
jitter, no strong augmentation, and no cutout. Each bundle contains only
calibrator, prototype, dispersion, transport, lineage, and environment state;
it stores no image or pixel replay.

| Seed | GPU UUID | Manifest SHA-256 | Split SHA-256 | Bundle SHA-256 |
|---:|---|---|---|---|
| 0 | `GPU-2052f3b4-88f8-4be9-d43c-5068fafb02a5` | `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3` | `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88` | `d113ce1f45c3d6ce2ef25f62b572895e8e838ac637557d97a5df54be1a0a1ac7` |
| 1 | `GPU-07e6f403-f306-3fc3-709a-37c245743ee3` | `d5d2913054bc96f13b2baec0f21109a7da92c1a2f5b07f0cde234b35bbfd92a9` | `87affde62045894a8ce89701137f254ed56ba1f00951041bd2f6282cccbb5727` | `edaa98fbe25bc47496e3606679bcca92c6dc4023813165ec343a7b8ded4783c1` |
| 2 | `GPU-885f8cbc-d6bd-3ba9-f65d-a34373a93c0c` | `78379dc43035259f41b0f598e0bda25a31e68b15600bb611758ccc61cd2a0727` | `af2f48281d8eb16d299871f12824a729d08cb3854b3753d69d42c0d842e34dd3` | `2c589cf515951ed94d3c44b3d48c9c59058203bdf1f4efc78e88039fc83dba29` |

Runtime was Python 3.10.6, PyTorch 2.2.1+cu121, CUDA 12.1, cuDNN 8902,
driver 580.173.02, on three RTX 3090 GPUs.

## 4. Gate A: reliable unlabeled memory

**Result: FAIL.**

The frozen 0.90 reliability threshold selected broad foreground coverage, but
the selected pixels did not achieve the required 0.90 hidden-GT precision in
any seed for either foreground class.

| Class | Seed-0 precision | Seed-1 precision | Seed-2 precision | Required seed count | Observed |
|---|---:|---:|---:|---:|---:|
| optic-disc-rim (1) | 0.792632 | 0.803854 | 0.840448 | at least 2/3 | 0/3 |
| optic-cup (2) | 0.867145 | 0.871994 | 0.862327 | at least 2/3 | 0/3 |

Other Gate-A checks passed:

- coverage >= 0.02: 3/3 seeds for both foreground classes;
- median `Delta_LU`: +0.011580 >= +0.005;
- nonnegative `Delta_LU`: 77.78% >= 70%;
- 10th-percentile `Delta_LU`: -0.004098 >= -0.010.

The aggregate prototype benefit does not override the failed pixel-precision
safety condition.

## 5. Gate B: static prototype drift

**Result: PASS.**

- Median static cosine degradation: 0.789083 >= 0.030.
- Fraction with degradation >= 0.050: 100%.

This strongly supports the presence of relation-coordinate drift, but it does
not authorize training when Gate A fails.

## 6. Gate C: evidence-adaptive transport

**Result: PASS.**

- Median shrinkage-minus-static oracle cosine: +0.691856 >= +0.020.
- Shrinkage better than static: 100% of pairs.
- 10th-percentile improvement: +0.349164 >= -0.020.
- Median shrinkage cosine: 0.829138.
- Median full-shift cosine: 0.832731; shrinkage was within the allowed 0.010.

All transport estimates used 16 labeled cases for REFUGE->RIM-ONE-r3 and 10
for RIM-ONE-r3->Drishti-GS. No transport-strength hyperparameter was introduced.

## 7. Gate D: site-mode utility

**Result: FAIL.**

- Median nearest-prototype distance reduction: 6.93%, below 10%.
- Worst max-over-site NCM minus global NCM: -0.008570, below -0.005.
- Own-site top-mode >= 0.55 occurred in only 1/3 seeds for each foreground class,
  below the required 2/3.
- Historical prototype occupancy was nonzero; minimum occupancy was 105 pixels.

Thus site modes existed but were not sufficiently useful or consistently
site-identifiable under the frozen gate.

## 8. Feasibility decision

The final status is `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`, determined by the
first failed research gate. Gate D also failed independently. Gates B and C
passed, but the protocol requires A-D all to pass.

Per the preregistration, H3 cannot be silently promoted to the proposed method
when unlabeled memory fails. A new protocol is required for any labeled-only
transport follow-up.

## 9. Hidden-GT isolation and engineering checks

- Memory reconstruction reports `hidden_gt_usage=none`.
- Only `scripts/audit_aspr_prototype_feasibility.py` imports the diagnostic
  label resolver; its outputs report `hidden_gt_usage=post_hoc_only`.
- `lcrseg/memory/` and `scripts/reconstruct_aspr_memory.py` contain no hidden-label
  or diagnostic-resolver import.
- Three seed summaries, all four canonical CSVs, and all expected row counts
  were complete.
- No ASPR training process or optimizer step was started.
- GPUs 5-7 were idle after the audit.

## 10. Tests

- ASPR Part-A primitive tests: 5/5 passed.
- Project test boundary `pytest tests/`: 176/176 passed.
- A root-level unrestricted pytest attempt also collected vendored JASCL/MMDet
  tests and stopped at collection because `mmcv/mmengine` are intentionally not
  installed. This was outside the project test boundary and caused no source or
  environment change.

## 11. Added files

Part-A implementation only:

- `lcrseg/memory/__init__.py`
- `lcrseg/memory/site_prototype_memory.py`
- `lcrseg/memory/site_prototype_builder.py`
- `lcrseg/memory/reliability_calibrator.py`
- `lcrseg/memory/prototype_transport.py`
- `scripts/audit_aspr_relation_space.py`
- `scripts/reconstruct_aspr_memory.py`
- `scripts/audit_aspr_prototype_feasibility.py`
- `scripts/compile_aspr_feasibility.py`
- `tests/test_aspr_feasibility_primitives.py`

No ASPR method, hierarchical loss, experiment config, training runner change,
or formal training run was created after the hard stop.

## 12. Canonical evidence

- Gate report: `reports/experiment_status/ASPR_FEASIBILITY_AUDIT.{md,json}`
- Memory quality: `reports/analysis/asprseg_v0_1/memory_selection_quality.csv`
- Prototype drift: `reports/analysis/asprseg_v0_1/prototype_drift.csv`
- Transport quality: `reports/analysis/asprseg_v0_1/transport_quality.csv`
- Site-mode utility: `reports/analysis/asprseg_v0_1/site_mode_utility.csv`
- Per-seed bundles and summaries: `reports/analysis/asprseg_v0_1/seed0/`,
  `seed1/`, and `seed2/`.

Canonical analysis SHA-256 values:

- memory selection: `f7a63e61ac6a61413f0704374c6371d73726706409462f4037aa0ccf9272ac21`
- prototype drift: `6720286b7895a7142bb62947488f0eaa7593220e4fbe5ac1ad7019668e7dd122`
- transport quality: `175a3f5e4ceb98714bc5eba323fa311b59ac5f1c1725d8d1f78fc6200f1b8ca8`
- site-mode utility: `8e8783257b17fabe65dab5886c0727b97cfbe05edf8edd753faf33fc8a5cd439`

## 13. Explicitly unexecuted

Because feasibility failed, the following were prohibited and not executed:

- ASPR hierarchical relation loss and training method;
- H0-H4 training configs;
- site0/site1 R0 equivalence;
- seed-0 and seed-1 pilots;
- seed-0 full matrix and shift-swap control;
- multi-seed full runs and frozen post-hoc training analyses;
- external baselines;
- Prostate A/B/C;
- M&Ms A/B/C.

No result is claimed for any unexecuted item.
