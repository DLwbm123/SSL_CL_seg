# SPARC-Seg V0.1 final report

Generated: `2026-08-28T23:44:45+08:00`

**Final preregistered status:** `SPARC_PAS_NOT_SUPPORTED`  
**Protocol consequence:** hard stop before SPARC method registration, training configuration, or optimizer execution.  
**Optimizer steps:** `0`  
**Hidden-GT training usage:** `none`  
**Hidden-GT analysis usage:** `independent post-hoc metrics only`

The documents supplied by the user were treated as the experiment protocol. The user's request authorized execution, but it did not authorize bypassing a failed preregistered gate. The feasibility compiler applies the registered failure precedence, so the PAS failure is the final status even though additional downstream diagnostic gates were also evaluated.

## 1. Previous-method freeze

`SPARC_PREVIOUS_METHODS_FROZEN` passed before any SPARC audit work. `STOP_NEW_RELATION_METHODS` remains binding for relation variants; SPARC was isolated as a representation-level protocol. Uniform relation KD and every historical method artifact remained unchanged. RAGM, EMA teacher in the proposed method, and cross-site prototype memory were not used.

Frozen upstream outcomes were preserved: V0.3 `FUNDUS_V0_3_INTERNAL_GATE_FAILED`, SR-GAS V0.2 `SRGAS_V0_2_SEED0_PILOT_FAILED`, ASPR `ASPR_UNLABELED_MEMORY_NOT_SUPPORTED`, TARC `TARC_RELATION_FIDELITY_NOT_SUPPORTED`, and BPRC `BPRC_GRADIENT_SCALE_NOT_SUPPORTED`.

Evidence: `reports/experiment_status/SPARC_PREVIOUS_METHOD_FREEZE.md/json`.

## 2. Source audit

Status: `SPARC_SOURCE_AUDIT_PASSED`.

- JASCL official repository commit: `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53` (exact required commit).
- STAR official repository commit: `6c9203aa0c91e9a2d4e40664c97754ae02226675`.
- LAG was recorded as conceptual inspiration only; no LAG module was implemented.
- Source-specific differences were reported rather than silently copied into SPARC. In particular, SPARC retained its frozen final-normalization/minimum-cell prototype contract and did not copy STAR's exact feature-maintaining formula.

Evidence: `reports/experiment_status/SPARC_SOURCE_AUDIT.md/json`.

## 3. Model-path audit

Status: `SPARC_MODEL_PATH_AUDIT_PASSED`.

The existing U-Net forward path now exposes the already-computed `dec3` and `dec1` tensors without adding trainable parameters or changing their storage. All six seed-transition checkpoint pairs loaded strictly and were shape-compatible. The pre/post golden comparison had maximum absolute difference `0.0` for logits, relation features, `dec3`, and `dec1`. Golden SHA256: `07cb1d786a387c2ec9c69ed9d110787592c7c843153a55939a8be555a96fac98`.

Evidence: `reports/experiment_status/SPARC_MODEL_PATH_AUDIT.md/json` and `SPARC_MODEL_PATH_GOLDEN_BEFORE.pt/json`.

## 4. Prototype/PAS feasibility

The current-site prototypes used all and only current-site `train_labeled` cases, with per-case normalization, equal case weighting, final normalization, and the frozen 32-cell minimum. The audit covered seeds 0/1/2 and both registered transitions.

| Foreground class | Mean current-PAS precision improvement | Positive pairs | Coverage >= 0.05 pairs | Registered requirement | Result |
|---:|---:|---:|---:|---|---|
| 1 | `0.012005` | `5/6` | `6/6` | mean >= `0.020`, positive >= `4/6`, coverage >= `4/6` | fail: mean |
| 2 | `0.013931` | `3/6` | `6/6` | mean >= `0.020`, positive >= `4/6`, coverage >= `4/6` | fail: mean and consistency |

Coverage was substantial, but the precision gain was too small and class 2 was not consistently positive. Therefore A1 failed without threshold tuning or seed replacement.

Evidence: `reports/analysis/sparcseg_v0_1/prototype_validation_quality.csv` (SHA256 `24c542174e173807191132c076cb0ee56369a7b87408d9108db7ecaf6ce8935e`).

## 5. Stable/plastic quality

A2 stable/plastic separation also failed, although it is not the status-defining first failure:

| Class | Median stable-minus-plastic precision | Stable > plastic pairs | Stable coverage >= 0.02 | Plastic coverage >= 0.02 | Result |
|---:|---:|---:|---:|---:|---|
| 1 | `0.016334` | `4/6` | `6/6` | `6/6` | fail: median < `0.030` |
| 2 | `0.114978` | `6/6` | `6/6` | `5/6` | pass |

A3 spatial coverage passed for both foreground classes. Boundary/interior stable-coverage ratios were `0.553745` for class 1 and `0.450189` for class 2, with nonzero plastic-boundary correct pixels (`20,574` and `4,663`).

Evidence: `reports/analysis/sparcseg_v0_1/partition_quality.csv` (SHA256 `129366493c29a3191e5637aa0f6dc8df7a43b826ffd207d1c13165458ec3069e`). Empty-denominator cells are represented as non-finite derived metrics and were not silently imputed.

## 6. Feature separation and gradient scale

Both B gates passed:

- Feature separation: `24/24` expected comparisons, median stable-minus-plastic cosine `0.142047`, stable-distance-not-greater fraction `0.791667`.
- Gradient scale/localization: `192/192` rows, median SFM/relation ratio `0.367364`, p10 `0.084896`, p90 `1.313955`, zero non-finite rows, all localization checks passed, and all frozen-old-model gradients were zero.

Evidence: `feature_separation.csv` SHA256 `6537ec6ac29e135df7df6748e4f9d863134ca4de50b2ca9f46c7df5464806bc2`; `gradient_scale.csv` SHA256 `3e8cd20b15bc255428672f1e9d80881d19d20907ce9938e15b14f0ca3034642a`.

## 7. Virtual-step gate

All `1,152/1,152` S0-S5 virtual-step rows were present. C1 previous utility passed and C2 current safety passed. S3 improved previous validation loss over S1 in `85.9375%` of paired batches. C3/C4 targeting/complementarity failed: the registered S3 comparisons against S4/S5 were not all satisfied, even though the S3 comparisons against S1/S2 were satisfied.

Evidence: `reports/analysis/sparcseg_v0_1/virtual_steps.csv` (SHA256 `a674f6dbf1ac6675d681ac81fd29d1c743b34bbbb2976ae2e00eef2fc602f82d`).

## 8. Feasibility status

Engineering completeness passed: no required artifact was missing; freeze, source audit, and model-path audit all passed. The registered precedence then returned `SPARC_PAS_NOT_SUPPORTED` at A1. Consequently `training_method_registered=false`, `training_configs_created=false`, and `optimizer_steps=0`.

Canonical evidence: `reports/experiment_status/SPARC_FEASIBILITY_AUDIT.json` (SHA256 `dc1b45e0db8833239b35a8e80317fec40013c7276610b13abcc2d849d6586502`) and `.md` (SHA256 `c534a0eca77c57687966baa6617b300590495ea5eae97119fe9dd871b88d1ca8`).

## 9. Added or modified files

Source and tests:

- `lcrseg/contracts.py`
- `lcrseg/models/unet.py`
- `lcrseg/semantics/__init__.py`
- `lcrseg/semantics/session_prototypes.py`
- `lcrseg/semantics/anchored_validation.py`
- `lcrseg/losses/__init__.py`
- `lcrseg/losses/stable_feature_maintaining.py`
- `scripts/audit_sparc_sources.py`
- `scripts/audit_sparc_model_paths.py`
- `scripts/audit_sparc_feasibility.py`
- `scripts/compile_sparc_feasibility.py`
- `tests/test_model_output_contract.py`
- `tests/test_sparc_feasibility_primitives.py`
- `third_party/STAR_REFERENCE/` official source checkout

Generated evidence is under `reports/experiment_status/`, `reports/analysis/sparcseg_v0_1/`, and `reports/failure_bundles/sparc_v0_1_posthoc_boundary_20260828T234044/`.

No SPARC training-method registry entry or training configuration was added.

## 10. Tests and engineering recovery

- Targeted SPARC/model contract tests after the post-hoc repair: `9 passed`.
- Complete project test boundary: `204 passed in 17.31s`.
- Python compilation checks passed for the audit and test scripts.
- A broad repository-root collection attempt reached vendored JASCL tests and produced dependency-collection errors for optional `mmcv/mmengine`; those are not part of the LCR-Seg project test boundary, and no environment packages were installed to mask this distinction.

The first post-hoc attempt failed before writing any post-hoc result because a diagnostic boundary cache assumed `256x256` while the clean Fundus view exposed a `384x384` decoder tensor. The failure logs, exit codes, original script, GPU state, and hashes were frozen. A minimal repair derived cache sizes from the actual frozen relation/decoder tensors; it changed no metric, threshold, data view, checkpoint, or random trajectory. The repaired attempt completed on all three seeds with exit code 0. Prototype-bundle hashes matched the visible stage and image maximum absolute difference versus the training view was `0.0` for all transitions.

## 11. S0 equivalence

Not executed. It is Stage B work and was conditional on `SPARC_FEASIBILITY_SUPPORTED`.

## 12. Seed-0 S0-S5 pilot

Not executed. No SPARC optimizer pilot was authorized. The S0-S5 computations reported above were preregistered zero-commit virtual steps only.

## 13. Seed-1 independent pilot

Not executed due to the feasibility hard stop.

## 14. Seed-0 full

Not executed due to the feasibility hard stop.

## 15. Mask-shuffle

Not executed because no seed-0 full run or downstream gate was reached.

## 16. Multi-seed

No SPARC training run was executed. The feasibility audit itself covered all three registered seeds.

## 17. Partition, feature, and cost analysis

The preregistered feasibility partition and feature analyses were completed as reported above. Training cost analysis was not executed because there was no SPARC training implementation or run; inference-overhead and checkpoint-size claims are therefore not made.

## 18. External/JASCL baseline

Not executed. The JASCL source was audited only; no external baseline training was authorized after the hard stop.

## 19. Prostate

Not executed.

## 20. M&Ms

Not executed.

## 21. Run and evidence paths

- Remote project: `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg`
- Canonical feasibility report: `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/reports/experiment_status/SPARC_FEASIBILITY_AUDIT.json`
- Per-seed and combined analysis: `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/reports/analysis/sparcseg_v0_1/`
- Runtime logs: `/home/jiangsuiyang/SSL_CL/runs/sparcseg_v0_1_feasibility_logs/`
- Frozen failure bundle: `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/reports/failure_bundles/sparc_v0_1_posthoc_boundary_20260828T234044/`
- Visible/post-hoc seed assignment: seed 0 physical GPU 5, seed 1 GPU 6, seed 2 GPU 7; recorded UUIDs are in each seed summary.
- No SPARC training-run directory exists because no training run was launched.

## 22. All unexecuted items

The following were deliberately not executed: SPARC method registration; SPARC training configs; Stage B method/equivalence suite; one-batch and trajectory S0 equivalence; optimizer-based S0-S5 pilots; seed-1 independent pilot; seed-0 full run; mask shuffle; multi-seed training; training cost analysis; external/JASCL baseline; Prostate; M&Ms; and every downstream gate or report dependent on those runs.

## 23. Claim boundary

This report claims only the completed freeze, source/path audits, three-seed/two-transition feasibility diagnostics, tests, and the registered hard-stop outcome. It does not claim SPARC training effectiveness, S0 equivalence, continual-learning gains, external generalization, runtime overhead, or any result for an unexecuted stage.
