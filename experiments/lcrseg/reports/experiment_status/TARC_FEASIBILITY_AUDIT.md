# TARC-Seg V0.1 feasibility audit

**Final status:** `TARC_RELATION_FIDELITY_NOT_SUPPORTED`  
**Part B authorized:** `false`  
**Optimizer steps:** `0`

## Engineering boundary

- ASPR freeze and TARC relation-space audit passed before this audit.
- Transport used only current-site `train_labeled` cases and all classes `0,1,2`, including background.
- Previous/current validation labels were used only for frozen post-hoc evaluation.
- Functional virtual updates used stateless parameter views; model SHA-256 values were unchanged.
- No hidden GT, unlabeled prototype memory, site modes, optimizer update, checkpoint mutation, or TARC training method was used.

## Gate A — all-class anchor transport: PASS

- median class-minus-static cosine: `0.824633` (required >= 0.020)
- class better than static: `100.00%` (required >= 70%)
- p10 improvement: `0.251887` (required >= -0.020)
- background median improvement: `0.943351` (required >= 0.010)
- positive-seed counts by class: `{'0': 3, '1': 3, '2': 3}`

## Gate B — historical relation fidelity: FAIL

- median KL reduction: `96.60%` (required >= 10%)
- class KL lower than static: `100.00%` (required >= 70%)
- median top-1 agreement improvement: `0.899517` (required >= 0.010)
- classwise median margin-agreement deltas: `{'0': 0.006729375136203808, '1': -0.03095356927428755, '2': 0.025737242190682075}`
- failure: class 1 median margin-agreement delta is below `-0.010`.

## Gate C — current-site safety: FAIL

- minimum accuracy delta vs static: `0.023766` (required >= -0.005)
- minimum margin delta vs static: `-0.011178` (required >= -0.010)
- non-finite count: `0`

## Gate D — functional virtual step: FAIL

- T3 better than T0 on previous-val delta: `39.58%` (required >= 60%)
- median previous delta T0/T2/T3: `-0.00167272` / `-0.00106583` / `-0.00165457`
- median current delta T0/T3: `-0.00047178` / `-0.00052739`

## Protocol decision

Gate B is the first failed research gate, so the exact preregistered state is `TARC_RELATION_FIDELITY_NOT_SUPPORTED`. Gate C and Gate D also fail independently. Part B is prohibited: no TARC loss/method/config, equivalence bridge, pilot, or full run may be implemented or launched under V0.1.

## Canonical artifacts

- `reports/analysis/tarcseg_v0_1/anchor_transport_audit.csv`
- `reports/analysis/tarcseg_v0_1/relation_fidelity_audit.csv`
- `reports/analysis/tarcseg_v0_1/virtual_step_audit.csv`
- `reports/experiment_status/TARC_FEASIBILITY_AUDIT.json`
