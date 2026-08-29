# CRISP-Seg V0.1 final preregistered report

**Final status:** `CRISP_ROLE_NOT_REPRODUCIBLE`  
**Interpretation:** preregistered feasibility hard stop; the split-half ranks were reproducible, but the composite Gate A failed because channel roles were non-degenerate in `0/6` pairs per layer.  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`  
**CRISP method/config created:** `no`  
**Pilot/full/external training:** `not authorized and not run`

## 1. Frozen boundary

`SPARC_V0_1_FROZEN_FOR_CRISP` passed. SPARC remains frozen at `SPARC_PAS_NOT_SUPPORTED`; its optimizer-step count is zero, PAS/prototype spatial gating is not continued, `STOP_NEW_RELATION_METHODS` remains binding, and uniform historical relation KD remains unchanged. All seed manifests, splits, and nine R0 site-end checkpoints were hashed without modification.

## 2. Source audit

`CRISP_SOURCE_AUDIT_PASSED`.

- LAG official repository commit: `3dbd3f8dcee06770aa9c4412f447147b2520c71f`.
- STAR official repository commit: `6c9203aa0c91e9a2d4e40664c97754ae02226675`.
- DC²T primary DOI: `10.1109/TMI.2024.3469528`; no official implementation was located, so only publication-level provenance was used.
- CRISP's case-equal content/style scores, continuous roles, and dual IFC/PFC allocation are protocol-specific adaptations, not direct reproduction.

## 3. Model path and style probe

- `CRISP_MODEL_PATH_REVALIDATION_PASSED`: U-Net parameter count `507,731`; state-dict keys constant; logits, relation features, `dec3`, and `dec1` all had max-absolute difference `0.0` from the frozen golden; decoder storage identity passed.
- `CRISP_STYLE_PROBE_AUDIT_PASSED`: contract SHA256 `7d9c2f9a43b0a6c688de8c9543713bd78779231a32f7d4669d98ffd560399ec3`; same deterministic geometry; existing contrast/brightness/noise only; no cutout, new augmentation, hidden label, pseudo-label, model, or optimizer interface.

## 4. Gate A — role non-degeneracy and reproducibility

Split-half reproducibility passed strongly:

| Layer | Median Spearman | Top-quartile Jaccard | Bottom-quartile Jaccard |
|---|---:|---:|---:|
| dec3 | 0.991094 | 0.882353 | 0.882353 |
| dec1 | 0.971879 | 0.800000 | 0.600000 |

Non-degeneracy failed in every pair:

| Layer | Joint pass | Mean-alpha range | Median alpha IQR | Median ESS(alpha)/D |
|---|---:|---:|---:|---:|
| dec3 | 0/6 | 0.078989–0.130371 | 0.083096 | 0.202774 |
| dec1 | 0/6 | 0.001521–0.126093 | 0.001492 | 0.190356 |

Every mean alpha is below the preregistered lower bound `0.20`. Roles are therefore stable but overwhelmingly plastic, with particularly collapsed `dec1` allocations.

## 5. Gate B — independent held-out content/style validation

| Layer | Median Fisher top/bottom | Positive pairs | Median style plastic/stable | Min top-beta alive | Result |
|---|---:|---:|---:|---:|---|
| dec3 | 54.009930 | 6/6 | 9.631086 | 1.00 | content/style pass |
| dec1 | 4.012726 | 3/6 | 2,281,696.094981 | 0.50 | content/style fail |

The enormous `dec1` style ratio is not evidence of a useful role: its denominator is near zero and only half of selected top-beta channels are active in the worst pair. `dec1` also fails the required `4/6` positive Fisher comparisons and foreground-completeness check.

## 6. Gate C — gradient scale

| Quantity | Median | p10 | p90 | Gate |
|---|---:|---:|---:|---|
| IFC / relation | 0.648002 | 0.193734 | 1.524457 | pass |
| PFC / assimilation | 3.898486 | 1.689159 | 9.826043 | fail |
| C3 / C0 total | 3.019627 | — | 7.166891 | fail |

All `192/192` rows were finite and old-model `.grad` was zero. The failure is excessive PFC and total-gradient scale, not numerical non-finiteness.

## 7. Gate D — stateless virtual step

- C3 vs C0: fail; previous-loss better fraction `0.5625 < 0.60`.
- IFC contribution, C3 vs C2: fail; median previous-loss margin `3.91e-5 < 1e-4` despite `0.875` better fraction.
- PFC contribution, C3 vs C1: pass.
- Continuous C3 vs hard C4: fail; C4 has better median previous loss.
- Continuous C3 vs uniform C5: fail; C5 has better median previous and current losses.

All `1,152/1,152` virtual rows were finite and stateless; model/checkpoint and role-state hashes remained unchanged.

## 8. Diagnosed attempt-1 engineering failure

The first functional attempt stopped before writing formal functional artifacts because constant/dead channel maps produced an undefined zero-norm derivative in PFC. The failure bundle is preserved. A zero-vector subgradient of zero was added without changing any forward value; `13` targeted tests passed, a one-batch comparison reproduced all pre-repair C0–C5 forward totals exactly, and all six gradient norms became finite. Attempt 2 then completed all three seeds successfully.

## 9. Execution and evidence

- Physical GPUs: seed0/1/2 on GPU `5/6/7` respectively.
- Local remote run root: `/home/jiangsuiyang/SSL_CL/runs/crispseg_v0_1_feasibility_audit`.
- Coverage: `12` role rows, `12` reproducibility rows, `12` held-out rows, `192` gradient rows, `1,152` virtual rows.
- Main report SHA256: `73a4fdbe1dac8d4434dd187c9f03a9c0ca5dc57f9ddc071fdd47ec833885cb69`.
- Analysis hashes are recorded in `CRISP_FEASIBILITY_AUDIT.json`.
- NAS sync: confirmed at `/data_nas/jiangsuiyang/LCR-Seg/crispseg_v0_1`; the non-overwriting initial bundle contained `87` files (`2.2 MB`) and passed full SHA256 verification. Initial manifest SHA256: `59bd68ed568d94986a0e1cac05e5e3b5162dd3da801b349b445c642b13d2f353`.

## 10. Protocol stop

Because the exact status is not `CRISP_FEASIBILITY_SUPPORTED`, no `lcrseg/methods/crispseg_v0_1.py`, CRISP configuration, C0 equivalence run, pilot, full run, shuffle control, multiseed training, external baseline, Prostate, or M&Ms experiment exists. Promoting C1, C2, C4, or C5 from intermediate evidence is prohibited. This protocol is complete at the hard stop.
