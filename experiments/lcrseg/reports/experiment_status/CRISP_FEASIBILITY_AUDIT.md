# CRISP-Seg V0.1 feasibility audit

**Status:** `CRISP_ROLE_NOT_REPRODUCIBLE`  
**Optimizer steps:** `0`  
**Hidden-GT usage:** `none`  
**Method/config registration:** `none`

## Gate A — channel roles

- `dec3`: nondegenerate `0/6`; median Spearman `0.991094`; top/bottom Jaccard `0.882353` / `0.882353`; pass `False`.
- `dec1`: nondegenerate `0/6`; median Spearman `0.971879`; top/bottom Jaccard `0.800000` / `0.600000`; pass `False`.

## Gate B — independent validation

- `dec3`: Fisher ratio `54.009930` (6/6 positive); style ratio `9.631086` (6/6 positive); content/style pass `True` / `True`.
- `dec1`: Fisher ratio `4.012726` (3/6 positive); style ratio `2281696.094981` (6/6 positive); content/style pass `False` / `False`.

## Gate C — gradient scale

- IFC ratio median/p10/p90: `0.648002` / `0.193734` / `1.524457`; pass `True`.
- PFC ratio median/p10/p90: `3.898486` / `1.689159` / `9.826043`; pass `False`.
- C3/C0 total median/p90: `3.019627` / `7.166891`; nonfinite `0`; pass `False`.

## Gate D — stateless virtual step

- D1 C3 vs C0: `False`
- D2 IFC contribution: `False`
- D3 PFC contribution: `True`
- D4 continuous vs C4/C5: `False` / `False`

No CRISP method or training configuration was registered unless and until the exact supported status is emitted.
