# BPRC-X1 exploratory protocol

**Protocol ID:** `bprcseg_x1_exploratory`  
**Created before execution:** `2026-08-28T14:43:55Z`  
**Authorization:** direct user request after the frozen BPRC V0.1 hard stop  
**Epistemic status:** exploratory; not a BPRC V0.1 result and not preregistered by the original plan

## Immutability boundary

- All BPRC V0.1 source snapshots, reports, CSV files, STOP artifacts, and the NAS package remain unchanged.
- X1 writes only to `reports/analysis/bprcseg_x1_exploratory` and `BPRC_X1_*` status artifacts.
- Frozen R0/TARC checkpoints, datasets, batches, temperatures, weights, schedules, validation functions, and random seeds are reused read-only.
- No optimizer step or checkpoint mutation is permitted during the diagnostic stage.

## Single candidate

`X0` is the exact frozen B0 categorical pixel-mean control.

`X1` is the existing B2 top-2 Bernoulli pairwise class-balanced loss divided by the fixed number of semantic classes:

```text
X1 = B2 / C
C = 3
```

The factor `1/C` is fixed analytically before execution. It is not fitted from observed outcomes. No other candidate, temperature, lambda, GT weighting, pair selection, or post-hoc rescaling is allowed in this run.

## Fixed evidence

- seeds: `0,1,2`;
- transitions: `REFUGE->RIM_ONE_r3`, `RIM_ONE_r3->Drishti_GS`;
- update batches: `32` per transition and seed;
- previous validation batches: `16`;
- current validation batches: `16`;
- virtual step norm: `1e-3`;
- fixed batch construction and exact TARC/R0 evaluators: reused from the completed BPRC V0.1 audit.

## Frozen diagnostic gates

All gates must pass before a training pilot is allowed.

1. Gradient scale:
   - median X1/X0 ratio in `[0.5,2.0]`;
   - p10 `>=0.25`;
   - p90 `<=4.0`;
   - non-finite count `0`.
2. Previous-site utility:
   - X1 previous loss delta lower than X0 in at least `60%` of paired comparisons;
   - median X1 delta `<=` median X0 delta minus `1e-4`.
3. Current-site safety:
   - median X1 current loss delta `<=` X0 plus `2% * abs(X0)`;
   - median X1 current Dice delta `>=` X0 minus `0.002`.
4. Disc-rim margin:
   - X1 improves disc-rim agreement versus X0 in at least `60%` of comparisons;
   - median paired improvement `>=+0.005`;
   - no class median paired delta below `-0.005`.

If any gate fails, the exploratory run stops without method registration or training. If all pass, only a separate seed-0 1000-step pilot may be implemented and launched.
