# R1.5 final mechanism and development interpretation

**Status: R1_5_DEVELOPMENT_NOT_MET.** All 20 endpoints and 20,000 valid suffix updates completed. Confirmation was not started. Original R1 results remain unchanged.

## Answers to the 15 registered questions

1. **Original monopolization recurs.** At the frozen prefixes, the fraction of images with >6 queries assigned to one prototype was 0.700–1.000 across the four cells (800 images in 400 paired batches). This operational definition does not by itself establish the cause of segmentation errors.
2. **CC enforces capacity.** Every forensic CC assignment satisfied registered supported-class bounds. Every A3/A4 diagnostic at the 100-step cadence had all three groups within 3–6. The production solver enforces these constraints at each call; checkpoint audit passed.
3. **Singleton degeneration decreases.** Original mean batch singleton fractions ranged from 0.0500 to 0.6600; CC was zero in all four forensic cells.
4. **Assignment regret is positive.** Mean cosine regret was DINO/Drishti 0.015624, DINO/RIM 0.014045, UNet/Drishti 0.081033, UNet/RIM 0.055557. FORENSIC_ROUTING.csv includes every batch mean/p50/p95/max. These summaries are means of batch statistics, not pooled query percentiles.
5. **Seg–GRQA gradient cosine does not improve at the prefixes.** Mean original→CC cosine: DINO/Drishti 0.027064→−0.013374; DINO/RIM 0.017326→−0.032386; UNet/Drishti 0.169733→0.093303; UNet/RIM 0.057300→0.028801. CC reduces it in all four cells. This is a fixed-state diagnostic, not a training causal proof.
6. **No evidence that warm-up mainly reduces early clipping.** At local steps 100/200/300, A2 clipping exceeds A1 and A4 exceeds A3 for both backbones. See the table below; logging does not cover every early step.
7. **Routing-only benefit is backbone-dependent.** A3−A1 equal-domain macro is +0.006015 for UNet and −0.000233 for DINO. A3 still trails A0 on both backbones. Capacity repair alone is not a consistent segmentation improvement.
8. **Warm-up-only benefit is not consistent.** A2−A1 is +0.001226 for UNet and −0.000097 for DINO.
9. **No consistent interaction gain.** Macro interaction is −0.002245 for UNet and +0.000229 for DINO. Rim/cup and both main effects are fully reported in FACTOR_EFFECTS.csv.
10. **A4−A0 per cell:** UNet/RIM −0.000938; UNet/Drishti −0.002535; DINO/RIM −0.000780; DINO/Drishti +0.000546.
11. **Class integrity is mixed.** UNet equal-domain rim/cup deltas are −0.003004/−0.000469; DINO deltas are +0.000441/−0.000675. No individual A4−A0 domain/class drop exceeds 0.01. The preregistered conservative nonnegative class-mean rule is not met.
12. **Development gate fails.** Both backbone macro means are below +0.003 and only 1/4 cells improves (required ≥3/4). Failure is decisive even without the conservative class-mean rule. Worst cell −0.002535 remains above the −0.003 floor.
13. **Confirmation was not entered.** Its conditional 32,000-update budget was never enabled.
14. **Confirmation success is not measured.** No seed262/263 confirmation result is claimed.
15. **One Class E patch; no scientific-path patch.** E0 `faa023324ece2294c4bd8fc855043500fc787985` completed zero-update forensic. Diagnostic field collisions caused 15 recorded training exceptions before step100 diagnostics. E1 `2205de41b64a7eb263e72c2d4254d222ff813d4a` fixed logging and permitted explicitly audited checkpoint resumes. Training AST excluding dictionary logging edits was unchanged; CPU contracts and zero-update full CUDA state roundtrip passed. All final students use E1. Original checkpoint source commits are retained; publication commits are separate.

## Accounting and state audit

- Forensic: 400 student forwards, 400 reference forwards, 1,600 autograd.grad calls, zero optimizer calls. Bank-initialization forwards are setup, outside these diagnostic counts.
- Development: 21,564 physical optimizer attempts and commit events, including 1,564 recovery/replay calls; 20,000 valid final-path updates. Replay stayed below the 2,000 cap.
- Optimizer-call failures: 0. This does **not** mean no engineering failures: 15 Python logging exceptions occurred; three still-running affected workers were stopped during repair to avoid further known failures.
- All 20 checkpoint student tensors finite and readable; optimizer, scheduler and data cursor at global step3000; RNG present; prefix bindings match; deployment equality passed20/20; 200 training diagnostic records.
- No missing endpoints, unauthorized tuning, confirmation training, R2 or R3. Private raw logs, checkpoints, per-case schedules and receipts remain on the server.

## Early clipping (mean fraction over two domains at steps 100/200/300)

| Backbone | A1 | A2 | A3 | A4 |
|---|---:|---:|---:|---:|
| DINOV2_VITS14_QUERY | 0.159722 | 0.250000 | 0.173611 | 0.180556 |
| UNET_QUERY_128 | 0.020833 | 0.076389 | 0.027778 | 0.173611 |

## Contemporaneous reproduction

R1_REPRODUCTION.csv compares A0 with old Q1 and A1 with old Q3 without changing the old reports. The maximum absolute macro discrepancy is 0.000232125. Fixed seeds and data schedules do not guarantee bitwise CUDA reproducibility under warn-only deterministic behavior. The contemporaneous A0/A1 controls are the registered comparison basis.

| Backbone | Domain | Comparison | Macro delta versus R1 |
|---|---|---|---:|
| DINOV2_VITS14_QUERY | Drishti_GS | A0−Q1 | -0.000122374 |
| DINOV2_VITS14_QUERY | Drishti_GS | A1−Q3 | -0.000232125 |
| DINOV2_VITS14_QUERY | RIM_ONE_r3 | A0−Q1 | -0.000004405 |
| DINOV2_VITS14_QUERY | RIM_ONE_r3 | A1−Q3 | -0.000049524 |
| UNET_QUERY_128 | Drishti_GS | A0−Q1 | +0.000000000 |
| UNET_QUERY_128 | Drishti_GS | A1−Q3 | +0.000000000 |
| UNET_QUERY_128 | RIM_ONE_r3 | A0−Q1 | +0.000000000 |
| UNET_QUERY_128 | RIM_ONE_r3 | A1−Q3 | +0.000000000 |

## Scope

CC removes the registered routing monopolization in these observations, but its gradient alignment and segmentation utility do not consistently improve. This negative development result does not support advancing the fixed candidate to confirmation. One development seed on exposed validation supplies neither independent-patient confirmation nor statistical significance. Shared-GPU latency is descriptive, not a controlled speed comparison.
