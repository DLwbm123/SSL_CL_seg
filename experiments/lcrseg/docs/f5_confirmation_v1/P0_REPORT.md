# P0: read-only historical audit

Seed 161 is DEVELOPMENT; seed 162 is ALREADY_OBSERVED. No new confirmation results exist.
Existing aggregate receipts only; no patient payload, model load, re-evaluation, or optimizer update.

## O2 old-domain breakdown

| Seed | Method | Domain | Early | Final | Early − final | F5 − B0 final |
|---:|---|---|---:|---:|---:|---:|
| 161 | B2_PARENT_PAS_KL | REFUGE | 0.823157 | 0.721326 | +0.101831 | -0.049622 |
| 161 | B2_PARENT_PAS_KL | Drishti_GS | 0.714356 | 0.490028 | +0.224328 | -0.019192 |
| 161 | F5 | REFUGE | 0.823157 | 0.672289 | +0.150868 | -0.049622 |
| 161 | F5 | Drishti_GS | 0.698500 | 0.448685 | +0.249815 | -0.019192 |
| 161 | B0_PARENT_LCTX | REFUGE | 0.823157 | 0.721911 | +0.101246 | -0.049622 |
| 161 | B0_PARENT_LCTX | Drishti_GS | 0.699657 | 0.467877 | +0.231780 | -0.019192 |
| 162 | B0_PARENT_LCTX | REFUGE | 0.831168 | 0.707056 | +0.124112 | +0.001352 |
| 162 | B0_PARENT_LCTX | Drishti_GS | 0.714451 | 0.446507 | +0.267944 | -0.004133 |
| 162 | F5 | REFUGE | 0.831168 | 0.708408 | +0.122760 | +0.001352 |
| 162 | F5 | Drishti_GS | 0.727742 | 0.442374 | +0.285368 | -0.004133 |

## Interpretation and missing evidence

The per-domain rows expose compensating effects; a positive overall Final is not a claim that every old domain improved.
F5 seed161 two-order ΔFinal vs B0 = -0.000113407; seed162 = +0.013952836 (descriptive, already observed).
B2 C06 development mean Final = 0.658917919. B2 seed162 has no historical target receipts; it is a new execution arm.
P0_EVIDENCE.json preserves saved timelines, complete resolved_options and available stage-entry spectral summaries.
Unrecorded full histories, post-training spectrum, SWD/KL decomposition, and patient-independent evidence = NA.
SOURCE_REUSE.json verifies metadata/file presence only; tensor schema/hash/forward remain PENDING.
No inference gate is evaluated before the complete P1 matrix. No performance-based early stop or extra seed is allowed.
