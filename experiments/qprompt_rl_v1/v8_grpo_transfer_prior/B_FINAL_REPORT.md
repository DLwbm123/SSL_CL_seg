# V8 final: first-domain prior failed the frozen screen

All 20 development endpoints were sealed before Q_dev readout. Both 64-group priors completed; no optimizer invocation failed. The frozen gate failed for both priors, so V8 ends here and C/D are NOT_RUN. Image-level exploration only; patient-independent generalization is not claimed.

| Method | New Dice (%) | Old Dice (%) | Utility × 100 |
|---|---:|---:|---:|
| NATIVE | 73.4741 | 81.2819 | -7.3256 |
| FIXED_BEST | 73.7624 | 81.0925 | -7.2268 |
| UNIFORM_ACTION | 73.7753 | 81.2891 | -7.0331 |
| PRE_FROZEN_601 | 73.6224 | 81.2629 | -7.2197 |
| PRE_FROZEN_602 | 73.7188 | 81.3024 | -7.1275 |

## Findings

Against UNIFORM_ACTION, prior 601 changes new/old Dice by -0.1529/-0.0262 percentage points; prior 602 by -0.0565/+0.0133. Both have lower mean utility than uniform. Against NATIVE the priors have small average new-Dice gains, but that does not establish learned control: matched random actions do better. FIXED_BEST also fails to beat uniform on average utility. Full per-context rim/cup scores are retained in B_DEVELOPMENT_RESULTS.json.

All five methods show absolute old-query deterioration against frozen clean-memory references (about 4–6 percentage points on the averaged old queries); a small relative difference is not prevention of forgetting. These are simulated conditions within one first domain, not target-domain or sealed-test evidence.

Actor updates were skipped in 46/64 and 48/64 groups (71.875%/75%). The learned distributions changed; this is neither a frozen actor nor an inactive transfer gate (mean gate ≈0.9914). The global 0.00202752 threshold was estimated from cross-stream absolute reward changes, although candidate rewards within a group use shared RNG. This suggests a calibration mismatch, not proof that a lower threshold is safe or useful. A separate preregistered round will test paired action-contrast noise; no V8 threshold is edited.

## Cost and delivery

This round used 78,829 student optimizer invocations (29 qualification + 8,000 auxiliary + 1,200 entries + 14,400 audit + 51,200 prior + 4,000 development) and 144 actor invocations (8 qualification + 136 prior). The historical shared source adds 8,000 separately documented student updates. All attempts succeeded except earlier assertion/launch failures outside optimizer invocations; the 10 consumed qualification steps from the repaired assertion are included.

Anonymous full group trajectories, all 20 development outcomes, A outcomes and the cumulative physical ledger are published. Images, labels, image-role mappings, checkpoints and private snapshots remain on NAS.

## Eight requested conclusions

1. Action signal passed in 4/8 A contexts, exclusively two-image support; eight-image contexts failed.
2. GRPO did not beat uniform in the frozen development screen.
3. It did not establish practical superiority over fixed structure weighting.
4. Target transfer versus cold start: NOT_RUN.
5. Online value/cost on targets: NOT_RUN.
6. Low-label target advantage: NOT_RUN.
7. Small relative retention differences coexist with absolute forgetting; no retention claim.
8. Do not deploy this pretrained controller on the present evidence.
