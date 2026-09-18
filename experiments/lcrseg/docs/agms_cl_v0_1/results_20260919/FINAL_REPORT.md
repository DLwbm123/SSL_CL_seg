# AGMS P1: complete development results

Execution commit: `89dc7657c4f0b7c6f51de339d0f6bc2a621da55f`. Status: **COMPLETE_AGMS_P1_AWAITING_SCIENTIFIC_REVIEW**.

All 10 new second-target stages completed and actual-model integrity receipts are VERIFIED. Two A0 endpoints are historical imports. The frozen single-seed (163), two-order matrix was completed without pruning or scientific configuration changes.

## Main result

Both frozen advancement gates failed: **P_perf=false, P_joint=false**. A5's two-order mean Final is 0.639210, versus A0 0.657466, a difference of -0.018255 (about -1.83 percentage points). The strongest simple control by mean Final is A1; A5 minus A1 is -0.018276. This is a negative development result, not an engineering failure.

Orders are averaged within seed before comparison. Dice values use the frozen rim/cup macro definition, not disc-union Dice.

| Arm | Final | Old | Incoming | Forget | Final minus A0 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.657466 | 0.626902 | 0.718594 | 0.145186 | +0.000000 |
| A1 | 0.657486 | 0.626885 | 0.718689 | 0.145203 | +0.000020 |
| A2 | 0.638859 | 0.598992 | 0.718593 | 0.173096 | -0.018607 |
| A3 | 0.639280 | 0.599543 | 0.718755 | 0.172545 | -0.018186 |
| A4 | 0.639172 | 0.599376 | 0.718764 | 0.172712 | -0.018294 |
| A5 | 0.639210 | 0.599444 | 0.718743 | 0.172644 | -0.018255 |

A5-A0 in O1: Final -0.029227, Old -0.050504, Incoming +0.013328. In O2: Final -0.007284, Old -0.004412, Incoming -0.013029. Both order-specific Final conditions fail; O2 also fails Old and Incoming conditions. Full per-order results and all nine contrasts are retained in the CSV files.

## Mechanism evidence and limits

H alone (A1) is numerically close to A0 in both orders. M alone (A2) reduces mean Final by about 1.86 points and mean Old by about 2.79 points, with essentially zero mean Incoming gain: the O1 gain reverses in O2. Adding H or multiscale gating does not recover this deficit. A5 minus A4 mean Final is only +0.00003842.

A5 risk weights are nonuniform, so the adaptive branch operated. However, all eight saved A5 training-L shadow diagnostics show zero uniform-versus-risk selection disagreements. This does not measure disagreement on U or at every update. Its admitted coarse fraction is 2.035% in O1 and 0.359% in O2; sparse median weighted H/supervised AB gradient norms are approximately 0.211% and 0.100%, compared with DS/supervised 15.024% and 22.652%. These observations motivate testing DS strength, but do not prove gradient magnitude caused forgetting.

P0 is a confidence-only screening diagnostic, not PAS: 26 current-L images, zero updates. Parent accuracy among selected pixels is 1.0000 in O1 and 0.7258 in O2. Training-L diagnostics are not independent calibration and do not establish correctness of selected U pixels.

## Completion and cost

- Formal scientific and physical optimizer calls: **26,500 / 26,500**; ten closed PASS formal sessions, zero failed formal sessions.
- Real L-only smoke: **24**; total real-data optimizer calls **26,524**.
- Native synthetic CUDA: **36**, including two prescribed after-optimizer failure injections. These are charged physical calls; no unexpected CUDA qualification failure.
- Current study CPU preparation: **80** calls over four attempts, including retained failed attempts and the explicit repair authorization. Earlier-study CPU **134** is separate. R1 regression: zero optimizer calls. None rerun for this release.
- Extra diagnostic VJPs: **160** at **40** points. Reports: **12 Final / 36 domain / 27 paired** rows.
- Formal summed worker time, including endpoint evaluation: **6.7525 hours**. This is not exclusive GPU time. Full operation counts, model acceptance/prefix/P0/qualification sessions and memory peaks are preserved in COST_AND_COMPLETION.json.

## Interpretation and next question

This is one observed seed and two orders, not independent-seed confirmation, independent-patient generalization, original KI reproduction or SOTA. Final development evaluations have informed the next hypothesis; subsequent tuning on these same domains must not be presented as independent validation. No patient-level outcomes, images, labels, checkpoint payloads or private paths are published.

The next proposed discriminating test changes only A5 lambda_DS from 0.25 to 0.125, with the same original B2 prefixes and both second-target orders (2 nodes, 5,300 updates). It is a separately registered future development study, not part of these completed costs and not evidence of improvement. Reducing DS also changes auxiliary-head learning; it is not a pure shared-feature gradient intervention. Its code/qualification authority must be independently bound before execution. The historical frozen protocol's no-automatic-follow-up state is preserved; separate subsequent work uses the user's later explicit delegation.
