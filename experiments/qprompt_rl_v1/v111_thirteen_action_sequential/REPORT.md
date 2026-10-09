# V111 complete: OFF expansion did not establish an RL advantage

All264 trajectories,528 snapshots,26 jobs and the coordinator completed. The frozen decision is **V111_NO_PRACTICAL_SEQUENTIAL_RL_GAIN**. The original gate was independently recomputed; the new absolute-benefit diagnostic does not change it.

| Method | New Dice | Old Dice | Utility | Weighted new gain |
|---|---:|---:|---:|---:|
| SAMPLE_WARM | 0.739317934 | 0.815659972 | -0.066091284 | -0.023664045 |
| SAMPLE_CE | 0.739190058 | 0.816091042 | -0.065523717 | -0.023527548 |
| SAMPLE_RL | 0.740404194 | 0.816447399 | -0.064256759 | -0.022616947 |
| SAMPLE_DISTILL | 0.740404194 | 0.816447399 | -0.064256759 | -0.022616947 |
| UNIFORM | 0.728668648 | 0.810356344 | -0.079934191 | -0.032203324 |
| GLOBAL | 0.729918499 | 0.808665030 | -0.080683416 | -0.031261236 |
| TIME | 0.735346259 | 0.815460450 | -0.069888491 | -0.027261731 |
| RIDGE | 0.747258840 | 0.818882407 | -0.056509405 | -0.017304601 |
| NN | 0.745174566 | 0.818927987 | -0.058027030 | -0.018867807 |
| OFF | 0.736987304 | 0.815943579 | -0.068174579 | -0.026030947 |

Sampled RL improves original CE by 0.121414 percentage points new Dice and 0.035636 points old Dice, below the practical thresholds. Mean utility equals target-matched CE exactly; paired RL-minus-targetCE utility is +0.000281397408 for seed601 and −0.000281397408 for seed602. Equal aggregate outcomes do not mean identical per-seed trajectories. RL is below Ridge by0.006854647 new Dice and0.007747354 utility, and below1NN by0.004770372 new Dice and0.006229728 utility. Neither sampled-primary nor a secondary argmax result can rescue the failed gate.

Mean ENTRY100 new Dice is 0.767029988. Sampled RL finalnew−ENTRY100 is -2.662579 percentage points; its weightednewgain is -2.261695 points. OFF also remains negative (-2.603095 weighted points). V112 separately measures missing ENTRY100 old scores before comparing old retention with this entry. V111 old-reference differences use the memory model, not the ENTRY100 student.

## Historical paired comparison

All240 shared method/context/stream keys were paired with V108 without new queries. The contrast jointly changes action space, OFF-prefix coverage and state weighting; it cannot isolate an RL mechanism. UNIFORM13 differs from UNIFORM12. Full positive and negative paired rows and per-seed/stream comparisons are published.

| Method | New difference vs V108, pp | Old difference, pp | Utility difference |
|---|---:|---:|---:|
| GLOBAL | 0.123176 | -0.358965 | -0.002218488 |
| NN | 0.914339 | 0.365010 | 0.010996419 |
| RIDGE | 1.274601 | 0.494309 | 0.014935640 |
| UNIFORM | 0.047204 | 0.029089 | 0.000698619 |
| SAMPLE_WARM | 0.393366 | 0.111729 | 0.004207999 |
| SAMPLE_CE | 0.359791 | 0.140985 | 0.004524099 |
| SAMPLE_RL | 0.423938 | 0.139149 | 0.004849336 |
| SAMPLE_DISTILL | 0.422055 | 0.140004 | 0.004845798 |

## Verification and delivery

All56,905 root/child optimizer attempt-success pairs matched:52,808 nativeupdates and4,097 actorupdates; zero failedcalls. One data ridge solve,2,128 stableprobes and792 querycalls/3,168 Q_devimageevaluations were recorded. All264 reward formulas and all528 action/probability traces passed. All trajectories finished before the endpoint barrier and every evaluation job started afterward. The original nine-control gate and both-seed comparisons were independently reproduced. Every OFF update used the zero-U-read guard. No additional queries were used for the audit.

Cumulative throughV111:769,409 nativeupdates excluding shared source training (777,409 including8,000 sourceupdates),181,459 actorupdates,57 linear solves (53data,4synthetic). Prior failures and negative findings remain accounted for.

Next is the separately frozen V112 ENTRY100 absolute-benefit diagnostic:16 old-image evaluations,zero optimizerupdates. Its complete readout controls admission to the conditional quarter-rate V113; no V111 endpoint, seed or threshold was changed. These are repeatedly viewed shared-source development findings, not independent patient/domain confirmation.

Public artifacts include source, protocol, all264 scalarrows,240 historical pairedrows,528 action/probability records without private statefeatures, costs, audits and this report. Private images, features, checkpoints and raw logs remain on NAS.
