# V126 source and continuation generalization drift

DIAGNOSTIC_COMPLETE: the largest held-image regression occurs before ENTRY100. This is evidence for an early generalization failure, not proof of a unique causal mechanism.

| Interval | Fit Dice change pp | Held Dice change pp | Old Dice change pp | Fit loss down / held loss up |
|---|---:|---:|---:|---:|
| SOURCE→100 | +6.508788 | -5.712658 | +2.973362 | 20/32 |
| 100→150 BASE | -0.486775 | -0.023948 | +0.777812 | 3/32 |
| 100→300 BASE | -0.719670 | -0.128154 | +0.901870 | 5/32 |
| 100→150 OFF | +0.302905 | -0.597154 | -0.281176 | 20/32 |
| 100→300 OFF | +0.616455 | -1.075345 | -0.563970 | 24/32 |

SOURCE→ENTRY100: fit native supervised loss −0.201283861, held +0.075429646; fit improves while held Dice falls in27/32 units. The historical ENTRY safety reference already includes this deterioration. Early training therefore needs attention before further tweaking100→300. Frozen SOURCE and frozen ENTRY will be explicit distinct future safety controls, not evidence of learning.

BASE100→300 is not ordinary pooled fit-loss overfitting: fit loss increases0.002255997 while held loss decreases0.022111675 despite held Dice decreasing0.128154pp. CE/Dice objectives and clean/mixed forward settings must not be conflated. OFF100→300 more clearly widens the fit/held gap: fit loss−0.014515224, held+0.021617952, with24/32 opposite loss directions. U remains beneficial relative to OFF. This does not establish cross-image mixing as the cause; the next experiment instead tests whether retaining the frozen source teacher from the beginning avoids the much larger early drift.

All16 contexts and both streams remain. These are correlated, repeatedly observed first-domain photometric development images, not independent patients/datasets or later-task continual-learning confirmation. Two directions of the fold split each have only4 held images. The auxiliary source was fit only on M_fit16 and its held/fit readouts were newly measured here. Clean native supervised loss uses its own1e-5 Dice smoothing; reported native soft-Dice uses historical+1 smoothing.

## Full subgroup readout

| Group | Interval | Fit pp | Held pp | Old pp | Loss gap cases |
|---|---|---:|---:|---:|---:|
| fold=0 | ENTRY100 | +5.948505 | -9.856743 | +1.738032 | 13/16 |
| fold=0 | BASE150 | -0.468359 | +0.378526 | +1.026095 | 0/16 |
| fold=0 | BASE300 | -0.825869 | +0.375661 | +1.221355 | 0/16 |
| fold=0 | OFF150 | -0.063352 | -0.582735 | -0.094676 | 5/16 |
| fold=0 | OFF300 | +0.113948 | -1.005782 | -0.108006 | 8/16 |
| fold=1 | ENTRY100 | +7.069070 | -1.568572 | +4.208692 | 7/16 |
| fold=1 | BASE150 | -0.505191 | -0.426423 | +0.529528 | 3/16 |
| fold=1 | BASE300 | -0.613470 | -0.631970 | +0.582384 | 5/16 |
| fold=1 | OFF150 | +0.669163 | -0.611572 | -0.467676 | 15/16 |
| fold=1 | OFF300 | +1.118961 | -1.144907 | -1.019934 | 16/16 |
| stream=3 | ENTRY100 | +6.411515 | -5.853328 | +2.954549 | 11/16 |
| stream=3 | BASE150 | -0.386720 | -0.051891 | +0.774564 | 2/16 |
| stream=3 | BASE300 | -0.663774 | -0.261091 | +0.867214 | 4/16 |
| stream=3 | OFF150 | +0.397987 | -0.654858 | -0.274091 | 11/16 |
| stream=3 | OFF300 | +0.720591 | -1.202395 | -0.577586 | 12/16 |
| stream=4 | ENTRY100 | +6.606060 | -5.571987 | +2.992175 | 9/16 |
| stream=4 | BASE150 | -0.586830 | +0.003994 | +0.781059 | 1/16 |
| stream=4 | BASE300 | -0.775566 | +0.004782 | +0.936526 | 1/16 |
| stream=4 | OFF150 | +0.207823 | -0.539450 | -0.288262 | 9/16 |
| stream=4 | OFF300 | +0.512318 | -0.948294 | -0.550354 | 12/16 |
| source_step=2000 | ENTRY100 | +5.287599 | -6.978538 | +2.029324 | 13/16 |
| source_step=2000 | BASE150 | -0.215040 | +0.027143 | +0.867002 | 1/16 |
| source_step=2000 | BASE300 | -0.238779 | -0.124595 | +0.936868 | 2/16 |
| source_step=2000 | OFF150 | +0.452151 | -0.580838 | -0.252843 | 13/16 |
| source_step=2000 | OFF300 | +0.921988 | -1.044700 | -0.556182 | 16/16 |
| source_step=8000 | ENTRY100 | +7.729977 | -4.446778 | +3.917400 | 7/16 |
| source_step=8000 | BASE150 | -0.758510 | -0.075040 | +0.688622 | 2/16 |
| source_step=8000 | BASE300 | -1.200560 | -0.131713 | +0.866871 | 3/16 |
| source_step=8000 | OFF150 | +0.153660 | -0.613469 | -0.309510 | 7/16 |
| source_step=8000 | OFF300 | +0.310922 | -1.105989 | -0.571758 | 8/16 |
| labeled_images=2 | ENTRY100 | +7.428029 | -5.823237 | +1.663484 | 10/16 |
| labeled_images=2 | BASE150 | -0.518512 | +0.301765 | +1.044457 | 0/16 |
| labeled_images=2 | BASE300 | -0.824504 | +0.436437 | +1.314034 | 0/16 |
| labeled_images=2 | OFF150 | -0.017706 | -0.735013 | -0.304954 | 9/16 |
| labeled_images=2 | OFF300 | +0.209594 | -1.136875 | -0.782792 | 12/16 |
| labeled_images=4 | ENTRY100 | +5.589546 | -5.602078 | +4.283240 | 10/16 |
| labeled_images=4 | BASE150 | -0.455038 | -0.349662 | +0.511166 | 3/16 |
| labeled_images=4 | BASE300 | -0.614835 | -0.692745 | +0.489706 | 5/16 |
| labeled_images=4 | OFF150 | +0.623517 | -0.459294 | -0.257399 | 11/16 |
| labeled_images=4 | OFF300 | +1.023316 | -1.013814 | -0.345148 | 12/16 |
| condition=brightness | ENTRY100 | +5.319654 | -5.524290 | +3.389873 | 11/16 |
| condition=brightness | BASE150 | -0.485275 | -0.194145 | +0.725552 | 3/16 |
| condition=brightness | BASE300 | -0.717076 | -0.331922 | +0.889922 | 3/16 |
| condition=brightness | OFF150 | +0.350160 | -0.617125 | -0.244246 | 11/16 |
| condition=brightness | OFF300 | +0.774923 | -1.055134 | -0.421144 | 12/16 |
| condition=contrast | ENTRY100 | +7.697922 | -5.901026 | +2.556851 | 9/16 |
| condition=contrast | BASE150 | -0.488275 | +0.146248 | +0.830072 | 0/16 |
| condition=contrast | BASE300 | -0.722263 | +0.075613 | +0.913817 | 2/16 |
| condition=contrast | OFF150 | +0.255651 | -0.577182 | -0.318107 | 9/16 |
| condition=contrast | OFF300 | +0.457986 | -1.095555 | -0.706795 | 12/16 |
| context=0 | ENTRY100 | +2.283128 | -11.668096 | +1.963838 | 2/2 |
| context=0 | BASE150 | +0.247134 | -0.048797 | +1.113672 | 0/2 |
| context=0 | BASE300 | +0.461645 | -0.111663 | +1.327542 | 0/2 |
| context=0 | OFF150 | +0.131511 | -0.701599 | +0.418956 | 1/2 |
| context=0 | OFF300 | +0.603529 | -0.941320 | +0.550939 | 2/2 |
| context=1 | ENTRY100 | +6.615619 | -13.836188 | +0.737461 | 2/2 |
| context=1 | BASE150 | -0.013296 | +0.719865 | +1.070160 | 0/2 |
| context=1 | BASE300 | +0.123196 | +0.748931 | +0.996489 | 0/2 |
| context=1 | OFF150 | -0.003967 | -1.088871 | +0.210477 | 1/2 |
| context=1 | OFF300 | +0.196125 | -2.251967 | -0.136375 | 2/2 |
| context=2 | ENTRY100 | +2.737965 | -10.769128 | +1.330684 | 2/2 |
| context=2 | BASE150 | -0.154733 | +0.201950 | +1.011966 | 0/2 |
| context=2 | BASE300 | -0.166110 | +0.213705 | +1.264347 | 0/2 |
| context=2 | OFF150 | +0.243199 | -0.267757 | -0.123966 | 2/2 |
| context=2 | OFF300 | +0.645330 | -0.278615 | +0.151663 | 2/2 |
| context=3 | ENTRY100 | +4.290591 | -12.631035 | +0.776623 | 2/2 |
| context=3 | BASE150 | -0.063088 | +0.851142 | +1.178720 | 0/2 |
| context=3 | BASE300 | -0.287076 | +0.480089 | +1.175462 | 0/2 |
| context=3 | OFF150 | +0.023324 | -0.222492 | -0.026103 | 1/2 |
| context=3 | OFF300 | +0.287316 | -0.501541 | +0.166623 | 2/2 |
| context=4 | ENTRY100 | +4.139291 | -5.834146 | +0.274758 | 0/2 |
| context=4 | BASE150 | -1.536026 | +0.774119 | +1.146460 | 0/2 |
| context=4 | BASE300 | -2.839683 | +0.965084 | +1.432104 | 0/2 |
| context=4 | OFF150 | -0.530447 | -0.718123 | +0.097989 | 0/2 |
| context=4 | OFF300 | -0.226731 | -0.835977 | +0.250783 | 0/2 |
| context=5 | ENTRY100 | +11.736809 | -8.705623 | +1.411833 | 2/2 |
| context=5 | BASE150 | -0.712173 | +0.417748 | +0.953986 | 0/2 |
| context=5 | BASE300 | -1.257136 | +0.477794 | +1.512755 | 0/2 |
| context=5 | OFF150 | -0.612262 | -1.199054 | -0.367786 | 0/2 |
| context=5 | OFF300 | -0.850346 | -2.411810 | -0.555981 | 0/2 |
| context=6 | ENTRY100 | +6.398312 | -3.375655 | +4.935355 | 1/2 |
| context=6 | BASE150 | -0.815361 | +0.116795 | +0.933369 | 0/2 |
| context=6 | BASE300 | -1.210073 | +0.169975 | +1.321771 | 0/2 |
| context=6 | OFF150 | +0.177459 | -0.215407 | -0.267167 | 0/2 |
| context=6 | OFF300 | +0.257584 | -0.456083 | -0.326896 | 0/2 |
| context=7 | ENTRY100 | +9.386324 | -12.034078 | +2.473700 | 2/2 |
| context=7 | BASE150 | -0.699328 | -0.004613 | +0.800428 | 0/2 |
| context=7 | BASE300 | -1.431717 | +0.061375 | +0.740371 | 0/2 |
| context=7 | OFF150 | +0.064364 | -0.248575 | -0.699810 | 0/2 |
| context=7 | OFF300 | -0.001220 | -0.368945 | -0.964802 | 0/2 |
| context=8 | ENTRY100 | +7.587112 | -2.361517 | +2.243055 | 2/2 |
| context=8 | BASE150 | -0.266078 | -0.001092 | +1.094634 | 0/2 |
| context=8 | BASE300 | -0.395402 | +0.339059 | +1.457578 | 0/2 |
| context=8 | OFF150 | +0.377242 | -0.681057 | -0.786131 | 2/2 |
| context=8 | OFF300 | +0.737105 | -0.395063 | -1.306963 | 2/2 |
| context=9 | ENTRY100 | +10.584295 | +0.759124 | -0.854338 | 0/2 |
| context=9 | BASE150 | -0.677775 | +0.372211 | +1.730952 | 0/2 |
| context=9 | BASE300 | -0.901454 | +0.661512 | +2.081877 | 0/2 |
| context=9 | OFF150 | +0.306804 | -0.530521 | -1.045061 | 2/2 |
| context=9 | OFF300 | +0.584959 | -0.633743 | -2.673996 | 2/2 |
| context=10 | ENTRY100 | +3.852176 | -4.380674 | +4.891110 | 2/2 |
| context=10 | BASE150 | -0.408106 | -1.350857 | -0.378634 | 1/2 |
| context=10 | BASE300 | -0.245954 | -2.353545 | -0.703106 | 1/2 |
| context=10 | OFF150 | +1.219964 | -0.957441 | -0.632921 | 2/2 |
| context=10 | OFF300 | +2.324039 | -2.827650 | -1.219898 | 2/2 |
| context=11 | ENTRY100 | +4.349901 | -0.940789 | +5.146159 | 1/2 |
| context=11 | BASE150 | -0.384376 | -0.527277 | +0.114544 | 0/2 |
| context=11 | BASE300 | -0.499080 | -0.974849 | -0.105243 | 1/2 |
| context=11 | OFF150 | +1.319126 | -0.196968 | -0.037990 | 2/2 |
| context=11 | OFF300 | +1.997499 | -0.527704 | +0.018553 | 2/2 |
| context=12 | ENTRY100 | +7.242111 | -4.682007 | +4.240326 | 2/2 |
| context=12 | BASE150 | -0.417461 | +0.353143 | +0.858514 | 0/2 |
| context=12 | BASE300 | -0.740812 | +0.371543 | +1.117941 | 0/2 |
| context=12 | OFF150 | +0.175816 | -0.366269 | -0.319884 | 2/2 |
| context=12 | OFF300 | +0.527863 | -0.639753 | -0.859936 | 2/2 |
| context=13 | ENTRY100 | +9.235868 | -0.257445 | +3.290939 | 0/2 |
| context=13 | BASE150 | -0.772423 | -0.173078 | +0.387279 | 0/2 |
| context=13 | BASE300 | -1.046383 | +0.039233 | +0.585984 | 0/2 |
| context=13 | OFF150 | +0.013655 | -0.594612 | -0.648190 | 1/2 |
| context=13 | OFF300 | +0.104246 | -0.985370 | -1.530808 | 2/2 |
| context=14 | ENTRY100 | +8.317133 | -1.123099 | +7.239857 | 0/2 |
| context=14 | BASE150 | -0.531569 | -1.598423 | +0.024434 | 2/2 |
| context=14 | BASE300 | -0.600218 | -2.249532 | -0.098798 | 2/2 |
| context=14 | OFF150 | +1.006534 | -1.029346 | -0.340843 | 2/2 |
| context=14 | OFF300 | +1.330668 | -2.066609 | -0.608848 | 2/2 |
| context=15 | ENTRY100 | +5.383964 | +0.437830 | +7.472432 | 0/2 |
| context=15 | BASE150 | -0.583741 | -0.486014 | +0.404503 | 0/2 |
| context=15 | BASE300 | -0.478457 | -0.889178 | +0.322843 | 1/2 |
| context=15 | OFF150 | +0.934164 | -0.536367 | +0.069609 | 2/2 |
| context=15 | OFF300 | +1.345310 | -1.083364 | +0.022421 | 2/2 |

## Execution and accounting

2112 clean student image forwards /2112 image reads;576 role-level rows;0training/actor/solve/new annotation/Q_dev. All32 child jobs EXIT0, snapshot immutability and matching previously measured V124 macros passed. Root initially EXIT1 due solely to writing create-only COSTS.json over the scheduler accounting file after all rows were produced. Original EXIT/FAILURE retained; recovery wrote costs and completion metadata without model/image operations. The repair script also needed a tuple-typo correction in its final audit write, likewise zero model/image operations. Source now uses the existing overwrite-capable metadata writer for COSTS. COMPLETION_AUDIT.json and RECOVERY_COMPLETION.json document completion; do not claim the original root exited0.

No model rerun, extra seed, selected checkpoint, test access or causal proof. All anonymous rows, paired deltas and subgroup summaries are published; private states/images/IDs remain on NAS.
