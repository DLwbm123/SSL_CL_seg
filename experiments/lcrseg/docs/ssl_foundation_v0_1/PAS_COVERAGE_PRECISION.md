# PAS coverage and precision

All table values are equal-patient means at epoch100, teacher argmax, training_like (four draws first averaged within patient). P=accepted-correct/accepted; C=accepted/predicted-class; I=accepted/valid-image-pixels; R=accepted-correct/GT-class. These are val measurements, never train-U hidden-label measurements. Full CSVs include clean mode, all epochs, student labels, pooled counts and actual defined-denominator counts.

## Actual training prototype library

| Domain | Arm | Class | P | C | I | R | Patients with defined P |
|---|---|---|---:|---:|---:|---:|---:|
| Drishti_GS | MT_PAS_G0 | background | 0.989301 | 0.990537 | 0.829327 | 0.975905 | 25 |
| Drishti_GS | MT_PAS_G0 | rim | 0.715555 | 0.886377 | 0.069828 | 0.576170 | 25 |
| Drishti_GS | MT_PAS_G0 | cup | 0.711898 | 0.923055 | 0.079773 | 0.829476 | 25 |
| Drishti_GS | MT_PAS_G1 | background | 0.997016 | 0.968642 | 0.809551 | 0.960206 | 25 |
| Drishti_GS | MT_PAS_G1 | rim | 0.794709 | 0.644331 | 0.050156 | 0.454369 | 25 |
| Drishti_GS | MT_PAS_G1 | cup | 0.724013 | 0.908889 | 0.080368 | 0.867407 | 25 |
| RIM_ONE_r3 | MT_PAS_G0 | background | 0.998188 | 0.836023 | 0.751635 | 0.836470 | 40 |
| RIM_ONE_r3 | MT_PAS_G0 | rim | 0.744108 | 0.918710 | 0.071470 | 0.738527 | 40 |
| RIM_ONE_r3 | MT_PAS_G0 | cup | 0.822357 | 0.861660 | 0.019907 | 0.605132 | 40 |
| RIM_ONE_r3 | MT_PAS_G1 | background | 0.994853 | 0.967769 | 0.880656 | 0.975762 | 40 |
| RIM_ONE_r3 | MT_PAS_G1 | rim | 0.751583 | 0.655663 | 0.045799 | 0.476599 | 40 |
| RIM_ONE_r3 | MT_PAS_G1 | cup | 0.774874 | 0.489483 | 0.013896 | 0.388145 | 40 |

## Fresh diagnostic prototypes: within-network filtering

| Domain | Arm | Class | Filter | P | C | R |
|---|---|---|---|---:|---:|---:|
| Drishti_GS | MT_CONF_G0 | rim | raw | 0.697256 | 1.000000 | 0.631333 |
| Drishti_GS | MT_CONF_G0 | cup | raw | 0.693612 | 1.000000 | 0.839611 |
| Drishti_GS | MT_CONF_G0 | rim | confidence | 0.712818 | 0.885556 | 0.582102 |
| Drishti_GS | MT_CONF_G0 | cup | confidence | 0.711445 | 0.914908 | 0.809042 |
| Drishti_GS | MT_CONF_G0 | rim | PAS | 0.712661 | 0.884256 | 0.581023 |
| Drishti_GS | MT_CONF_G0 | cup | PAS | 0.711445 | 0.914908 | 0.809042 |
| Drishti_GS | MT_CONF_G1 | rim | raw | 0.723979 | 1.000000 | 0.615506 |
| Drishti_GS | MT_CONF_G1 | cup | raw | 0.646276 | 1.000000 | 0.927443 |
| Drishti_GS | MT_CONF_G1 | rim | confidence | 0.750202 | 0.855924 | 0.549793 |
| Drishti_GS | MT_CONF_G1 | cup | confidence | 0.672261 | 0.914779 | 0.893418 |
| Drishti_GS | MT_CONF_G1 | rim | PAS | 0.756973 | 0.842002 | 0.545544 |
| Drishti_GS | MT_CONF_G1 | cup | PAS | 0.686477 | 0.899770 | 0.892978 |
| Drishti_GS | MT_PAS_G0 | rim | raw | 0.698073 | 1.000000 | 0.628008 |
| Drishti_GS | MT_PAS_G0 | cup | raw | 0.696275 | 1.000000 | 0.869718 |
| Drishti_GS | MT_PAS_G0 | rim | confidence | 0.715620 | 0.887690 | 0.577109 |
| Drishti_GS | MT_PAS_G0 | cup | confidence | 0.711898 | 0.923055 | 0.829476 |
| Drishti_GS | MT_PAS_G0 | rim | PAS | 0.715579 | 0.886228 | 0.576095 |
| Drishti_GS | MT_PAS_G0 | cup | PAS | 0.711898 | 0.923055 | 0.829476 |
| Drishti_GS | MT_PAS_G1 | rim | raw | 0.747495 | 1.000000 | 0.664708 |
| Drishti_GS | MT_PAS_G1 | cup | raw | 0.702670 | 1.000000 | 0.909748 |
| Drishti_GS | MT_PAS_G1 | rim | confidence | 0.770312 | 0.882143 | 0.609241 |
| Drishti_GS | MT_PAS_G1 | cup | confidence | 0.723496 | 0.909831 | 0.867603 |
| Drishti_GS | MT_PAS_G1 | rim | PAS | 0.783607 | 0.603207 | 0.414375 |
| Drishti_GS | MT_PAS_G1 | cup | PAS | 0.724087 | 0.908743 | 0.867375 |
| RIM_ONE_r3 | MT_CONF_G0 | rim | raw | 0.736971 | 1.000000 | 0.773119 |
| RIM_ONE_r3 | MT_CONF_G0 | cup | raw | 0.787914 | 1.000000 | 0.657982 |
| RIM_ONE_r3 | MT_CONF_G0 | rim | confidence | 0.751585 | 0.924828 | 0.733038 |
| RIM_ONE_r3 | MT_CONF_G0 | cup | confidence | 0.815705 | 0.859477 | 0.599339 |
| RIM_ONE_r3 | MT_CONF_G0 | rim | PAS | 0.751447 | 0.902882 | 0.721355 |
| RIM_ONE_r3 | MT_CONF_G0 | cup | PAS | 0.814730 | 0.857291 | 0.598915 |
| RIM_ONE_r3 | MT_CONF_G1 | rim | raw | 0.693628 | 1.000000 | 0.790688 |
| RIM_ONE_r3 | MT_CONF_G1 | cup | raw | 0.802738 | 1.000000 | 0.586097 |
| RIM_ONE_r3 | MT_CONF_G1 | rim | confidence | 0.721761 | 0.881727 | 0.727421 |
| RIM_ONE_r3 | MT_CONF_G1 | cup | confidence | 0.859835 | 0.643853 | 0.452835 |
| RIM_ONE_r3 | MT_CONF_G1 | rim | PAS | 0.729475 | 0.853915 | 0.709900 |
| RIM_ONE_r3 | MT_CONF_G1 | cup | PAS | 0.866048 | 0.640165 | 0.452835 |
| RIM_ONE_r3 | MT_PAS_G0 | rim | raw | 0.730245 | 1.000000 | 0.782899 |
| RIM_ONE_r3 | MT_PAS_G0 | cup | raw | 0.796558 | 1.000000 | 0.663943 |
| RIM_ONE_r3 | MT_PAS_G0 | rim | confidence | 0.743803 | 0.935570 | 0.748652 |
| RIM_ONE_r3 | MT_PAS_G0 | cup | confidence | 0.822420 | 0.862390 | 0.605561 |
| RIM_ONE_r3 | MT_PAS_G0 | rim | PAS | 0.744164 | 0.918668 | 0.738530 |
| RIM_ONE_r3 | MT_PAS_G0 | cup | PAS | 0.822362 | 0.861636 | 0.605139 |
| RIM_ONE_r3 | MT_PAS_G1 | rim | raw | 0.733072 | 1.000000 | 0.663768 |
| RIM_ONE_r3 | MT_PAS_G1 | cup | raw | 0.732797 | 1.000000 | 0.605569 |
| RIM_ONE_r3 | MT_PAS_G1 | rim | confidence | 0.757693 | 0.673755 | 0.492068 |
| RIM_ONE_r3 | MT_PAS_G1 | cup | confidence | 0.777051 | 0.492480 | 0.389422 |
| RIM_ONE_r3 | MT_PAS_G1 | rim | PAS | 0.751583 | 0.655663 | 0.476599 |
| RIM_ONE_r3 | MT_PAS_G1 | cup | PAS | 0.774874 | 0.489483 | 0.388145 |

Fresh prototypes are recomputed from current labeled training data at each snapshot and never modify training state. The epoch100 actual library was last refreshed at epoch96. Actual-vs-fresh differences are explicit; epoch20 actual library is NOT_STARTED. SUP/CONF PAS masks are post-hoc diagnostic candidates, not their training masks.

PAS often adds very little filtering beyond confidence under G0, and precision changes are not uniformly positive. MT_PAS_G1 Drishti rim gains some precision while losing substantial recall; RIM foreground precision slightly falls. These observations do not establish pseudo-label errors as the sole cause of the failed screen. Both segmentation and precision were actually evaluated; no coverage-only claim is made.
