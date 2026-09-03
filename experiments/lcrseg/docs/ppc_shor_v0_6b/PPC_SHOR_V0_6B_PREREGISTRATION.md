# PPC-SHOR V0.6B preregistration

PPC-SHOR V0.6B is an executor-accounting recovery of V0.6A followed, if qualification
passes, by one development outer-OOF adjudication. It is not a retry of V0.6A and cannot
change V0.6A or any earlier status, artifact, method, candidate, route rule, or gate.

The scientific design is identical to V0.6A: 990 calibration seed-rows from 575 patients,
198 segmentation-value seed-rows from 177 patients, five patient-group outer folds, weighted
PAV, 200 Bayesian bootstrap fits, kappa `[10,30,100,inf]`, tau `[0.90,0.95,0.98]`, rho
`0.80`, and controls C0-C8. The per-fold candidates are fixed before qualification to
`[k010_tau095,k010_tau095,k010_tau095,k100_tau090,k100_tau090]`. Their stitched C6 route
has 155/198 historical assignments and SHA-256
`c03c457d0b5ab7a529dd8ec07076b06c2fe41a22e7439265f24ab046687a41e7`.

Only executor semantics change. Prediction validity is counted independently for each
historical expert across all 200 finite calibrator predictions; route eligibility remains a
separate top-1-dependent count. Effective PAV parameters are contiguous runs of bitwise-equal
float64 fitted probabilities, without tolerance and without changing fitted predictions.
Constituent Bayesian routes provide gain/drop distributions; ensemble modal disagreement is
the hard stability statistic, while any-flip fraction is diagnostic only.

Qualification must reproduce V0.6A weights, fitted probabilities, all 60 candidate-fold route
arrays and route hashes, pass counts `[8,8,8,11,11]`, feasibility `200/200`, the five registered
corrected parameter ratios, ten unique global candidate routes, the fixed selection, and the
stitched route exactly. Qualification performs no expert forward and reads no outer domain or
segmentation GT.

The formal scientific attempt begins only when `FORMAL_GT_ACCESS_RESERVATION.json` is created
atomically with `O_EXCL`, after a qualified freeze commit has been pushed and its remote SHA
verified. Before the verified candidate seal, the controller holds no outer domain or label
field. The evaluator alone may reveal domain and GT after seal verification. A reservation
permits exactly one complete evaluator run and forbids code changes or partial reruns.

Status precedence, all numeric gates, recovery expectations, immutable input identities, and
the exact three-commit sequence are frozen in the JSON registration. V0.4 `formal_03` reads,
training/optimizer updates, main merge, and external testing are all forbidden.
