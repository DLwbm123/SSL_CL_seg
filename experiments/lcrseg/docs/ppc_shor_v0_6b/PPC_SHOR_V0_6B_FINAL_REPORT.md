# PPC-SHOR V0.6B final report

## Outcome

The single registered development outer-OOF adjudication completed with **FAIL_PPC_SHOR_CURRENT_SAFETY**. V0.6A remains
`BLOCKED_PROTOCOL_OR_LEAKAGE`; no prior status or artifact was changed. This is development
evidence, not external confirmation.

Recovery equivalence passed before the qualified freeze: V0.6A Bayesian weights, PAV fitted
probabilities, all 60 candidate-fold routes and hashes were reproduced exactly. The two V0.6A
false-negative gates were repaired by separating prediction validity from route eligibility and
by counting contiguous bitwise-identical PAV probability levels. Constituent Bayesian routes are
used only for gain/drop quantiles; ensemble modal disagreement is the hard stability statistic,
while any-flip fraction is diagnostic.

The formal GT reservation was created only after five full candidate seals, expert probability
caches, predictions, routes, calibration models, bootstrap weights, sensitivity artifacts and
case order were durably written and reverified. C6 routed 155/198 cases (0.782828). Three-domain gain
was 0.200514, historical gain 0.308428, current-domain drop 0.015315 and domain-oracle gap 0.032462.

Stability: historical-gain p10 0.308824, shared-gain p10 0.197200, current-drop p90 0.032798,
maximum seed-domain-drop p90 0.052448, modal disagreement 0.017449, any-flip fraction 0.090909 and median
case consensus 1.000000. Prediction validity was 200/200 for every case/expert.

Current safety failed all five registered limits: current-domain drop 0.015315 > 0.010,
maximum current-class drop 0.019822 > 0.015, maximum seed-domain drop 0.045946 > 0.020,
current-domain-drop p90 0.032798 > 0.015, and maximum seed-domain-drop p90 0.052448 >
0.025. Value, C3 noninferiority, nondegeneracy, stability, and isolation passed.

Outer evaluator domain/GT reads were 198/198 and occurred after verified seals. Segmentation and
router optimizer/update counts were zero. V0.4 `formal_03` reads, main merges and external tests
were zero.

One pre-reservation engineering run stopped before model forward because its isolated source
worktree lacked the separately maintained official JASCL reference checkout. No GT reservation,
expert forward, domain read, or GT read occurred in that run. The pinned clean reference was
linked through the repository's ignored `third_party` path; the frozen source was unchanged.
