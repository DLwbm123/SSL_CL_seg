# V20 / B1 / B2 — same seed and order

Seed163, REFUGE→RIM→Drishti. Both new experiments completed, with all selected targets sealed before their scheduled unified readouts. No new training or evaluation was performed for this comparison. B2 is the frozen historical matching cell, not its two-seed/two-order mean.

| Method | Final % | Old % | Incoming % | Forget pp | Final vs B2 pp |
|---|---:|---:|---:|---:|---:|
|B2|66.8373|64.0572|72.3975|12.3198|+0.0000|
|B1|68.1853|65.6290|73.2979|10.6550|+1.3480|
|V20|70.3506|69.9515|71.1487|6.1051|+3.5133|

| Domain, final stage | B2 macro % | B1 macro % | V20 macro % |
|---|---:|---:|---:|
|REFUGE|71.7731|75.5193|79.1607|
|RIM_ONE_r3|56.3413|55.7388|60.7423|
|Drishti_GS|72.3975|73.2979|71.1487|

V20 improves Final by3.5133pp and Old by5.8943pp versus this B2 cell, while Incoming falls1.2488pp. Its Drishti cup is68.9539%,4.3759pp below B2; old-domain retention does not imply all classes improve.

B1 improves Final by1.3480pp, Old by1.5718pp and Incoming by0.9004pp versus this B2 cell. Simpler confidence-only eligibility is not necessarily a weaker baseline. Its RIM macro falls0.6025pp despite the higher overall score. V20 exceeds B1 Final by2.1653pp but Incoming is2.1492pp lower.

The full V20 matrix has Final68.5165% versus B2+0.6155pp, but fails the original B2 gate: seed164 mean Final−1.5031pp, order2 Old−1.1616pp, and164/O2 Drishti rim/cup−5.4213/−15.8466pp. The selected163/O1 win is not an overall success claim. B1 has one seed/order by explicit user scope reduction and cannot establish multi-seed robustness.

Complete normalized scores and paired deltas are in COMPARISON.json; class cells are in CLASS_METRICS.csv. No independent-patient, SOTA or clinical claim is made.
