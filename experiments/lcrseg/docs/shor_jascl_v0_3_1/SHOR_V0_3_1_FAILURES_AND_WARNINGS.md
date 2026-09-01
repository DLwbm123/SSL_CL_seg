# SHOR-JASCL V0.3.1 failures and warnings

- Final scientific status: `FAIL_SELECTIVE_OVERRIDE_STABILITY`.
- H5 failed because current-domain-drop p90 was `0.029784 > 0.015`, maximum seed-domain-drop p90 was `0.029784 > 0.025`, and stage-2/domain-1 threshold feasibility was only 3/5 for seed 1 and 2/5 for seed 2.
- H5 shared-gain p10 (`0.098874`), historical-gain p10 (`0.151029`), and finite-value requirements passed, but partial H5 success cannot rescue the gate.
- H1, H2, H3, H4, and H6 passed. S0, S1, S2, and S4 are controls and cannot rescue S3.
- The V0.3 inactive-NaN engineering blocker was repaired and did not recur; the old V0.3 record remains unchanged with null scientific status.
- Private paths, case IDs, alphas, predictions, labels, and raw CSV rows are intentionally excluded from public reporting.
- The single V0.3.1 formal attempt is consumed. No retry or downstream evaluation is authorized.
