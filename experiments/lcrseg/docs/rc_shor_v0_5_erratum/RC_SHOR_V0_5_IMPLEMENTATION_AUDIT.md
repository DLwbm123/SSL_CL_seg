# RC-SHOR V0.5 implementation audit

This is a derived-artifact-only, non-adjudicative audit. It performed no model forward,
checkpoint load, image read, label-H5 read, domain-manifest read, router refit, candidate
selection, or outer evaluation.

## Findings

1. `routing_rows()` filtered routes in outer-fold append order and paired them with utility
   arrays in global `row_index` order. Sorting every policy by `row_index` repairs the summary.
2. `routes_for_bootstrap_draws()` did not use `rho`, did not apply the feasible >= 90 ensemble
   gate, and did not execute the full consensus procedure. Its p10/p90 values therefore do not
   characterize bootstrap realizations of final C6.
3. All five folds had fewer than 90 feasible replicates ([59, 38, 74, 83, 44]); final C6 consequently remained C0.
   The five common replicate indices are an outer-fold intersection, not per-unit feasibility.

Corrected gains are C3 0.21708218085929243, C4 0.11330836336206514, C5 0.020747627739757475, and C6 0. C3 historical gain is
0.3256232712889387 and its current-domain drop is 0. The machine-readable audit contains per-fold,
per-expert, per-seed/domain feasibility, inactive-row finite-prediction counts, attainable
consensus, conformal degeneracy, lambdas, C5 coverage, and candidate-route identities.

RC-SHOR V0.5 remains `FAIL_RC_SHOR_VALUE`; no original V0.5 artifact or conclusion changed.
