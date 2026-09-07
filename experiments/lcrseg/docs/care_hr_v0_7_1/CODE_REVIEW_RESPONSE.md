# CARe-HR V0.7.1 review response

Status: `BLOCKED_EVALUATOR_SEMANTICS_MISMATCH`. This is a partial A0/A1 delivery, not a capacity experiment completion or method admission.

| Issue | New location / action | Evidence / remaining work |
|---|---|---|
| union versus macro foreground target | `semantic_audit.py`, registration | Exact required synthetic witness: union gain 0, macro gain 4/15; new final scoring API withheld pending rule resolution |
| Ignore-label and all-invalid images | `semantic_audit.py` | GT=255 counterexamples reproduced from exact historical function ASTs; no convenient scoring rule selected |
| Patient conformal finite-sample rank | `conformal.py` | n counts unique patient maxima, exact decimal rank, infinity for k>n, no risk guarantee; synthetic tests pass |
| Length/shape/nonfinite errors | `conformal.py`, `actions.py` | Broadcast/truncation rejected; probabilities, mask area/centroid/overlap/IDs, route and vote counts validated; full risk-policy interface not implemented |
| Frozen action budgets | `actions.py` | Reuses unchanged proposal builder and blending; no-op, 12/4/3 caps, common lambdas, strict area budgets, zero-foreground and nesting checked |
| Multiple-action scoring / harm | Not implemented | Int64 confusion increments, oracle enumeration scoring and brute-force parity require semantic resolution |
| GT/domain isolation and lineage | Blind action signatures and registration | Synthetic GT-sentinel check only; real population, routes and private inputs not opened |
| Original review lock and history | `HISTORY_PROTECTION.json` | 82 SHA-256-protected files and inherited immutability tests pass; no old source/test/document changed |
| Validation counts | `SYNTHETIC_TEST_REPORT.json` | 129/0/0/0 on local and server exact source, with 34 new and 95 inherited tests; complete A1 explicitly false |
| Refit stability | `DRAFT_GATE_ACCOUNTING.json` | NOT_EVALUATED; no real fitting or substituted bootstrap performed |

The historical scorer is `shor_v0_4_test.case_metrics`, called by V0.6B through V0.6A. Extracting this pure function from public source does not open the forbidden V0.4 private test artifacts. The score AST audit is a source diagnostic, not a substitute for full evaluator integration.
