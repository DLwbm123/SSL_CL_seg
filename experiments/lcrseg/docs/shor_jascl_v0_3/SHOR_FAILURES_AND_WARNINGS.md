# SHOR-JASCL V0.3 failures and warnings

- Formal status: `BLOCKED_NUMERICAL_FAILURE`; exact error: `nonfinite ridge alpha`.
- Durable completion: `COMMAND_FAILED`, child exit code 1, recorded by the server-local parent.
- Scientific status: none. H1-H5 were not evaluated; H6 has only partial isolation evidence.
- Completed stage barrier: `input_audit`. The threshold seal, candidate seal, validation evaluator, gate compiler, and success-report phases were not reached.
- No validation GT or test GT was read. Model construction, model forward, autograd, optimizers, and training remained zero.
- The failed archive and the frozen private input both passed exact manifest verification. Success-only registered artifacts are absent by design after the fail-closed stop.
- A pre-admission test run at the earlier implementation commit had 184 passes and three environment failures because the frozen official JASCL reference checkout was not linked into the clean clone. That evidence was preserved. The dependency was linked read-only at its pinned commit, and the final execution commit passed 188/188 tests.
- A live preflight corrected the implementation's stage-1 cardinality from 110 to the frozen value 140 before the formal directory existed. The corrected execution commit was republished and retested before launch.
- No retry, threshold modification, validation refit, or scientific interpretation of the partial OOF artifact is authorized.
