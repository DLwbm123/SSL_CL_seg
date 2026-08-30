# Gate 1A v2 failures and warnings

Scope: the unique formal null-aware v2 attempt using exact diagnostic code `8ae5d7532f90aee5d53c0d966706ef64c18a19ac`. Historical v1 numerical failures remain untouched and must not be relabeled by the v2 result.

## Formal execution

- One formal attempt, exit code0, all72 feature units and432 geometry jobs completed.
- No full-map/registered NaN or Inf, invalid center, model mutation, leakage, missing/duplicate panel, checksum mismatch or protocol blocker was reported. The independent postrun checks all passed.
- All22 registered null observations were retained, never normalized, and given worst-case distance2. Their existence is expected under v2, not an engineering failure.
- No inactive prototype slots, all-null original fits or all-null bootstrap fits occurred in these real data. The corresponding paths were covered by synthetic tests.
- No retry, additional formal attempt, feature-source switch, seed/K/coordinate/threshold change, optimizer construction, training or downstream gate was performed.

## Frozen-iteration convergence warnings — not suppressed

There were2,592 fits (432 original +2,160 bootstrap), each with five restarts, for12,960 restart records. **79 restarts reached the frozen100-iteration cap without satisfying the angular-movement stopping criterion. Six selected minimum-objective fits were nonconverged; all six have K=5.** All K=2/3 selected fits converged. The six records are:

| Panel | Seed | Stage/domain | Class | K | Fit | Selected restart | Iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0-EMA | 2 | 0 / REFUGE | 2 | 5 | bootstrap replicate2 | 4 | 100 |
| B0-student | 0 | 2 / Drishti_GS | 0 | 5 | original | 0 | 100 |
| B0-student | 1 | 0 / REFUGE | 2 | 5 | original | 2 | 100 |
| C0-EMA | 0 | 0 / REFUGE | 2 | 5 | bootstrap replicate3 | 4 | 100 |
| C0-student | 0 | 0 / REFUGE | 1 | 5 | original | 3 | 100 |
| C0-student | 0 | 0 / REFUGE | 1 | 5 | bootstrap replicate4 | 0 | 100 |

Replicates and restarts use the preregistered zero-based IDs. No iterations were added, no restart was rerolled, and none of these results was discarded or replaced. The predefined minimum-objective restart selection remains unchanged. Numerical finiteness and all admission conditions hold; convergence itself was not an extra preregistered admission threshold. This remains an optimization-approximation limitation for independent review, not grounds to claim every fit converged.

All79 per-restart warnings and the six selected-fit records are in [GATE1A_V2_FIT_WARNINGS_AUDIT.json](gate1a_v2_results/postrun_8ae5d75_attempt1/GATE1A_V2_FIT_WARNINGS_AUDIT.json). Original records also remain inside all432 geometry-unit JSON files. The short runner-written warning file inside the raw attempt is preserved verbatim; this additive report provides the detailed warning census rather than overwriting it.

## Test and development record

- Synthetic development pass1:32 passed,0 failed,1.88s.
- Synthetic pipeline plus v1 core/recovery regression pass2:97 passed,0 failed,6.03s.
- Published exact code, including real known-zero integration:98 passed,0 failed,0 skipped,11.41s. No pytest warnings were emitted.
- Development outputs are separately archived under `gate1a_v2_results/development/`. They are synthetic verification, not additional formal diagnostic attempts.

## Report transfer events

These events occurred after the formal attempt completed and did not modify its source, data or artifacts:

1. `rsync` transfer failed with remote `rsync: command not found` (exit12). No environment/package installation was performed.
2. An uncompressed read-only tar transfer was intentionally stopped because of low transfer throughput. Its local partial archive copy produced the expected `Truncated tar archive` error. Only this transfer process was terminated; no model/diagnostic process was interrupted.
3. A complete gzip-compressed tar transfer subsequently finished with exit0. Every included formal artifact was checked against its original manifest. The temporary partial local copies were replaced by the matching complete originals. Raw cloud artifacts were never rewritten.

No secret/private key, checkpoint or raw feature tensor is included in the public report. Tensor paths/sizes/checksums are retained in the original manifest. No historical evidence was removed.

## Byte-preserving publication check

The default Git whitespace checker flagged the original CSV files' standard CRLF line endings. These are unchanged raw diagnostic bytes, not data corruption. `git -c core.whitespace=cr-at-eol diff --cached --check` passed with no diagnostics. No CSV line-ending conversion or other formatting rewrite was applied; index bytes were independently checked against all588 delivery-manifest entries.

## Stop

`model_optimizer_steps=0`; `transport_optimizer_steps=0`; `hidden_gt_training_usage=none`; `test_gt_usage=none`; `method_registered=false`; `di_dmpa_training_launched=false`; Gate1B=false; Gate1C=false. No Prostate, MnMS, Gate2, transport/reliability/gradient-conflict/noise/theory diagnostic, training or main merge. `STOP_FOR_INDEPENDENT_REVIEW`.
