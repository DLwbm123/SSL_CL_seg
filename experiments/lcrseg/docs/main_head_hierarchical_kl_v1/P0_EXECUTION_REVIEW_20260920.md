# HKL P0-B execution review

Status: **P0-B executed and stopped at the frozen scientific gate**.

This note is a public, source-only summary for external review. It contains no patient data, model weights, private logs, credentials, or operational storage paths.

## What was run

- Candidate base: `e2023ec59ca88860ddb3edde38b47e1ac5924db7`.
- Corrected execution implementation: commit `35c7fd9249a9ff9cc3602ba6cc142615d81dbaa1`.
- Environment: the project server's existing `py38` environment, CUDA device 5 only for this bounded P0.
- Frozen scope: 4 bound states, 42 student forwards and 42 dense-EMA teacher forwards, 84 total; no optimizer, autograd, VJP, checkpoint write, or formal training.
- Result: all 84 forwards completed and the process exited normally.

## Engineering issue found and fixed

The first execution attempt exposed adapter defects before a valid scientific conclusion could be made:

1. stage-2 start payloads were assigned the wrong current-domain stage;
2. the bound batch was indexed by role even though the dispatcher already passed one batch;
3. frozen student/teacher calls omitted `mode="eval"`;
4. the diagnostic used a global ranking instead of the frozen image/GT-class/boundary-stratum ranking and gate.

The first attempt is retained as an invalid engineering attempt. It is not used as scientific evidence.

## Corrected P0 result

The corrected implementation completed the full 84-forward budget. The endpoint gates were:

| endpoint | supported images | conditional errors | captured errors | matched-random expectation | three-class capture | gate |
|---|---:|---:|---:|---:|---:|---|
| O1 | 10 | 123,218 | 30,876 | 30,813.7108 | 35,856 | `INSUFFICIENT_EVIDENCE` |
| O2 | 16 | 75,025 | 31,640 | 18,777.5057 | 26,053 | `PASS` |

For O1, the frozen random-enrichment threshold is `1.25 × 30,813.7108 = 38,517.1385`; the observed 30,876 captured errors do not meet it. O1 therefore fails the preregistered gate even though its support and error-count sufficiency conventions are met. O2 passes both enrichment checks.

## Decision

The failure is not a CUDA, OOM, environment, or process error. It is a reproducible P0 scientific-gate failure for O1 after the adapter was corrected. The controller correctly did not start CUDA qualification, smoke, or the 10,600-update formal matrix. No formal optimizer budget was consumed.

The result must not be rescued by lowering thresholds, changing the endpoint, swapping checkpoints, or retrying the same fixed P0. The next review question is whether O1's weak enrichment is caused by a remaining state/data binding issue or is a genuine limitation of the HKL diagnostic signal. Any code or protocol change requires a separate bound review commit and a new finite authorization.

