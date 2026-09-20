# Fixed review entry: MAIN_HEAD_HIERARCHICAL_KL_V1

This entry is the fixed-code handoff for the same branch:
`codex/main-head-hierarchical-kl-v1`.

## Exact source handoff

- Prior reviewed branch tip: `0bc08ce8db1c59bcd987513a379526d4f3972af4`.
- R1 remediation commit on the pushed branch: `886f218ba933267d11d704b9204659a144c6079f`.
- R2 ledger-binding commit on the pushed branch: `a1dbfb391872e0b7d2653d7af37afb7a6a539d0d`.
- Fixed review tree before this metadata-only entry update: `7253f87be96c67754c7b9b35f3a34439ef09064c`.
- The final full SHA is supplied with this handoff. Review the exact tree at that SHA, not the moving branch name.

## R1/R2 evidence

R1 adds the explicit arm-aware factory `for_resume(model, provider, options=None, *, arm)` and keeps the constructor's native/data/CUDA rejection. The interface regression reads the signature without constructing a trainer and reported PASS.

R2 binds the study-global ledger through registered identity `MAIN_HEAD_HKL_LEDGER_REGISTRY_R1`. The entry point resolves the registry's canonical path and verifies ledger_id, study_id, authorization identity, source commit, immutable historical evidence bindings and the historical 15-call floor before reservation. Missing, corrupt, mismatched, copied or exhausted state is a hard stop; output-directory changes cannot reset quota. `tests.py` charges before optimizer invocation and closes PASS/FAIL records. The pure regression covers replacement-path, wrong-identity and degraded-history rejection with zero optimizer calls.

## Binding and accounting

- Historical CPU evidence remains bound to its original preparation source and records 15 optimizer calls in one PASS attempt.
- New remediation regression evidence is separate: `REPAIR_REGRESSION.json` records zero new optimizer calls, zero native/private reads, and no CUDA, smoke, P0-B, or formal execution.
- `REPAIR_SOURCE_BINDINGS.json` binds the reviewed trainer, budget, regression, and test sources by SHA-256.
- `CPU/BUDGET_LEDGER_PUBLIC_SUMMARY.json` is a redacted registration summary. It contains no private path, patient information, weights, checkpoint, log, or credential.

## Execution boundary

This handoff requests source review only. It does not authorize P0-B, CUDA qualification, smoke, formal training, or monitoring. Those remain blocked until this exact fixed tree receives independent external approval and a separate user launch confirmation. No historical 15-call report has been relabeled as a new full test.

## Fixed review URL

After push, use the full branch-tip SHA supplied with this handoff and open:
`https://github.com/DLwbm123/SSL_CL_seg/tree/<FULL_BRANCH_TIP_SHA>/experiments/lcrseg/docs/main_head_hierarchical_kl_v1`
