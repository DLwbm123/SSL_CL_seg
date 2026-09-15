# User-authorized reduction of unstarted experiments

On 2026-09-15 the user explicitly requested one replication seed instead of three,
plus further reductions of experiments that had not started. This amendment
supersedes the remaining dispatch schedule, not the historical run records or
the qualified training implementation. Full node membership is recorded in
[SCHEDULE_REDUCTION_20260915.json](SCHEDULE_REDUCTION_20260915.json).

| Scope | Original | Amended |
|---|---:|---:|
| Development target stages | 304 | 280 |
| Replication target stages | 120 | 32 |
| Total target stages | 424 | 312 |
| Source models already trained | 4 | 4 preserved |
| Total DAG nodes, including selection | 440 | 328 |
| Target optimizer update cap | 1,123,600 | 826,800 |
| Source plus target update cap | 1,155,600 | 858,800 |

The reduction cancels **112 unstarted target stages / 296,800 updates**. It does
not mark cancelled stages as completed and does not reset any physical ledger.

## Retained work

- Replication uses only **seed 162**, the first previously specified replication
  seed. Development seed 161 and all completed source artifacts remain intact.
- Both domain orders and full 3,200/2,100 target-stage budgets remain.
- All F1–F5 frameworks remain. Replication retains B0 parent LCTX, B3 feature-only
  LCTX and B4 joint PAS/KL feature baseline. B3 and B4 were already frozen as the
  strong labeled-only and strong SSL references before this amendment. B1/B2
  development results remain reportable; only their unstarted replication is removed.
- F5 retains **C01 and C02**, the candidates with a started trajectory at the
  dispatcher pause. Their remaining stages and both orders finish. C03–C08
  are cancelled by execution state, without using their performance to prune.
- No new method, candidate, data access, seed or Phase E experiment is introduced.

## Engineering and provenance

The original dispatcher is paused and replaced; its workers are **not killed,
restarted or resumed**. The replacement adopts their PID/start-time identities,
then dispatches only the validated subset through the original qualified worker
entry. It refuses cancellation of any launched node, missing prerequisites,
failed nodes or automatic retries. GPU allocation remains 5/6/7 with the existing
12,000 MiB free-memory admission rule. An exclusive controller lock prevents
duplicate replacement dispatchers.

Worker commit remains `cc21871a43e3cd105cddaf831f5c3ea2fef59b9e`; the new controller
is separately versioned. Original CPU/CUDA/smoke receipts continue to qualify
the unchanged worker. CPU checks cover subset arithmetic, prerequisite retention,
forbidden source cancellation, single-seed aggregation and missing paired orders.
No additional real smoke or optimizer calls are needed for this scheduling change.

The amended completion artifacts are `aggregate_results_reduced.json` and
`replication_summary_reduced.json`. They include the amendment and retained
receipts, all three baseline comparisons, and per-order paired metric differences.
Historical receipts and the original protocol/authority files are not rewritten.

## Limits of the resulting evidence

There is **one replication seed**, so no multi-seed stability claim, across-seed
standard deviation or seed confidence interval is reported. These remain
development patients, not independent patient validation. F5 has two development
candidates while F1–F4 had eight; equal tuning budget across frameworks must not
be claimed. The reduction occurred after partial results were available and was
requested to reduce compute, not pre-registered before the study began.
