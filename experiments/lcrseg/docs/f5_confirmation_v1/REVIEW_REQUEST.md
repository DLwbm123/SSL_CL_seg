# External review request: F5_CONFIRMATION_V1

**STOP_AWAITING_EXTERNAL_CODE_REVIEW. Real-data optimizer updates = 0.**

Review branch: `codex/f5-confirmation-v1`, starting from results anchor `96ba0e29f601589a069bcbfdf50b2f4aafc1d883`. Actual reused training implementation: `cc21871a43e3cd105cddaf831f5c3ea2fef59b9e`. Parent: **NATIVE_LR_SRC_A_3DOMAIN_V1**, explicitly not original KI. No shared training code or historical sealed artifact is modified.

The implementation adds one metadata module, one finite execution controller, one CLI and one finite synthetic test suite. It directly reuses native_runner.target_task, StageTrainer, native parent/data/model/checkpoint and Counter. No replacement loss/network, new permission monkeypatch or original B/C/D scheduler is introduced. The serial controller intentionally avoids a second parallel scheduling framework.

## Review evidence

- [PLAN.json](PLAN.json): exact 14 trajectories / 28 new target stages / 74,200 formal scientific and physical calls; three reused sources and eight imported historical target stages counted separately; frozen options, hashes, analysis and gates.
- [P0_REPORT.md](P0_REPORT.md), [P0_STAGE_METRICS.csv](P0_STAGE_METRICS.csv), [P0_EVIDENCE.json](P0_EVIDENCE.json): development 161 and already-observed 162, complete selected timelines/options and saved spectral diagnostics.
- [SOURCE_REUSE.json](SOURCE_REUSE.json), [REUSE_METADATA.json](REUSE_METADATA.json): source file metadata and declared hashes; tensor/schema/forward PENDING.
- [TEST_REPORT.json](TEST_REPORT.json), [CPU_ATTEMPTS.json](CPU_ATTEMPTS.json), [CPU_PHYSICAL.jsonl](CPU_PHYSICAL.jsonl): bounded CPU synthetic checks with cumulative cost and failed attempts retained. A test initially asserted identity R instead of zero residual R; only that test expectation was corrected.
- [CODE_MANIFEST.json](CODE_MANIFEST.json): actual Python source tree including transitively reused implementation. It excludes mutable reports/private payloads and does not self-reference a commit hash.
- [RUNBOOK.md](RUNBOOK.md): implemented prepare/plan/qualify/run/report commands, new authority binding, private configuration and pending production checks.

## Requested review focus

1. Full resolved_options agree across sealed C/D receipts, selected P1/F5_C02/B2_C06 manifests and actual original resolver; no candidate reselection. R lr and F5 nested coefficients remain correct.
2. P1 is the only executable matrix. B0/F5 seed162 remain imports, B2 seed162 is new, seed161 is development only, no source/P2/other arms run.
3. New genuine external approval and user launch are required before a production capability is issued; old authority is rejected. CPU tests use no fake production permit.
4. Current-domain permissions and own-trajectory inheritance remain delegated to the original implementation. B0 never constructs U data; U labels are never read. Current EMA/reset/merge/resume semantics are preserved.
5. Strict durable per-stage and total physical accounting, failure retention, lost-tail refusal and sealed skipping are correct. No cap extension or budget borrowing.
6. Paired seed/order metrics and G1–G3 are budget decisions only after complete P1; negative domains/seeds remain visible. Missing independent patients/external domain/formal method comparisons are explicit.

P0 does not imply across-seed superiority: F5's development seed161 mean Final differs from B0 by -0.000113407, while observed seed162 differs by +0.013952836. In O2 seed162, REFUGE changes +0.001352 and Drishti_GS -0.004133 versus B0; improved total Final does not establish uniformly better retention. B2 seed162 is missing historically and is deliberately included in the new plan.

No training, CUDA qualification, real smoke, automatic monitoring or P2 has started. Review approval must be supplied independently and followed by explicit user launch confirmation. Native CUDA/source tensor/current-L qualification remains future work; CPU synthetic success is not represented as production validation.
