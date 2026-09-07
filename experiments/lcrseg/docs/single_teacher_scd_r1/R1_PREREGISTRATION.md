# Single-teacher R1 fixed preregistration

Authorization: the attached R1 protocol, starting bd29494153aa0175b5b52d360f87a44255293d9f. This new branch preserves every prior source, lock, report and attempt.

Primary fixed matrix: U0/L05/L10, each stage1 3200 and stage2 2100 updates, 15,900 total. Conditional E_R1: one fresh 5,300-update trajectory only after complete numeric qualification. No common/old-arm retraining. Maximum new formal updates 21,200; historical 36,008 remain separately accounted.

All V0.1 data, sampling streams, architecture, learning rates, losses and schedules are inherited except the explicitly listed ablation coefficients and E projector implementation. Inputs are current labeled/U by recipe; val stays in a separate evaluator; test and hidden U GT are forbidden. Public stage0 is the only stage1 initialization; stage2 requires this R1 arm's own new stage1.

The complete frozen R1 gates are in R1_AUTHORIZED_PROTOCOL.md sections 5.2–5.5. Pseudo-CE, U-KD, E increment and current capability get separate outcomes. There is no combined scientific PASS. No validation feedback changes a recipe. Final students only; no early stopping or best selection.

GPU mapping: U0=4, L05=5, L10=6; conditional E_R1=7. One of our trainers per card, live memory admission, no termination of other users' work. All private outputs use a create-only NAS root and the storage wrapper.

Qualification runs the inherited regression suite plus explicit R1 identities, label-only access sentinel, lineage/resume, analytic certificates, CPU 100-digit reference and GPU parity. A single three-update replay of the preserved epoch47 checkpoint captures all seven projector calls (successful and failing); it is isolated and never initializes a formal run. E's real diagnostic cap is eight successful updates, with three planned.

Production projector: analytic upper bound, 64 bisections in log1p(scaled eta), 65536 maximum block, original 1e-9 normalized residual. A finite-domain or certificate failure closes E. No second post-freeze numeric repair.
