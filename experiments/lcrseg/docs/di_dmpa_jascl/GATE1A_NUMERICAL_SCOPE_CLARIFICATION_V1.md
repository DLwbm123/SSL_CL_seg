# Gate 1A numerical scope clarification V1

Date: 2026-08-30, Asia/Shanghai. This is a user-authorized **post-failure technical scope clarification**, not a mechanism result or a change to admission thresholds.

## Frozen provenance and publication boundary

- Branch: `codex/gate1a-sampled-norm-recovery`, created from report `945b484072cb9f2757be98df34e5d72844596e84`, not main.
- Preregistration: `cfb62554f1e6a2a36850547485b1857dc9a28a20`; original authorization: `25ec97c988af290a4fb7a637c4b7cdfe462deb87`.
- Attempt1 source: `8f4a71a5ea8d145183a3007ccd398ab79387478e`; report: `945b484072cb9f2757be98df34e5d72844596e84`.
- MD SHA256: `32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a`.
- JSON SHA256: `6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf`.

Both original preregistration files remain byte-identical. Attempt1 remains permanently **BLOCKED_NUMERICAL_FAILURE**, with clustering_jobs=0, A1–A6 uncomputed and selected_K=null. `post_failure_scope_clarification=true`; `mechanism_outcomes_observed_before_clarification=false`.

Publish these two clarification files as an independent commit and verify the remote SHA **before any new model forward**. Recovery code and report commits must remain distinct from this clarification commit.

## Clarified numerical guard scope

1. Any NaN/Inf anywhere in the complete feature map blocks.
2. Finite full-map vectors with norm <=1e-12 are diagnostic-only unless registered. Post-ReLU full-map zeros alone do not establish numerical corruption.
3. Gather the exact registered coordinates without deletion, substitution or resampling. Any registered NaN/Inf or norm <=1e-12 blocks.
4. Compute registered norms in float64. Only after validation, normalize by explicit `selected / selected_norm[:,None]`, without eps.
5. Every prototype/cluster center still requires norm >1e-12. Do not modify `geometry_metrics.normalize`.
6. Never drop panel-specific coordinates. No random, neighbor or class-mean replacement; no feature tap, pre-ReLU, source, panel, K-grid, A1–A6 or threshold change; no control rescue.

Keep decoder.dec1 post-ReLU, 16-D, stored384x384, float32 forward, AMP off, eval/no_grad, stochastic_classifier=false and registered post-load forward seeds unchanged.

## Diagnostic and error provenance

For every case/source/role record full-map vector/nonfinite/exact-zero/near-zero counts, minimum positive norm, exact-zero coordinate hash and at most32 row-major coordinate examples. Hash all exact-zero coordinates as H([[y,x],...]); never serialize the full zero list.

Per class record registered count, nonfinite count, norm<=1e-12 count, exact-zero count, min/p01/median/max norm, normalized norm maximum absolute error and full-exact-zero/registered intersection. Norm-summary quantiles use float64 linear quantiles for diagnostics only; registered geometry weighted-ECDF definitions are unchanged.

Registered numerical errors carry panel, baseline, feature source, seed, stage/domain/role, checkpoint ID/SHA, case/class/y/x, invalid count, minimum selected norm, sampling unit SHA and sampling plan SHA. Preserve before/after model-state audits on success and failure.

## Exact plan reuse and known-failure audit

Read-only copy the original attempt1 SHARED_GEOMETRY_SAMPLING_PLAN.json. Verify raw SHA `96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24`, chmod read-only and record reuse. Do not read labels to rematerialize coordinates. Four panels retain identical coordinates, multiplicities, fixed bootstrap draws and seeds. Do not reuse the48 partial feature caches.

First audit **all** registered cases/coordinates of B0/seed1/stage0 REFUGE EMA val, batch_size=8, checkpoint SHA `1e3c99ab3fe39de9755401a31779b5670c624064a73e772938ae57cbb2c3a1b8`. The exact checkpoint path and original sampling-plan path are frozen in the accompanying JSON.

Emit GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json with full/registered zero and nonfinite counts, coordinate hashes/intersections, classwise minimum norms, before/after state and checkpoint hashes. No clustering or optimizer is allowed. Invalid registered vectors are never normalized or passed to geometry; localization may collect diagnostics across all registered cases before reporting the block.

- All registered zero/nonfinite counts zero and no full-map nonfinite: PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED; attempt2 may proceed.
- Any registered norm<=1e-12: BLOCKED_REGISTERED_ZERO_FEATURE; stop without attempt2 and await a new Gate1 v2 preregistration.
- Any NaN/Inf: BLOCKED_NUMERICAL_FAILURE; no attempt2.

## Cooperative cancellation

A failed shard writes STOP_REQUESTED.json. Peers finish the current checkpoint's ImmutabilityGuard and check the flag between checkpoints. Fixed shutdown timeout: **600 seconds**; only after this deadline may SIGTERM be used, with any missing after-state audit disclosed. Scheduling must not change unit seeds or results.

## Conditional attempt2 and hard stop

Freeze and publish the exact recovery code before real localization/formal execution. Only a localization PASS authorizes `gate1a_formal_<RECOVERY_CODE_SHA>_attempt2`. Verify all18 checkpoints; produce all72 feature units with strict registered validation, complete state/cache audits before any of432 geometry jobs. All panels, K=1/2/3/5, five fixed bootstraps and boundary/interior must complete before B0-EMA A1–A6 adjudication. Controls cannot rescue admission. No outcome reinterpretation or threshold change.

All method flags remain false, method_registered=false, di_dmpa_training_launched=false, model_optimizer_steps=transport_optimizer_steps=0, hidden_gt_training_usage=test_gt_usage=none. Gate1B/C, transport, reliability, formal gradient-conflict/teacher-noise, theory final, training, Prostate/MnMS, Gate2 and main merge remain unauthorized. Publish reports, verify remote SHA, then STOP_FOR_INDEPENDENT_REVIEW.
