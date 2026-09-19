# Bounded invocation and authority contract

Current state: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. This document is an execution contract, not permission to run it. Both CPU attempts have been used; do not invoke `test` again.

## Public interfaces

`python -m experiments.lcrseg.agms_observer_v1 plan` emits the canonical proposal. `prepare` writes only this study's frozen documents. `test --reference REF --evidence EVIDENCE` is CPU-generated only, bounded by immutable attempt records and NAS scope; exhausted now. `p0` always refuses. `qualify --mode cuda --config CONFIG`, `qualify --mode smoke --config CONFIG`, `run --config CONFIG`, and `report --config CONFIG` require the new admission gate. These module-form examples describe the API; use the neutral process form below on the server.

Config fields: `code` (clean reviewed checkout), `execution_commit`, `review`, `launch_confirmation` (external files, outside the checkout), `run_root` (new create-only NAS namespace), `prefix_root` (original B2 immutable results), `reference` (native reference code), and `data` (frozen dataset binding). Recover the concrete historical paths from private authority records; do not place paths, credentials or patient material in public Git.

`authority.preflight` binds reviewed_code_commit, code_tree_sha256, plan_sha256, prefix_sha256, import_sha256 (A0 plus controls), environment_sha256 and execution_sha256. The genuine external review must name this study, the actual reviewed commit and bindings, external reviewer/evidence, exact CAPS and phases CUDA/SMOKE/P1, and must not be a template. A separate user launch receipt must bind those same fields and the real review digest, with `user_confirmed=true` and prompt identifier `AGMS_OBSERVER_V1_USER_LAUNCH`. This identifier is not confirmation by itself. Preserve the user's actual new instruction as evidence; never infer it from the preparation prompt or old delegation. No approval example with a passing decision is shipped.

Before future authorized execution: verify clean HEAD/code manifest/CPU evidence and exact canonical plans, then NAS mount/free space/real write-read probe. Set a currently suitable GPU from5/6/7 using actual VRAM sharing. Do not change environment fingerprint, scientific options or historical code to pass admission.

## Neutral argv and storage

Store private invocation JSON and logs on NAS. Use environment variables `TASK_MODULE`, `TASK_ARGS` (JSON list), `TASK_PYTHON`, `TASK_WRAPPER`; the launcher resolves them without putting method/project paths on process argv. A reviewed operator may use:

```sh
# TASK_WRAPPER is the existing with_nas_storage.sh in the reviewed checkout.
# TASK_MODULE and TASK_ARGS are set privately to the gated interface above.
set -- "$TASK_PYTHON" -c 'import os,json,importlib; importlib.import_module(os.environ["TASK_MODULE"]).main(json.loads(os.environ["TASK_ARGS"]))'
source "$TASK_WRAPPER"
```

The visible Python argv contains only the neutral loader. For long work use the existing reliable detached launcher and NAS stdout/stderr; perform one initial `ps -ww`/GPU mapping check. No new monitoring is authorized. Auxiliary imports inherit the same neutral argv. Do not run the public module-form examples verbatim as long-lived processes.

## Future sequence and acceptance

1. New review AND separate user launch, zero-update preflight. No file may silently substitute an old review or launch.
2. CUDA qualification10, once: continuation5, B2/shadow4, expected injected failure1. Non-predefined error stops; no automatic retry.
3. Both original B2 prefix payloads must pass exact receipt, tensor schema/hash, file integrity and environment binding. Then smoke4 per order, L-only, no U and no reused smoke state.
4. Formal controller processes exactly the two canonical OBS0 nodes. Counter records every optimizer invocation durably. A missing checkpoint tail, wrong identity or failure record stops. Same-stage restore reconstructs the new ObserverTrainer with full heads/EMA/risk/optimizer/diagnostics/RNG. It cannot replay a physical update.
5. Save deployment as merged main network only; preserve training checkpoint on NAS. Verify both deployed models and full prefix/target integrity, physical5300/smoke8/CUDA10, all cost sessions and diagnostic32VJP, then export complete10/30/12/8 report. Historical inputs remain imports, new diagnostics remain NA on old endpoints.
6. Independently publish only allowed aggregate report/metadata in a separate results checkout if the new user launch grants publication. Record public delivery separately from original execution receipts. End `COMPLETE_OBSERVER_AWAITING_SCIENTIFIC_REVIEW`; no new search, conditional weighting, seeds, ROUTE_HALF or monitoring follows automatically.

Only after new real approval and confirmation may concrete production paths/receipts be registered. This preparation performed none of steps2–6.
