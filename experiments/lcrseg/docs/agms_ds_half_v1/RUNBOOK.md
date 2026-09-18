# Runbook: separate DS-half study

State: STOP_AWAITING_EXTERNAL_CODE_REVIEW. No real prefix/P0/CUDA/smoke/formal action executed for this study. Its CPU attempt is already complete; do not run `test` again. Old R2 approval is not accepted.

## Resolve authority before production

Use a clean detached checkout of the exact new approved commit, separate from original AGMS execution. Keep real approval, review evidence and user launch receipt outside it. There is no shipped APPROVED template. Existing external review requirements are preserved in authority.py.

The real review must name study_id AGMS_DS_HALF_V1, reviewer_role external, non-template decision APPROVED_FOR_EXPERIMENTS, reviewer and evidence; approve phases `[CUDA, SMOKE, P1]`; bind exact CAPS from EXECUTION_PLAN and reviewed_code_commit, code_tree_sha256, plan_sha256, prefix_sha256, import_sha256, environment_sha256, execution_sha256. import_sha256 is the digest of `{A0:plan.imports,A5_CONTROL:plan.control_imports}`. Each value is recomputed by `preflight`; do not hand-rewrite it to admit differences.

After genuine approval, record the user's already supplied autonomous delegation in a separate receipt: study_id, user_confirmed=true, prompt=AUTONOMOUS_DS_HALF_V1, real review canonical digest as review_sha256, and the same actual fields. Preserve the literal user delegation/source timestamp. This does not claim user authored the scientific selection or externally reviewed code. The user's permission exists; the missing prerequisite is independent review, not another routine permission question.

Copy the existing private runtime config to an independent authority directory and bind `code`, `execution_commit`, new create-only `run_root`, real `review` and `launch_confirmation`. Reuse reference/data/prefix_root exactly; never put these private absolute paths in public Git. Same environment, no pip/conda upgrades, no PYTHONOPTIMIZE/python -O. Check optimize=0 and frozen environment exactly.

## Sequential production entry

All large artifacts on the canonical project NAS. Verify actual NFS mount, free capacity and one real write/read probe. Use the existing `with_nas_storage.sh` wrapper and neutral env dispatcher. Limit visible GPUs to one of5/6/7 after memory check; do not alter other processes.

Set TASK_ENTRY to a small private neutral dispatcher outside checkout. It reads private TASK_CONFIG and calls the module's main with each phase in a fresh process. Actual visible argv can use:

```python
import os
exec(compile(open(os.environ['TASK_ENTRY']).read(), 'worker', 'exec'))
```

Inside that dispatcher use the existing module entry with:
1. `qualify --mode cuda --config <external config>`:11 fully synthetic native calls,2 predeclared after-optimizer injections charged. No prefix/data reads.
2. `qualify --mode smoke --config <external config>`: real acceptance of both original B2 stage1 prefixes,4 initial L-only updates in each order,8 total. No U/val/test; discard states. Do not run `p0`: original26-L screening is bound historical metadata only.
3. `run --config <external config>`: two nodes,5300 formal calls,32 extra diagnostic VJPs. No old AGMS run root and no alternative arms.

Each nonzero exit stops the sequence and preserves log/ledger/session/failed state. No retry, no tail replay, no partial-qualification reset. Resume only where existing exact-cursor and closed-session rules permit. The new failure budget is not transferable between CPU/CUDA/smoke/formal. Existing finite controller owns the run lock and actual-model final verification.

## Exit and publication

Require both target receipts SEALED, both models VERIFIED,5300 scientific/physical calls,8 smoke,11 synthetic CUDA,32 diagnostic VJPs,all sessions closed and6/18/9/8 report coverage. A0 and old A5 imports remain historical. Report DS_PRESSURE_SUPPORTED and all order/domain differences regardless of sign; no follow-up nodes automatically generated.

Publish only public aggregates/source/config and cost metadata from an independent release checkout, verify remote SHA plus anonymous access. Do not publish patient data, images/labels, weights, private evaluation records or paths. Keep the original AGMS completion/review evidence unchanged.
