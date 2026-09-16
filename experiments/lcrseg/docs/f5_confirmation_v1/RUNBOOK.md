# F5_CONFIRMATION_V1 runbook

**Current state: STOP_AWAITING_EXTERNAL_CODE_REVIEW.** This submission permits preparation and finite CPU synthetic checks only. No real optimizer update, smoke, CUDA qualification, training, or monitoring is authorized. The RUN prompt in the input archive is a future phase, not current launch authority. Old USER_AUTHORIZATION.json is explicitly rejected.

## Implemented commands

From the new branch checkout, with the existing Python/PyTorch environment:

```bash
python -m experiments.lcrseg.f5_confirmation_v1 prepare
python -m experiments.lcrseg.f5_confirmation_v1 plan
python -m experiments.lcrseg.f5_confirmation_v1 qualify --mode cpu
python -m experiments.lcrseg.f5_confirmation_v1 report --config "$PRIVATE_CONFIG"
```

`prepare` checks unchanged shared training code, binds complete sealed options, emits the finite plan, source metadata report, P0 outputs and code manifest. It does not read tensors or patient payloads. `plan` rejects any matrix, options, import or cap change. `report` reads aggregate receipts only, returns PENDING_FULL_MATRIX before completion and does not decide gates early.

`qualify --mode cpu` runs the nine fixed test groups in tests.py with synthetic tensors and no production permit. Cumulative CPU calls and all invocations (including failures) remain in CPU_PHYSICAL.jsonl and CPU_ATTEMPTS.json. This preparation protocol caps cumulative CPU calls at 40 and invocations at 8; do not erase ledgers to repeat tests. Temporary synthetic checkpoints must use a NAS TMPDIR on the server. CUDA_VISIBLE_DEVICES is empty. CPU success does not grant execution permission.

## Future production commands — NOT RUN in this submission

These commands are implemented, but require authentic external review plus separate explicit user launch confirmation, both bound to the exact submitted code and plan:

```bash
python -m experiments.lcrseg.f5_confirmation_v1 qualify --mode cuda --config "$PRIVATE_CONFIG"
python -m experiments.lcrseg.f5_confirmation_v1 qualify --mode smoke --config "$PRIVATE_CONFIG"
python -m experiments.lcrseg.f5_confirmation_v1 run --config "$PRIVATE_CONFIG"
```

Do not launch these directly with a method-bearing process command line. After future authorization, run via the existing NAS storage wrapper and neutral environment-based entry below. Private config supplies `code`, `execution_commit`, `run_root`, `reuse_root`, `data`, `reference`, `dependencies`, `review`, `launch_confirmation`. Reuse the previously known private runtime config: its old `run_root` becomes `reuse_root`; keep its data/reference/dependencies unchanged; choose a new create-only protocol root on the canonical NAS. Pin `code`/`execution_commit` to the reviewed new checkout. No private paths or credentials belong in Git.

```bash
export EXEC_MODULE=experiments.lcrseg.f5_confirmation_v1
export EXEC_CONFIG="$PRIVATE_CONFIG"
export EXEC_ARGS='["qualify","--mode","cuda"]'
export STORAGE_WRAPPER="$CHECKOUT/experiments/lcrseg/scripts/with_nas_storage.sh"
export EXEC_CODE='import os,json,runpy,sys;sys.argv=["runner"]+json.loads(os.environ["EXEC_ARGS"]);runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'
# PYTHON is the existing server interpreter. The source command immediately execs it.
bash -c 'source "$STORAGE_WRAPPER" "$PYTHON" -c "$EXEC_CODE"'
```

Only after that CUDA qualification passes, set EXEC_ARGS to `["qualify","--mode","smoke"]`. Only after smoke passes, set it to `["run"]`. Use the established background session for long execution and verify its neutral full argv with ps and nvidia-smi. Do not start or restore a heartbeat unless separately requested in that future round.

The simple controller is serial and selects an available GPU from physical 5/6/7 with at least 12 GB free; it does not claim three simultaneous workers. One lock owns the finite queue and separate cost ledgers. No competing process is killed. Insufficient GPU/NAS capacity stops admission for operator follow-up, not silent retry. The NAS wrapper plus controller mount/write/free-space check are required before large writes.

## Required authentic approval fields

No approval or launch receipt is shipped. The external reviewer supplies:

- `study_id`: F5_CONFIRMATION_V1; `is_template`: false; `decision`: APPROVED_FOR_EXPERIMENTS; `reviewer_role`: external; named `reviewer` and nonempty `review_evidence`.
- `approved_phases`: [P1]; `caps`: exactly PLAN.json caps, including zero new source updates.
- `reviewed_code_commit`: exact Git HEAD; `code_tree_sha256`: CODE_MANIFEST.json value; `plan_sha256`: PLAN.json value; `source_reuse_sha256`: canonical digest of SOURCE_REUSE.json; `parent_binding_sha256`: canonical digest of the historical PARENT_BINDING.json.

The distinct user launch receipt has `study_id`, `user_confirmed`: true, all five exact binding fields above, and `review_sha256` (canonical digest of the authentic review). The JSON format is an evidence binding, not cryptographic proof of reviewer identity. The executing operator must establish provenance; do not manufacture these fields or treat the agent as the external reviewer. Any review/code change requires rebinding and relevant CPU rechecks within the separate cap.

## Production preflight PENDING

Source metadata/file existence was checked without loading weights. Future `qualify` verifies three real source tensor hashes against declared receipt hashes, strict native state schema and a synthetic zero-input forward. It makes read-only source reuse links in the new root; source training is unreachable. Source tensor checks, native CUDA behavior, actual smoke, GPU headroom and production recovery are **PENDING**, not claimed as passed.

CUDA qualification uses the actual native parent with generated data: B0/B2/F5, four updates each = 12 planned calls, cap 60. Current-L-only smoke uses disposable source-initialized copies, eight per method = 24 calls. The original 3200-step warmup schedule stays intact during smoke; U payloads are disabled. Neither qualification state is inherited by formal targets. Failures and partial attempts retain their physical ledgers and stop; there is no automatic retry to spend unused qualification headroom.

Formal targets call the unchanged native_runner.target_task. It preserves native construction, stage reset, dense current EMA, original loaders/losses/random streams, deployment, and isolated seen-val evaluation. F5 loss is exactly `L_supervised + L_parent_constraint + ramp*0.25*(KL+0.2*SWD)`; A/B lr=.0005 and R lr=.001. No additional diagnostic forward/VJP/RNG call is introduced; existing detached receipt fields are exported, missing fields are NA.

## Recovery and completion

Formal caps are 74,200 scientific and physical calls; per-stage caps are exactly 3200 or 2100. A durable invocation is recorded by the existing Counter before every optimizer call, including failed attempts. Sealed receipts validate and skip idempotently; no overwrite. An unsealed checkpoint may resume only if saved step/cursor and the complete durable physical count agree. Native checkpoint.restore then checks and restores original model/EMA/optimizer/scheduler/prototypes/provider counters/Python+Torch+CUDA RNG state. Any lost tail consumes calls without committed scientific progress, so replay cannot fit the frozen per-stage cap: ENGINEERING_STOP. Failed nodes are not automatically retried or granted smoke budget.

The second target inherits only its own method/config/seed/order stage-one student/F. B0/F5 seed162 import eight historical target receipts solely for descriptive analysis; they never execute. The controller has 28 targets and no source, selection, old five-framework DAG, P2 or extension nodes.

After all 28 seal, PUBLIC_RESULTS.json contains normalized full stage metrics, primary 163/164 pairs, supplementary 162–164 pairs, all costs and G1–G3 decisions. Each seed first averages its two orders. Failure of a budget-decision gate is a scientific outcome, not an engineering stop; all outcomes are retained. Passing recommends P2 review only. No additional experiment or patient-independent/SOTA claim follows automatically.
