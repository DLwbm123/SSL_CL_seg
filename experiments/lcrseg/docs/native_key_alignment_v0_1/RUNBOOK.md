# R1 runbook — no production run authorized

Current stop: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. Do not run CUDA/smoke/D1 from this commit's CPU PASS or from the old F5 review. Do not rerun the exhausted old D0 suite or the now-exhausted supplemental two-attempt suite. Remaining physical budget is not extra invocation authority.

## Actual CPU command and environment

Both supplemental attempts used the same neutral process entry below. Variables were set in the shell environment, so no project/method/private path appeared in the Python command line; `ps -ww -p PID -o pid,args` checked this after launch. No GPU process was launched. Logs, generated temporary tensors and evidence went through the existing NAS wrapper. The server had Python3.10.6, PyTorch2.2.1+cu121, optimize0, twoCPU threads; CUDA_VISIBLE_DEVICES was empty and no dependencies changed. Worktree started at9915168 plus the submitted source overlay; each attempt recorded its actual full code-tree digest.

Private values are intentionally omitted here. `TASK_MODULE` was `experiments.lcrseg.native_key_alignment_v0_1.integration_tests`; `TASK_REFERENCE` points to the previously pinned JASCL source-code directory, never weights. `TASK_EVIDENCE` is the new isolated NAS supplemental directory; `TASK_WRAPPER` is this checkout's `experiments/lcrseg/scripts/with_nas_storage.sh`; `TASK_PYTHON` is the existing server interpreter. Never replace HOME or CODEX_HOME.

Actual process command, with environment values supplied privately:

```sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
# TASK_MODULE, TASK_REFERENCE, TASK_EVIDENCE, TASK_WRAPPER, TASK_PYTHON set privately
bash -c 'source "$TASK_WRAPPER" "$TASK_PYTHON" -c '\''import os,importlib,json;from pathlib import Path;m=importlib.import_module(os.environ["TASK_MODULE"]);m.LEDGER=Path(os.environ["TASK_EVIDENCE"]);print(json.dumps(m.run_tests(os.environ["TASK_REFERENCE"]),indent=2))'\''' >"$TASK_EVIDENCE/attempt1.log" 2>&1 &
TASK_PID=$!
ps -ww -p "$TASK_PID" -o pid,args
wait "$TASK_PID"
```

Attempt2 used `attempt2.log` and the same evidence directory, retaining attempt1. Each suite durably declared six cases and17 planned calls before native construction: canonical/runtime0; B2/C0/C3-zero6; C2/C3 continuous/resume10; after_optimizer failure1; generated integrity/cost0; generated reports0. Shared cost tests use3 gradient-only backward calls per attempt; diagnostic31 VJPs per attempt are included in60 autograd.grad calls. Two immutable attempt reports plus34 physical ledger rows are retained. All originalD0 evidence stays unchanged.

## Future external review and private configuration

These are interface definitions for a later separately authorized run, not commands to execute now.

A genuine new external review must identify this study and final commit, reviewer/evidence, `is_template=false`, `reviewer_role=external`, `decision=APPROVED_FOR_EXPERIMENTS`, only `approved_phases=[D1]`, exact caps, and:

- `reviewed_code_commit`: actual reviewed clean HEAD;
- `code_tree_sha256`: CODE_MANIFEST.json, all experiments Python sources;
- `plan_sha256`: unchanged canonical science digest;
- `prefix_sha256`: canonical digest of the four prefix records;
- `execution_sha256`: canonical digest of EXECUTION_PLAN.json.

Independent user launch confirmation must carry study_id, `user_confirmed=true`, `review_sha256` of that actual review, and all five bindings above. The implementation only validates these documents; this task generates neither. Store approval, confirmation and private runtime configuration outside the reviewed clean checkout. Preserve the actual R1 rejection and all historic approvals.

Private config fields: `code`, `execution_commit`, `reference` (pinned source code), `data`, `prefix_root` (existing F5 P1 outputs), `run_root` (new non-overlapping NAS protocol root), `review`, `launch_confirmation`. Fresh root ownership uses existing NAS checks/lock/run identity. No arbitrary source/prefix substitutions or path override flags exist. The clean checkout and exact code manifest are required; do not update status inside that checkout during execution.

## Future finite sequence, only after separate approval/launch

Use the existing NAS wrapper, neutral environment entry and the already reviewed server environment. Shared GPU choice uses5/6/7 with enough actual free memory; confirm one startup command line and GPU listing after a future authorized launch. No automatic monitoring is provided.

Set TASK_MODULE to this package, TASK_CONFIG to private JSON, TASK_COMMAND to one of `qualify`, `run`, `report`, and TASK_MODE only for qualify. A neutral production entry can use:

```sh
# NOT AUTHORIZED NOW. Run only with later genuine review and user launch receipt.
# Wrapper and interpreter are passed through environment exactly as in CPU execution.
"$TASK_PYTHON" -c 'import os,sys,runpy;sys.argv=["worker",os.environ["TASK_COMMAND"],"--config",os.environ["TASK_CONFIG"]];sys.argv+=(["--mode",os.environ["TASK_MODE"]] if os.environ["TASK_COMMAND"]=="qualify" else []);runpy.run_module(os.environ["TASK_MODULE"],run_name="__main__")'
```

1. `qualify --mode cuda`: generated native384 input only, fourarms × warmup1/resume4/failure1 =24 calls. Four failures are intended after_optimizer injections; all physical calls count. No real prefix access. Exact coverage/count/environment receipt required.
2. `qualify --mode smoke`: CUDA receipt/cost acceptance first; bound actual B2 prefix file/schema/identity/hash acceptance; fourarms seed163/O1 ×8 currentL-only =32. Formal first8 warmup semantics, noU/val/test. All states discarded. The other prefix records are accepted before their formal nodes. Missing or mismatched prefix stops without training it.
3. `run`: both qualification receipts, current environment and closed costs before16 stage2 nodes. Actual prefix validation before each fresh/resumed node, actual target-file validation after each node and sealed-skip. RIM3200/Drishti2100; exact42400 formal/physical cap, no source/stage1 training. Physical real total<=42432. Lost tail, dirty checkout, failed/unclosed attempt, OOM/nonfinite or mismatch stops, preserves evidence, no automatic repair/retry.
4. After16 nodes, generated aggregate outputs: FINAL_REPORT.md, FINAL_METRICS.csv(16), DOMAIN_METRICS.csv(48), PAIRED_COMPARISONS.csv(21), DIAGNOSTICS.json, COST_AND_COMPLETION.json, PUBLIC_RESULTS.json. Completion is D1 scientific-review waiting; D2/D3 are never launched. Any later public results must exclude private_val, model tensors, images/labels, sample IDs, private paths and credentials.

CUDA/smoke fixture PAS0/-1, foreground exposure and5-step CUDA total never enter formal scientific options. Qualification, prefix acceptance, formal, integrity and CPU cost categories are distinct; operation and semantic counters overlap, so do not add subsets twice.
