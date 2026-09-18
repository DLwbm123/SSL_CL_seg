# NKA_DOSE_V0_1 runbook

Current state: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. This delivery executes Prompt A only. No current command below grants production authority. Do not run Prompt B, CUDA, smoke, formal training, old suites or additional CPU attempts now. New CPU has exhausted its three attempts (70/96 calls); the old64 remain unchanged. Do not run prepare again after review to update approval status.

## Preparation actually performed

The isolated checkout starts at `145b3c1b3ab301031ea379c9d67452e57b28ae1d`, with only the new dose package/docs overlaid. The existing Python3.10.6 / PyTorch2.2.1+cu121 interpreter was used without dependency changes. Two CPU threads, FP32, optimize0, empty CUDA_VISIBLE_DEVICES and CUDA uninitialized are recorded in every report. The generated native model uses the bound JASCL source-code directory, never real weights. Each attempt declares29 calls before construction, with case-specific hard physical caps; actual totals12,29,29 are retained.

The NAS wrapper verified its actual NFS mount and a create/write/read probe; about30TiB was available at preparation. Generated checkpoints, temporary files and operation records remain outside Git on NAS. The Python process command was inspected with ps and has no project/method/private paths. Public evidence consists of bounded aggregate reports and the physical ledger, not generated checkpoint tensors or raw runtime logs.

The following records the used interface; it is **not a request to rerun it**. Private environment values are omitted. TASK_MODULE was `experiments.lcrseg.nka_dose_v0_1`. TASK_REF identifies the existing pinned source code; TASK_EVIDENCE is the single preserved new CPU ledger directory on NAS; TASK_PYTHON is the existing interpreter; TASK_WRAPPER is the checked-out `experiments/lcrseg/scripts/with_nas_storage.sh`.

```sh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
# During preparation only, before CPU testing: freeze current metadata/manifest.
bash "$TASK_WRAPPER" "$TASK_PYTHON" -c 'import os,importlib;p=importlib.import_module(os.environ["TASK_MODULE"]+".protocol");a=importlib.import_module(os.environ["TASK_MODULE"]+".authority");p.freeze();p.write(p.DOC/"EXECUTION_PLAN.json",a.execution_plan());p.write(p.DOC/"CODE_MANIFEST.json",p.code_manifest());t=importlib.import_module(os.environ["TASK_MODULE"]+".tests");print(t.run_tests(os.environ["TASK_REF"],os.environ["TASK_EVIDENCE"])["status"],flush=True)'
```

The CLI exposes `prepare`, `plan`, `test --reference ... --evidence ...`, `qualify --mode cuda|smoke --config ...`, `run --config ...`, and `report --config ...`. Use the neutral environment entry below for a later authorized process, avoiding sensitive paths in visible argv. `plan` is metadata-only. The test command refuses a fourth invocation and a ledger reset relative to published attempt history.

## Required future authority and configuration

A real external review must bind the actual clean final commit, the exact study NKA_DOSE_V0_1, reviewer/evidence, `is_template=false`, `reviewer_role=external`, `decision=APPROVED_FOR_EXPERIMENTS`, only `approved_phases=[DOSE]`, and the exact caps from PLAN. No such document is shipped.

Required bindings in both review and later independent user launch receipt:

| Field | Value source |
|---|---|
| reviewed_code_commit | Actual reviewed clean HEAD |
| code_tree_sha256 | CODE_MANIFEST.json |
| plan_sha256 | PLAN.json |
| prefix_sha256 | Canonical JSON digest of PLAN.prefixes |
| import_sha256 | Canonical JSON digest of PLAN.imports |
| environment_sha256 | Canonical JSON digest of PLAN.environment (wrapper including fingerprint and its digest) |
| execution_sha256 | Canonical JSON digest of EXECUTION_PLAN.json |

The independent user receipt also needs study_id, `user_confirmed=true`, and review_sha256 of the genuine review object. Digests use the repository canonical digest helper, not file whitespace hashes. A runtime environment receipt instead records the digest of the actual fingerprint itself; these two environment digest scopes are deliberately distinguished. Never fabricate review conclusions, recycle old permits or rewrite an unexpected runtime fingerprint. Keep review, launch receipt and private config outside the reviewed checkout; do not create a new commit to store them.

Private configuration fields (all required):

| Field | Meaning |
|---|---|
| code | Reviewed clean checkout |
| execution_commit | Its full approved SHA |
| reference | Existing pinned upstream source code, no substitute implementation |
| data | Existing bound manifest/split/data root |
| prefix_root | Existing original B2 stage1 artifact parent; preserve identities |
| run_root | New non-overlapping create-only NAS protocol directory |
| review | Genuine new external review file |
| launch_confirmation | Separate receipt of the later user's actual launch confirmation |

NAS root locking/ownership, free-space/probe checks, old binding checks and clean code/CPU hashes are required by the runtime. The controller selects one of GPUs5/6/7 with at least12GB currently free and verifies the frozen actual environment. It is a single finite controller; do not start duplicate launchers. No continual monitoring or automatic restart is provided.

## Later production sequence — pending separate authorization

After external approval plus the later user's explicit launch authorization, set TASK_CONFIG privately, TASK_COMMAND to `qualify`, `run` or `report`, and TASK_MODE to `cuda` or `smoke` only for qualify. Run from the reviewed checkout using the unchanged NAS wrapper:

```sh
# NOT AUTHORIZED BY PROMPT A.
bash "$TASK_WRAPPER" "$TASK_PYTHON" -c 'import os,sys,runpy;sys.argv=["worker",os.environ["TASK_COMMAND"],"--config",os.environ["TASK_CONFIG"]];sys.argv+=(["--mode",os.environ["TASK_MODE"]] if os.environ["TASK_COMMAND"]=="qualify" else []);runpy.run_module(os.environ["TASK_MODULE"],run_name="__main__")'
```

1. `qualify --mode cuda`:40 physical generated calls at native384; no real prefix access. Four cells × warmup1+continuous2+restore2+after_optimizer failure1 =24; C2/C3 each old0.05/new0.05 meter-on/off two updates =12; four small Adam oracles =4. Old comparison capability is generated-only and cannot open real data; no old production approval is forged. Every case/range, environment and ledger is checked. Fixture foreground bias, PAS0/-1 and five-step schedule are qualification-only.
2. `qualify --mode smoke`: accept CUDA receipt/costs first, then actual original seed163/O1 B2 prefix identity/schema/finite/file/student/F hashes. Run four dose cells ×8 current L-only updates with formal first-eight warmup semantics. U, val and test are not accessed; all smoke model/EMA/optimizer state is discarded. Any actual prefix mismatch blocks reuse and never causes source training.
3. `run`: validate both receipts and closed sessions, then each of16 stage2 nodes. Reaccept original B2 prefix before fresh or exact own-node resume. Never warm-start from another NKA/dose endpoint. RIM3200/Drishti2100;42,400 formal scientific/physical cap;32 smoke cap;42,432 real total. Actual target integrity runs after each sealed node and on skip, not a metadata-only substitution. Missing tail, partial qualification, failed attempt, unclosed cost, wrong identity, nonfinite state, OOM or Adam mismatch stops with evidence. No retry, tolerance relaxation, dependency replacement or additional budget is implied.
4. Completion exports FINAL_REPORT.md, FINAL_METRICS.csv(28), DOMAIN_METRICS.csv(84), PAIRED_COMPARISONS.csv(63), GRADIENT_DIAGNOSTICS.json, ADAM_COUNTERFACTUAL.json(64), COST_AND_COMPLETION.json, PUBLIC_RESULTS.json and COMPLETION_PROOF.json. The standalone report command requires the same authority and clean checkout; unfinished models return PENDING. Historical12 keep original provenance and absent Adam diagnostics. All16 complete before G_perf/G_key are evaluated.

Formal64 diagnostic points entail256 additional VJPs and128 read-only candidates, not additional optimizer updates;33,920 extra clean-U forwards remain distinct. Candidate arithmetic/effective-W counts and time are separate but nested within overall meter/session cost. CPU history, CUDA, smoke, formal, evaluation and integrity costs remain distinguishable; do not add overlapping telemetry subsets twice.

Final runtime status is COMPLETE_DOSE_AWAITING_SCIENTIFIC_REVIEW. If later authorized to publish results, use an independent publication checkout: publish only aggregate reports/necessary metadata, verify anonymous access, and preserve the reviewed execution checkout. Exclude weights, images/labels, per-patient records, private_val, private filesystem paths and credentials. LOCAL_REPORT_READY does not mean public delivery. No D2/D3 or full-trajectory follow-up is started automatically, even if a dose passes both development gates.
