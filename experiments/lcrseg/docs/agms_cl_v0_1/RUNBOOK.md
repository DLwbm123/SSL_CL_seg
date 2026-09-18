# AGMS_CL_V0_1 runbook

Current stop: **STOP_AWAITING_EXTERNAL_CODE_REVIEW**. Only Prompt A was executed.
Do not execute any future command until the final commit has passed independent external
review AND the user separately forwards Prompt B. No APPROVED artifact/template is shipped.
Do not re-run prepare after review, update approved docs, reuse an old review, or drift HEAD.

## Preparation

**Final CPU qualification is blocked.** Three attempts used2,28,22 physical calls
(total52/96); attempt2 passed, attempt3 stopped on a generated-fixture directory collision.
The fixture now uses per-attempt directories, but no fourth attempt is permitted. The
latest FAIL and every older report remain intact; its code digest is not silently rebound
to the repaired code. Production preflight deliberately refuses the current FAIL report.
A reviewed repair and separately authorized CPU qualification scope are necessary before
any future production readiness claim. Prompt B alone cannot bypass this prerequisite.

CLI commands: `prepare`, `plan`, `test --reference PATH --evidence PATH`,
`p0 --config PATH`, `qualify --mode cuda|smoke --config PATH`,
`run --config PATH`, `report --config PATH`.

CPU tests use the existing local interpreter and a detached source-only upstream checkout
at `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`; no packages were installed or upgraded.
Generated checkpoints/logs remain outside the repository. `CPU/TEST_REPORT.json` records
actual CPU environment; it is not a claim of matching the frozen production GPU environment.
Old CPU134 remain in their own ledgers. Every new attempt predeclares28 calls, maximum3
attempts/cumulative96; physical calls include failed attempts. No old suite is run.
Published CPU records cannot be reset by selecting a new evidence directory.

For a future authorized process, use environment values with a neutral visible argv.
Set TASK_MODULE to the new package, TASK_COMMAND to a CLI command, TASK_CONFIG privately,
TASK_PYTHON to the existing interpreter, TASK_WRAPPER to the original NAS wrapper.
Do not put project names/private paths into process argv; inspect `ps` and GPU process
names once after launch. The private environment variables and paths are not published.

```sh
# Future production only; this example grants no permission.
# TASK_ENTRY holds the generic Python dispatcher below, so argv contains no paths.
export TASK_ENTRY='import os,sys,runpy;sys.argv=["worker",os.environ["TASK_COMMAND"],"--config",os.environ["TASK_CONFIG"]];sys.argv+=(["--mode",os.environ["TASK_MODE"]] if os.environ["TASK_COMMAND"]=="qualify" else []);runpy.run_module(os.environ["TASK_MODULE"],run_name="__main__")'
bash -c 'set -- "$TASK_PYTHON" -c "$TASK_ENTRY"; source "$TASK_WRAPPER"'
```

Use the already-established neutral NAS launcher if available; do not change storage roots
or skip NFS/free-space/read-write probes. The unchanged storage wrapper's filesystem probe
subprocesses may show storage paths briefly; the long-running worker and training children
must use neutral argv. No remote access happened in preparation. GPU selection uses the
unchanged finite chooser for5/6/7 and available memory; availability is not claimed now.

## Future authority

`authority.preflight` requires clean reviewed HEAD and manifest plus a genuine external
review: study AGMS_CL_V0_1, external reviewer/evidence, is_template=false,
decision APPROVED_FOR_EXPERIMENTS, approved_phases=[P0,P1], exact budgets.
This describes the validator, not an approval document.

Both review and independent user receipt bind reviewed_code_commit, code_tree_sha256,
plan_sha256, prefix_sha256, import_sha256, environment_sha256, execution_sha256.
The user receipt also requires study_id, user_confirmed=true, prompt=B and canonical
review_sha256 of the real approval. Canonical digests use the repository helper.
Authority/config files remain outside checkout. Unexpected environment differences stop;
never regenerate a fingerprint to conceal a change.

Private config keys: code, execution_commit, reference, data, prefix_root, run_root,
review, launch_confirmation. Source/read-only prefix root and new NAS run root must not overlap.
The model and dataset identifiers come from PREFIX_BINDINGS and frozen hashes, not discovery
or subjective model choice. CPU qualification must match the current transitive code manifest.

## Future finite sequence (all PENDING)

1. P0: actual prefix integrity and fixed current-L opportunity counts, ≤32 images and0 updates.
   P0 is descriptive only and cannot prune the matrix. No U/val/test data are opened.
2. CUDA qualification: native generated384 data only; six arms ×5 warmup/continuous/resume
   calls=30, old B2/A0 equivalence4, preset failures2; total36. No real source/prefix is read.
3. Smoke: CUDA evidence required, both original prefix files accepted; A0–A5 each4 L-only
   formal warmup calls on O1=24. Discard all states, no U or evaluation access.
4. Run: recheck P0/CUDA/smoke/environment and finite budgets, then ten original-prefix
   stage2 nodes. Each sealed/skip path needs actual student file verification. No other arms,
   seeds, phases, first targets or source updates. Scientific scores never prune or tune.
5. Report: require all models/cost sessions/ledgers/40 diagnostics; export FINAL_REPORT,
   FINAL_METRICS12, DOMAIN_METRICS36, PAIRED_COMPARISONS27, GRADIENT_DIAGNOSTICS40,
   COVERAGE_AND_RISK, P0_REPORT, COST_AND_COMPLETION, PUBLIC_RESULTS, COMPLETION_PROOF.
   Final state COMPLETE_AGMS_P1_AWAITING_SCIENTIFIC_REVIEW; no automatic follow-up.

Missing authority, dirty code, wrong prefix, invalid identity, nonfinite/failed qualification,
unclosed cost session, physical tail or failure receipt stops before more work. No automatic
retry, source retraining, tolerance relaxation or dependency change. Production is unexecuted;
external review should inspect the future paths, not treat CPU success as production proof.

Public artifacts include only aggregate reports and necessary metadata. Keep model tensors,
images/labels, per-patient counts, private_P0/private_val, credentials and private paths out.
Use an independent results publication location; do not modify a reviewed execution checkout.
