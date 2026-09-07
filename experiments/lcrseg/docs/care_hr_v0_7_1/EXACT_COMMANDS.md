# Reproduction commands for this blocked preflight

All commands are source/synthetic only. Use a new isolated checkout at the exact source SHA in `SOURCE_LINEAGE.json`. Keep output outside the checkout. On the server the existing NAS wrapper must enclose execution, and `NEW_TEST_OUTPUT` must be a create-only NAS location. Python versions and packages are recorded in `SYNTHETIC_TEST_REPORT.json`; no packages were installed.

```bash
git worktree add -b codex/care-hr-v0-7-1-capacity-audit NEW_WORKTREE 61c1e302fae515fe51adf4e777887a1a489859be
# Run from a clean committed source checkout:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg python -m care_hr_v0_7_1.run_checks --output NEW_TEST_OUTPUT
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg python -m care_hr_v0_7_1.semantic_audit
# The audit intentionally returns 2 with BLOCKED_EVALUATOR_SEMANTICS_MISMATCH.
# Server test launch (existing interpreter; outputs resolved to NAS):
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=experiments/lcrseg bash experiments/lcrseg/scripts/with_nas_storage.sh EXISTING_PYTHON -m care_hr_v0_7_1.run_checks --output NEW_NAS_TEST_OUTPUT
```

The first local development command ran the three pytest directories directly and found two workspace-state static-audit failures before source commit. After committing to a clean worktree the exact-source suite passed; old guards were not edited. The final wrapper records the expanded pytest arguments through its fixed `argv` and JUnit receipt.

Direct server HTTPS clone failed with GnuTLS receive error -110. The source was instead transferred using a Git bundle of the published branch, streamed to a create-only NAS file through the storage wrapper, then cloned into a new home source checkout. Git reported the same source commit. This was source transport, not input substitution. The attempted cancellation found the already-exited clone; no process was killed. A server-local parent persisted the actual test child exit 0 separately from SSH status. Raw test logs and JUnit remain on NAS; only sanitized summaries are published.

No capacity/GT execution command exists in this partial source. No future fit or automatic-resume command is authorized by this document.
