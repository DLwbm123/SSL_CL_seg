# Revisable unlabeled analytic segmentation pilot

See [frozen protocol](PROTOCOL_ZH.md) and [Chinese results](results/REPORT_ZH.md).
This is an unsuccessful bounded pilot, not a validated TMI method: source RBF
readout calibration loses substantial segmentation ability and the revisable
arm does not beat fixed responsibilities after the complete sequence.

The feature/role/Dice interfaces reuse the existing project engine; analytic
updates are inference-only block solves, not a separate SGD training loop.
`core.py` defines the exact conditional objective, `run.py` performs the two-stage
sequence, `test_contract.py` contains small analytic checks, and `summarize.py`
audits receipts without rerunning inference. `diagnose.py` records one optional
post-run geometry check; it was executed once in this pilot, not automatically
repeated. `entry.py` dispatches these modes from a private JSON config.

Required private config fields: `output`, `code`, `support`, `base`, `data`,
`reference`, and `source_checkpoint`. They identify a create-only output job,
current source tree, previously deployed SPACE support tree, common project
engine tree, frozen role-bound data, pinned model reference, and completed source
EMA checkpoint. These paths and private artifacts are deliberately not published.

Launch only after applicable server/storage authorization: copy `entry.py` to a
neutral operational filename; set `TASK_SETTINGS`, `TASK_MODE=run`,
`CUDA_VISIBLE_DEVICES=3`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; use the project's
approved storage wrapper and existing PyTorch environment. This documentation
does not authorize a new run. Formal solve cap 82, engineering cap 16, actual
82+4; no SGD, test access, hidden U labels, tuning, or automatic retry.

Public results contain aggregates only. Private endpoint heads, summaries,
patient records and exposure logs remain on approved server storage.
