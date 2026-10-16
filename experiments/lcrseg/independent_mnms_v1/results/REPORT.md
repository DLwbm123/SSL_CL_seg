# Siemens independent training: completed, weak validation performance

Both seeds completed the frozen 200 epochs with exit code0 and7800 physical
optimizer updates each. All200 training records and20 validation records per
seed are present; final/best/latest checkpoint files exist and have nonzero
sizes. This audit read existing receipts and aggregates only: no checkpoint
reload, extra inference, retry, tuning, or test evaluation was performed.

|Seed|Final Dice %|LV %|MYO %|RV %|Best validation Dice %|Best epoch|
|---|---:|---:|---:|---:|---:|---:|
|163|60.136684|71.359929|60.271646|48.778476|62.435296|160|
|164|63.317806|74.543198|64.754796|50.655424|63.317806|200|
|Mean|61.727245|72.951564|62.513221|49.716950|62.876551|NA|

The fixed epoch200 result is primary. Best-validation results are secondary
and selected on these same validation cases, not independent confirmation.
Metrics cover23 Siemens patients/46 ED/ES volumes, averaged volume-to-phase,
patient, then foreground class, with predictions restored to the full canvas.
Validation foreground retention inside the fixed crop was100% for both seeds.
This rules out lost ground-truth foreground from this crop as the explanation
for these scores; it does not establish that the preprocessing is optimal.

Previously recorded V1 source results (reference commit19f30387, included
v1_stage_reference.json) were65.817577% and74.812923%, mean70.315250%.
The present mean is8.588005 percentage points lower. This is not a controlled
ablation: architecture, supervision, sampling and update budgets differ.
This run did not deliver the intended strong independent baseline; completion
of execution must not be presented as scientific success or paper reproduction.

Final training CE+soft-Dice losses were0.180245 and0.127310. These training
losses are not validation hard Dice. Validation fluctuated late in training;
seed163 peaked at160 and declined2.298612pp by200, while seed164 peaked at200.
The data are consistent with a generalization problem, but do not identify
its cause or prove that simply increasing epochs would resolve it. No causal
claim or new experiment is justified by this status check alone.

Formal cost:15600 optimizer updates,92000 labeled training slice exposures,
21760 validation slice forwards, no U or independent-test access. Recorded
seed runtimes1001.716s and1039.837s (about34.03minutes total); peak allocated
memory about2.013GB. Separate engineering consumed67 GPU optimizer updates
(64 in the preserved failed short-fit attempt,3 in the successful minimal
restore check); five CPU tests include3 synthetic optimizer updates, outside
formal/GPU accounting. No retries or uncommitted formal updates were reported.

Source and frozen settings are in the parent directory. RESULTS.json contains
aggregate receipts and validation history; VALIDATION_CURVES.csv provides the
same aggregate curves. Patient-level records, source data, raw logs, and
checkpoints remain private. No new run or recurring monitor was started.
