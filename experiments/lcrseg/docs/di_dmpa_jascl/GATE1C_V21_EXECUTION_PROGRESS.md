# Gate1C v2.1 execution progress

Latest terminal observation: **BLOCKED_INCOMPLETE_EVIDENCE**, exit1 at
2026-08-30T16:06:46.458937+00:00. The original decomposition guard failed on
B0/seed0/stage2/Drishti_GS/pair01 after 9/9 validation metrics and 25/72 draw0
pairs. All workers exited. Earlier RUNNING snapshots below are historical.
See [bounded failure investigation](GATE1C_V21_DECOMPOSITION_INVESTIGATION_PREREGISTRATION.md).
No full retry or scientific admission has occurred.

Follow-up completed 2026-08-30T16:23:40 UTC: the same native failure reproduced
once on each GPU, with identical numerical details. See the
[fixed-pair investigation report](GATE1C_V21_DECOMPOSITION_REPORT.md).
Both forensic processes have exited; none of the historical PIDs below is an
instruction to resume or repeat an occupied attempt.

Follow-up completed 2026-08-30T16:43:52 UTC: the independently preregistered
same-draw float64 reference is supported on both GPUs, with complete after-error
isolation checks. See the [numerical reference report](GATE1C_V21_FP64_REFERENCE_REPORT.md).
This changes neither the original failed formal attempt nor the scientific
Gate1B/overall Gate1 failure. No formal retry or method training is running.

Latest follow-up completed 2026-08-30T17:26:30.873403 UTC: the separately
preregistered v2.2 same-draw FP64 **three-pair numerical pilot passed all four
phases**, with exact code `7fdd4312278eb64dbfb471107bb47e6b897c6859` and numeric
preregistration `6357317749b0ff904e3acd39023b86430d6263ee`. Clean-code synthetic
tests:133/133. Observed real forwards:75; global objective comparisons:288;
class-component rows:630; model guards:12. See the
[precision pilot report](GATE1C_V22_PRECISION_PILOT_REPORT.md) and its sealed
runtime/independent audit receipts. All pilot workers have exited. The private
local archive independently passed at 2026-08-30T17:53:11.791970 UTC: all 66
files / 122,825,857 bytes match the sealed remote evidence. See the
[local archive audit](gate1c_precision_pilot_results/6357317/LOCAL_ARCHIVE_AUDIT.json).
The separate post-run operator-refusal repair `d6bd0707d7a889d7e42e00e61b3d242354335753`
passed 135/135 tests in a clean exact-code checkout; it did not rerun or relabel
the completed pilot. Never repeat an occupied pilot worker. The next
full diagnostic requires its own prospective execution/retention plan and
complete hash-verified evidence. No scientific C verdict or method training
follows automatically from a three-pair engineering pass.

Post-reference storage discovery (read-only, 2026-08-31 Asia/Shanghai): the
compute node has a separate mounted ext4 filesystem at `/tmp`, with
209,945,223,168 bytes available. `/root/LCRSeg` has 9,767,464,960 bytes available.
The local project filesystem has 58,948,328 KiB available. These are measured
capacities, **not** verified platform lifecycle/durability guarantees; the large
host overlay and `/dev/shm` must not be treated as additional persistent space.
No files were moved/deleted and no mount or permission settings were changed.

Historical next-iteration plan after the reference, now implemented by the pilot
above: independently preregister the diagnostic-precision amendment
and a small stage-covering integration pilot before further real forwards.
Keep native probabilities/features/PAS/targets/class strata and random draws,
use an isolated same-draw float64 student gradient receiver, and preserve the
original tolerances and C conditions. Publish code/tests first; no formal
retry or method training is authorized by the completed reference alone.
For any later full run, explicitly plan private scratch directories and
hash-verified retained artifacts (not scratch as the sole evidence copy),
recheck disk headroom and preserve every old failed attempt. The existing
separate scratch mount may avoid unnecessary duplication in `/root`, but its
use and artifact-retention policy must be in that prospective execution plan.

Observed **2026-08-30T15:50:03.822113+00:00**: **RUNNING**, validation-cache
phase. This is a dated operational snapshot, not a mechanism verdict or a claim
that the method has reproduced successfully. Check live completion/failure files.

## Verified readiness

- Input preregistration: `9d8ecc65730bee5bec46a1f098c9fe96a67a59b9`.
- Exact diagnostic code: `44a25254697fa535d2b48b64e27ecb226436f7d0`.
- Published-code full suite: **117 passed, 0 failed, 0 skipped**; real integration
  PASS, completed 2026-08-30T15:46:43.647925+00:00.
- All nine legacy payloads passed CPU readiness before diagnostic forwards;
  exactly one uses the preregistered reconstructed legacy bank.
- Seven real model immutability guards passed; original and affected-case PAS
  pixelwise parity, both fixed gradient pairs and both-GPU deterministic-CE
  checks passed. All nine original B0 checkpoint hashes stayed unchanged.
- No diagnostic optimizer updates, backward calls or parameter.grad writes.
  Prior independent baseline recovery updates remain separately recorded:400.

[Test receipt](gate1c_v21_test_results/44a2525/GATE1C_V2_UNIT_INTEGRATION_TEST_REPORT.json),
[real integration](gate1c_v21_test_results/44a2525/GATE1C_V2_REAL_INTEGRATION.json),
[JUnit](gate1c_v21_test_results/44a2525/pytest.xml), and
[passing test log](gate1c_v21_test_results/44a2525/pytest_output.txt).
Test receipt SHA256:
`53452e7bc0f1b239bf851d87fd66afe104bdde2a5c0abaf38c018bebeef1eac8`.

## Historical run snapshot (now terminal)

- Execution checkout: `/root/SSL_CL_gate1c_v21_44a2525` (clean, detached exact code).
- Run root: `/root/LCRSeg/runs/di_dmpa_gate1c_v21/9d8ecc65730bee5bec46a1f098c9fe96a67a59b9/gate1c_v21_44a25254697fa535d2b48b64e27ecb226436f7d0_attempt1`.
- Started: 2026-08-30T15:47:57.730035+00:00.
- Observed wrapper PID88926; CUDA worker PIDs88988/88989, physical GPUs0/1.
  Recheck full process commands before using these historical IDs.
- Process receipt: `/root/gate1c_v21_formal_44a2525_process.json`.
- Parent log: `/root/gate1c_v21_formal_44a2525.log`.
- Exit receipt after termination: `/root/gate1c_v21_formal_44a2525_exit.json`.
- Input audit: PASS (9 legacy payloads, 1 explicit reconstruction).
- Completed units at observation: seed0 stages0/1/2. Other units were still
  running; no failure files, final status or completion receipt existed.

Two independent GPU workers are used during GPU phases. Per-case NumPy/SciPy
scoring and compressed-cache writes run on CPU between GPU forwards; a point
sample showed both CUDA processes resident but 0% instantaneous GPU utilization.
This is **not** a claim of sustained maximum utilization. The frozen CPU metric
barrier also legitimately leaves GPUs idle. Do not change batches, precision,
sampling or phase order merely to fill the GPUs, and do not start duplicate work.

Exact executed commands on the compute node:

```sh
/root/.venvs/lcrseg-py310/bin/python /root/gate1c_v21_operator.py validate --code-commit 44a25254697fa535d2b48b64e27ecb226436f7d0
/root/.venvs/lcrseg-py310/bin/python /root/gate1c_v21_operator.py launch --code-commit 44a25254697fa535d2b48b64e27ecb226436f7d0
```

Operator SHA256:
`e5fe79c5e78ce1afa8a863743266cf00bedff6197801fae7f572ca10d26bd631`.
Environment and full pytest arguments are in the test receipt. The operator
launches `di_dmpa_gate1c_v2.runner run --input-contract v2.1` with the exact
code/tests/data/output roots above. **Do not repeat these occupied attempts.**

## Historical continuation rules (v2.1 attempt now terminal)

At this snapshot the same-thread 15-minute follow-up `ssl-cl-seg` pointed to
this run. The native long-running goal remains active for the **current method
only**; its latest completed work is reported at the top of this document.
The original completion requirements were to verify all 9/495 validation records,
72 gradient pairs, 576 teacher
draw records, phase barriers, model/bank/data hashes, complete reports and exit
status. Then analyze all results, record a finite evidence-based next plan and
continue within the user's authorization. Do not call test success method success.

Internal V2 filenames are retained for shared-engine compatibility; every runtime
metadata object identifies **v2.1**, and final V21 status/report aliases are
required. Original v2 remains incomplete; its 265 partial caches are not reused.
The missing historical bank still has no historical hash. Gate1B and original
overall Gate1 remain `FAIL_TRANSPORT_NOT_SUPPORTED`, regardless of this outcome.
No DI-DMPA method registration, new method optimizer training, full sweep or main
merge has occurred. Reports advance only the continuation branch; the running
checkout is never updated to a later report commit.

## Read-only follow-up: 2026-08-31 00:02 Asia/Shanghai

Observed **2026-08-30T16:02:14.830278+00:00**: runner PID88927 and its two CPU
metric workers PID89216/PID89219 were live. Validation caches are complete:
**9 units / 495 cases / 72,990,720 pixel rows**. Six of nine reliability metric
unit files existed; no phase-completion, failure, final-status or execution-
completion file existed. This remains a verified wait, not an admission result.

A metadata-only independent check verified the exact v2.1 preregistration/code,
all three hashes bound by the validation barrier, all nine validation-unit JSON
hashes, unique case IDs within each unit, active+null row accounting, and nine
passing model-immutability guards. It did not load arrays or labels, write into
the formal run, change a worker, or inspect partial scientific scores. The
validation barrier SHA256 is
`633df387501e2a4dd8ab033f91db2011cb7bf5384eb8e8173bf679724974e51a`.
The barrier attests the runner's completed cache/model checks; this follow-up
does not claim to have independently rehashed every raw array or checkpoint.

Resource check: available disk was **10,698,432,512 bytes** (about 9.96 GiB).
The remaining stored arrays in the unchanged execution code have raw payload
size `72 * 2 * 384 * 384 * 4 * (22 + 8*3) = 3,906,994,176 bytes` (3.64 GiB):
22 float32 channels per draw0 pair, plus eight three-channel teacher probability
draws. Gradient vectors are hashed, not persisted as full arrays. PoE validation
reuses the existing caches. JSON/CSV/container overhead is additional, so this
is a payload estimate, not a hard total-size guarantee. No cleanup, new storage,
batch/precision change or extra experiment is needed at this checkpoint.
The container's cgroup-v1 memory limit is 85,899,345,920 bytes (80 GiB), not the
much larger host-wide RAM displayed by `free`.

The next complete-evidence review must check 9 reliability units, four phases
of 72 original pairs, 576 shared teacher-draw records, 72 posterior-mean
controls, 9 PoE validation units and **297 model guards** (`9 + 4*72`). Expected
gradient table coverage from the frozen Cartesian products is 6,912 global,
41,472 blockwise and 15,120 class-contribution rows; these are completeness
requirements, not values to fabricate or a replacement for checking pair/draw
identity, hashes, numeric validity and the original C1-C8 conditions. The
formal process and its output directory remain untouched.
