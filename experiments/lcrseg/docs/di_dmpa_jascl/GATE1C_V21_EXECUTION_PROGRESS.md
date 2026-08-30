# Gate1C v2.1 execution progress

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

## Active run

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

## Continuation and interpretation

The same-thread 15-minute follow-up `ssl-cl-seg` is active and points to this
run; the native long-running goal remains active for the **current method only**.
On completion, verify all 9/495 validation records, 72 gradient pairs, 576 teacher
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
