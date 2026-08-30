# Legacy PAS recovery feasibility result

**RECONSTRUCTION_SUPPORTED_NOT_HISTORICAL_HASH_VERIFIED**.
The bounded plan succeeded; the frozen Gate1C v2 attempt remains incomplete.

## Observed evidence

- Plan: `05946f05484ab3bf612daf20a21e4fee541668ef`.
- Exact helper: `493de2bb007e73f2306d610d4d2d020893dfa12d`.
- Unmodified training source: `fb55e8022bc379e2515a46214c6fdf45ea818de6`.
- Controller launched 2026-08-30T15:21:57.418891+00:00; completed
  2026-08-30T15:22:22.180488+00:00. Controller PID88458, workers88459/88460;
  both workers and the comparison exited0. These are historical PIDs, not a
  claim that the jobs are still running.
- Each replica executed exactly 200 original supervised updates, steps3209–3408.
  All 200 identity fields and loss/LR rows matched the archived original trace;
  maximum observed numeric differences were **0.0** on both GPUs.
- The bank and complete captured student/EMA/optimizer/scheduler/GAS/RNG/stage/
  sampler state passed byte-exact comparison. Case-order trace hashes matched.
- Each replica used 416 current-domain labeled case accesses (400 training,
  16 prototype construction) and 960 validation evaluator case accesses.
  No test or hidden GT was read. No unlabeled, method or transport optimizer
  update occurred. Baseline recovery updates total **400**, not zero.
- All nine original B0 checkpoint hashes and the original training trace hash
  remained unchanged; both source checkouts were clean.

## Artifact identities

Remote root:
`/root/LCRSeg/runs/gate1c_legacy_pas_recovery/05946f05484ab3bf612daf20a21e4fee541668ef/attempt1`.
Tensor artifacts stay remote; only metadata and passing test evidence are public.

| Artifact | SHA256 |
| --- | --- |
| Each `legacy_pas_candidate.pt` | `86cfd1b7ab1b337f9a43f51d7a2ba2e7956f11bb3ae67cb59fe164f9a47de1eb` |
| Each complete `last.pt` capture | `3ee00939894c081533187ce764b77dab3e6ab9597c6980280c3f6e0dd13fe3d9` |
| Each case-order trace | `2ece5789e185dc7bd90a4f2e6d7b849ddec888c2e833d98762ef99dbcbcaeccf` |
| `RECOVERY_COMPARISON.json` | `ca8cf2b2d575402dc52b720ba83bbc3c683783a9aac6202109d6024cbc833db6` |

[Full comparison](gate1c_legacy_pas_recovery_results/05946f0/RECOVERY_COMPARISON.json),
[worker commands](gate1c_legacy_pas_recovery_results/05946f0/WORKERS.json),
[process receipt](gate1c_legacy_pas_recovery_results/05946f0/PROCESS.json),
[exit receipt](gate1c_legacy_pas_recovery_results/05946f0/EXIT.json), and
[25-test original-source JUnit](gate1c_legacy_pas_recovery_results/05946f0/gate1c_pas_recovery_source_tests_attempt1.xml).

Both commands used the existing Python with `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`LD_LIBRARY_PATH=/lib/x86_64-linux-gnu`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
and respectively `CUDA_VISIBLE_DEVICES=0`/`1`. Exact arguments are preserved in
`WORKERS.json`; the original runner retained its `warn_only` policy. The detached
launcher was `/root/.venvs/lcrseg-py310/bin/python
/root/launch_legacy_pas_recovery_493de2b.py --launch`, SHA256
`ef5a34053ce787962d27c7bd0dd6d427708f2c9ed0fe727093af4906d2b00c95`.
**Do not relaunch**: the one allowed attempt is occupied and completed.

## Interpretation and next plan

This supports reconstruction from a full frozen resume state; a missing old
bank has no historical hash against which to certify file identity. The original
trace has no per-batch image/augmentation fingerprint. These limitations remain
despite zero observed loss differences and identical independent replicas.

Before consuming a candidate, register a separate Gate1C **v2.1 input amendment**:
only the missing `B0/seed1/stage1` legacy bank can use this reconstructed tensor.
Keep all nine original student/EMA checkpoints, the other eight legacy banks,
K2 geometry, identity history, cases, pixels, seeds, formulas and thresholds.
No use of original partial v2 caches. Use the first replica in the already fixed
replica order as canonical; both artifact hashes are identical, so this is not
performance-based selection. Check all nine required payloads before forwards,
then rerun complete diagnostics in a new root after code/tests/publication gates.
Original Gate1B / overall Gate1 remains `FAIL_TRANSPORT_NOT_SUPPORTED`.
