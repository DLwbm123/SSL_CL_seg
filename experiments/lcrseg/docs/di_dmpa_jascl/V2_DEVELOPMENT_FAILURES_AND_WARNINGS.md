# Gate 0 v2 development failures and warnings

Date: 2026-08-30. No failure is treated as evidence of a passing experiment.

1. Initial local unit run: 4 failed, 24 passed, 4 skipped.
   - Two old compiler tests still expected the permissive v1 log schema; replaced
     with positive and adversarial v2 evidence tests.
   - The stochastic positive test used all-zero GAS. The official float32
     inverse-gradient expression cancels to zero in this state. The test now
     uses a nondegenerate GAS state to exercise sampling. Official source unchanged.
   - Manual masked reduction differed by 2.98e-8 due to reduction order.
     The formula comparison uses 1e-7 numerical tolerance; formula unchanged.
   These are development failures, not formal-run failures. Final remote suite:
   52 passed; local complete suite: 264 passed, 4 remote-data skips.
2. The first remote reference clone was attempted before the 92 MB bundle upload
   completed and reported early EOF. No run/data was affected. After transfer,
   local and remote SHA-256 matched, and a fresh clone succeeded at the pinned commit.
3. Preserved runtime warnings: scheduler.step(epoch) before optimizer.step,
   scheduler epoch-argument deprecation, and CUDA NLL deterministic-kernel warning.
   The requested schedule and 1e-6 resume tolerance were not changed.
4. Preflight zero-coverage failures: none. The first fixed real unlabeled batch
   in each of the three domains passed without retries or threshold changes.
5. Final formal-run audit: six complete runs, all exit 0, no training exception
   or non-finite loss/gradient. Across 11,970 PAS batches: zero-coverage count=0,
   zero-unweighted-consistency-gradient count=0. Minimum gradient norm was
   3.758291603148123e-6. All 24 best/final checkpoints passed tensor-finiteness
   and source/config checks. Full stderr and per-domain zero counts are in
   `V2_TRAINING_DIAGNOSTICS.json`; the compiler separately validated all logs.
6. Resource diagnostics: the combined `nvidia-smi --query-gpu ... -c 6` command
   was rejected; monitoring used supported `dmon`/`pmon` commands instead.
   cgroup v2 paths were absent; the active cgroup v1 files showed a 16-core
   CPU quota. Neither diagnostic error affected a training process.
