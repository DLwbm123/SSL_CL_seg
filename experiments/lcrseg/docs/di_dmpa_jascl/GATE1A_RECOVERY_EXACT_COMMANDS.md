# Gate 1A recovery exact commands

这是已经执行的记录，不是继续诊断授权。当前注册零向量hard stop后不得再运行本轮。

## Git publication barriers

```bash
git switch -c codex/gate1a-sampled-norm-recovery 945b484072cb9f2757be98df34e5d72844596e84
git add experiments/lcrseg/docs/di_dmpa_jascl/GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.md experiments/lcrseg/docs/di_dmpa_jascl/GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.json
git commit -m "docs: clarify Gate 1A full-map versus registered norm scope"
git push -u origin codex/gate1a-sampled-norm-recovery
git ls-remote origin refs/heads/codex/gate1a-sampled-norm-recovery
# e8336da9d7364f4b67912d03791195445318afc3

git commit -m "fix: scope Gate 1A norm guard to frozen registered coordinates"
git push origin codex/gate1a-sampled-norm-recovery
git ls-remote origin refs/heads/codex/gate1a-sampled-norm-recovery refs/heads/main
# recovery code a89716ddbd2eccbe76c574e97e520d424aa923ab
# main remains 46e892960240543c946c570a9378d409b226384b
```

## Exact source delivery

```bash
git bundle create /tmp/gate1a-recovery-publication.WDl4CZ/recovery.bundle 8f4a71a5ea8d145183a3007ccd398ab79387478e..HEAD
scp -P 31192 /tmp/gate1a-recovery-publication.WDl4CZ/recovery.bundle root@162.14.139.38:/root/gate1a-recovery-a89716d.bundle
# On cloud host:
git -C /root/SSL_CL_gate1 fetch /root/gate1a-recovery-a89716d.bundle HEAD
git -C /root/SSL_CL_gate1 worktree add --detach /root/SSL_CL_gate1_recovery_code a89716ddbd2eccbe76c574e97e520d424aa923ab
mkdir -p /root/SSL_CL_gate1_recovery_code/experiments/lcrseg/third_party
ln -s /root/SSL_CL_gate0_v2/experiments/lcrseg/third_party/JASCL_REFERENCE /root/SSL_CL_gate1_recovery_code/experiments/lcrseg/third_party/JASCL_REFERENCE
git -C /root/SSL_CL_gate1_recovery_code status --short
git -C /root/SSL_CL_gate1_recovery_code rev-parse HEAD
```

Earlier failed fetch/link attempts are recorded separately; no original checkout source was rewritten.

## Exact-code tests and localization

Actual command and complete environment: [EXACT_TEST_COMMAND.json](gate1a_recovery_results/gate1a_recovery_tests_a89716d_attempt1/EXACT_TEST_COMMAND.json).

```bash
cd /root/SSL_CL_gate1_recovery_code/experiments/lcrseg
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \\
 CUDA_VISIBLE_DEVICES=0 GATE1A_CODE_COMMIT=a89716ddbd2eccbe76c574e97e520d424aa923ab \\
 GATE1A_RECOVERY_LOCALIZATION_OUTPUT=/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_known_failure_a89716d_attempt1 \\
 /root/.venvs/lcrseg-py310/bin/python -m pytest -q \\
 tests/di_dmpa_gate1/test_gate1a_core.py tests/di_dmpa_gate1/test_gate1a_recovery.py \\
 tests/di_dmpa_gate1/test_gate1a_recovery_integration.py \\
 --junitxml=/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_recovery_tests_a89716d_attempt1/GATE1A_RECOVERY_PYTEST.xml
```

The real integration invokes known_failure_localization exactly once, records all100 cases/all427500 registered coordinates, and writes the localization verdict. No separate duplicate localization/model run was performed.

## Unique formal attempt2

```bash
cd /root/SSL_CL_gate1_recovery_code/experiments/lcrseg
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \\
 /root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1.gate1a_runner run \\
 --code-commit a89716ddbd2eccbe76c574e97e520d424aa923ab \\
 --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2 \\
 --tests /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_recovery_tests_a89716d_attempt1 \\
 --localization /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_known_failure_a89716d_attempt1/GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json \\
 --gpus 0,1 --workers 16
```

Child extract shards use the same module/output/data/code with `extract --shard 0 --shards 2` and `extract --shard 1 --shards 2`; each has CUDA_VISIBLE_DEVICES set to its GPU. RUN/shard metadata and original stdout are archived.

Detached launch wrappers and the post-run byte-only verifier are archived under [publication_evidence](gate1a_recovery_results/publication_evidence/). Wrapper PIDs: tests83506, formal83608; both finished. Formal child exits[1,0], main exit2. No geometry command ran.

Post-run verifier rehashed original92 artifacts, new174 artifacts and all18 checkpoints, without tensor/model loading. Tests, localization, launch records and attempt2 non-tensor artifacts were copied byte-exactly. Original sampling-plan bytes are referenced from the immutable attempt1 archive instead of duplicated in Git.
