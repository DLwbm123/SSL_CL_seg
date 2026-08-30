# Gate 1A exact commands

以下记录已经执行的操作，**不是新的运行授权**。本轮 hard stop 后禁止
继续机制诊断或重用失败 attempt。所有命令中的代码 SHA 保持区分。

## Publication barriers（本地仓库）

```bash
git add experiments/lcrseg/docs/di_dmpa_jascl/GATE1A_EXECUTION_AUTHORIZATION.md experiments/lcrseg/docs/di_dmpa_jascl/GATE1A_EXECUTION_AUTHORIZATION.json
git commit -m "docs: authorize Gate 1A-only execution after frozen preregistration"
git push origin codex/di-dmpa-gate1-diagnostics
git ls-remote origin refs/heads/codex/di-dmpa-gate1-diagnostics
# Verified authorization: 25ec97c988af290a4fb7a637c4b7cdfe462deb87

git add experiments/lcrseg/di_dmpa_gate1 experiments/lcrseg/tests/di_dmpa_gate1/test_gate1a_core.py experiments/lcrseg/tests/di_dmpa_gate1/test_gate1a_real_integration.py
git commit -m "feat: implement immutable four-panel Gate 1A geometry diagnostics"
git push origin codex/di-dmpa-gate1-diagnostics
git ls-remote origin refs/heads/codex/di-dmpa-gate1-diagnostics refs/heads/main
# Verified diagnostic code: 8f4a71a5ea8d145183a3007ccd398ab79387478e
```

## Remote synchronization

```bash
git -C /root/SSL_CL_gate1 fetch https://github.com/DLwbm123/SSL_CL_seg.git codex/di-dmpa-gate1-diagnostics
git -C /root/SSL_CL_gate1 merge --ff-only FETCH_HEAD
git -C /root/SSL_CL_gate1 rev-parse HEAD
git -C /root/SSL_CL_gate1 status --short
```

## Exact-code unit + real-checkpoint integration

Working directory: `/root/SSL_CL_gate1/experiments/lcrseg`。
同一进程运行46个 synthetic/unit tests 和1个明确授权的只读集成测试。

```bash
cd /root/SSL_CL_gate1/experiments/lcrseg
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  GATE1A_CODE_COMMIT=8f4a71a5ea8d145183a3007ccd398ab79387478e \
  GATE1A_REAL_CHECKPOINT_REPORT=/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_tests_8f4a71a_attempt1/REAL_CHECKPOINT_EXTRACTION_INTEGRATION.json \
  /root/.venvs/lcrseg-py310/bin/python -m pytest -q \
  tests/di_dmpa_gate1/test_gate1a_core.py \
  tests/di_dmpa_gate1/test_gate1a_real_integration.py \
  --junitxml=/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_tests_8f4a71a_attempt1/pytest.xml
```

stdout/stderr 保存到同目录 `pytest_output.txt`。随后只解析 JUnit，核对
47 cases、0 failures/errors/skips，检查 integration PASS 与不变性，
记录 transcript/JUnit/integration/source-file SHA，生成
`GATE1A_UNIT_INTEGRATION_TEST_REPORT.json`，没有模型执行或指标选择。

## 唯一正式 attempt

```bash
cd /root/SSL_CL_gate1/experiments/lcrseg
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=. \
  /root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1.gate1a_runner run \
  --code-commit 8f4a71a5ea8d145183a3007ccd398ab79387478e \
  --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1 \
  --tests /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_tests_8f4a71a_attempt1 \
  --gpus 0,1 --workers 16 --planning-workers 8
```

启动器以 `subprocess.Popen(start_new_session=True)` 运行 bash wrapper，
stdout/stderr、命令 argv、PID 和退出码分别保存在同级的
`gate1a_formal_8f4a71a_attempt1.launch.txt/.launch.json/.exit`。
这些文件复制在 `gate1a_results/publication_evidence/`；实际退出码为2。

源代码自动产生的两个 feature 子命令（各自继承上述环境，另设
`CUDA_VISIBLE_DEVICES=0` 或 `1`）：

```text
/root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1.gate1a_runner extract --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1 --data-root /root/LCRSeg --code-commit 8f4a71a5ea8d145183a3007ccd398ab79387478e --shard 0 --shards 2
/root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1.gate1a_runner extract --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1 --data-root /root/LCRSeg --code-commit 8f4a71a5ea8d145183a3007ccd398ab79387478e --shard 1 --shards 2
```

发生数值 hard stop 后未执行 geometry workers 或任何下游命令。

## Post-run read-only verification and archive

仅对原始 artifact manifest 的92个文件、18个输入 checkpoint、
冻结 config/protocol/manifest/split 做字节级 SHA 核对；读取已保存
feature manifests、immutability audits 和 traceback；没有再次运行模型。
生成独立的 post-run integrity 记录，不编辑原始 attempt。

非 tensor 产物通过只读 tar/scp 归档，保留原始字节。48个 `.npy` 不入 Git；
manifest 中仍保留其路径、shape/dtype 和 SHA。
最终报告提交与执行提交区分，Git push 后再以 `git ls-remote` 核对，
不合并 main。
