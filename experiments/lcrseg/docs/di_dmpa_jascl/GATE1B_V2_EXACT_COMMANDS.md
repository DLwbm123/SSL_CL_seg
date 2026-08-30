# Gate 1B v2 exact execution record

以下为已经执行并封存的命令，不是继续执行的授权。所有输出create-only；本轮不得重跑或覆盖attempt1。
诊断代码、预注册、授权和模型/数据输入的完整SHA见报告及各run metadata。SSH endpoint/认证不属于科研复现参数，未在公开记录中披露。

## Publication barriers

在从`9b2ffd04c7a8e9da73f08edb0760be3f269065d8`创建的
`codex/gate1b-v2-null-aware-transport`分支，依次独立提交、push并以`git ls-remote`核验：

1. Freeze：`58f19e968700bd7708ec00e44a11759b48ce756f`。
2. Preregistration：`b20f186deff287843f3c9f18bf4ab5633908f441`。
3. Authorization：`c6f72b86fdfa3683a6e2c7dbf593f73cab74c592`。
4. Exact code：`f2a3ed7476323119b1a4fa22481b44038bc4148c`。

```sh
git push origin HEAD:refs/heads/codex/gate1b-v2-null-aware-transport
git ls-remote origin refs/heads/codex/gate1b-v2-null-aware-transport refs/heads/main
```

Exact-code部署使用增量Git bundle，创建新的detached worktree；未更改既有Gate1A工作树：

```sh
git -C /root/SSL_CL_gate1 fetch /root/gate1b_v2_exact_f2a3ed7.bundle HEAD
git -C /root/SSL_CL_gate1 worktree add --detach /root/SSL_CL_gate1b_v2 f2a3ed7476323119b1a4fa22481b44038bc4148c
mkdir -p /root/SSL_CL_gate1b_v2/experiments/lcrseg/third_party
ln -s /root/SSL_CL_gate0_v2/experiments/lcrseg/third_party/JASCL_REFERENCE /root/SSL_CL_gate1b_v2/experiments/lcrseg/third_party/JASCL_REFERENCE
```

## Synthetic pre-publication checks

```sh
cd /root/SSL_CL_gate1b_v2_dev/experiments/lcrseg
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=. \
  /root/.venvs/lcrseg-py310/bin/python -m pytest -q \
  tests/di_dmpa_gate1b_v2/test_core.py tests/di_dmpa_gate1b_v2/test_pipeline_synthetic.py \
  --junitxml=/root/gate1b_v2_synthetic_v4.xml
```

结果：76 passed，23.03s。此前三轮JUnit及首次mock失败全部保留在postrun目录；没有真实forward/fit。

## Exact-code tests and real read-only integration

Driver：[run_exact_tests.py](gate1b_v2_results/postrun_f2a3ed7_attempt1/run_exact_tests.py)。
实际入口：`/root/.venvs/lcrseg-py310/bin/python /root/gate1b_v2_run_exact_tests_f2a3ed7.py`。
该driver只执行测试并生成receipt，不启动正式diagnostic。

实际pytest环境与命令：

```sh
cd /root/SSL_CL_gate1b_v2/experiments/lcrseg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu PYTHONPATH=/root/SSL_CL_gate1b_v2/experiments/lcrseg
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0,1
export GATE1B_V2_CODE_COMMIT=f2a3ed7476323119b1a4fa22481b44038bc4148c
export GATE1B_V2_INTEGRATION_OUTPUT=/root/LCRSeg/runs/di_dmpa_gate1b_v2_validation/f2a3ed7476323119b1a4fa22481b44038bc4148c/attempt1/integration
/root/.venvs/lcrseg-py310/bin/python -m pytest -q tests/di_dmpa_gate1b_v2 \
  --junitxml=/root/LCRSeg/runs/di_dmpa_gate1b_v2_validation/f2a3ed7476323119b1a4fa22481b44038bc4148c/attempt1/pytest.xml
```

结果：77 passed，26.09s，exit0；real integration PASS，两case各2048行，model/transport真实更新均0。

## Sole formal diagnostic

Driver：[run_formal.py](gate1b_v2_results/postrun_f2a3ed7_attempt1/run_formal.py)。
在检查77/77 exact-code测试receipt及全部源码SHA后，单独启动：
`/root/.venvs/lcrseg-py310/bin/python /root/gate1b_v2_run_formal_f2a3ed7.py`。

与上一节相同的OMP/MKL/OPENBLAS、LD_LIBRARY_PATH、PYTHONPATH、CUBLAS、CUDA_VISIBLE_DEVICES环境：

```sh
cd /root/SSL_CL_gate1b_v2/experiments/lcrseg
/root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1b_v2.runner run \
  --code-commit f2a3ed7476323119b1a4fa22481b44038bc4148c \
  --output /root/LCRSeg/runs/di_dmpa_gate1b_v2/b20f186deff287843f3c9f18bf4ab5633908f441/gate1b_v2_f2a3ed7476323119b1a4fa22481b44038bc4148c_attempt1 \
  --tests /root/LCRSeg/runs/di_dmpa_gate1b_v2_validation/f2a3ed7476323119b1a4fa22481b44038bc4148c/attempt1
```

Runner按冻结顺序执行：16个coordinate workers → GPU0/1 paired extraction →全部12-unit census及哈希/immutability屏障→
6个CPU float64 workers各1000步→GPU0/1 evaluator→全部9-unit完成→一次统一B1–B7判定。
`--data-root`使用冻结默认`/root/LCRSeg`；没有改LR、iterations、threshold、sampling或模型选择。

进程exit0；2026-08-30T13:13:47.613315+00:00开始，13:17:38.197869+00:00结束。
结果`FAIL_TRANSPORT_NOT_SUPPORTED`。exit0表示诊断完整完成，不表示科学准入通过。

## Postrun audit and immutable public subset

Auditor：[audit_and_export.py](gate1b_v2_results/postrun_f2a3ed7_attempt1/audit_and_export.py)。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu \
  /root/.venvs/lcrseg-py310/bin/python /root/gate1b_v2_audit_and_export_f2a3ed7.py
```

只读校验所有258个正式文件、checkpoint、缓存和既有metrics；无新forward、fit或oracle fit。
生成94个正式公开文件及2个postrun JSON的归档：
`/root/gate1b_v2_report_f2a3ed7.tar.gz`，3,595,832 bytes，SHA-256
`73630b698a4bcc009f8bcd9465c74a117c15be6339bca47dca2c3c015d8ca6ab`。
下载后再次核验归档及94个文件SHA；165个npy和既有geometry plan留远端，新的transport plan公开。
原始CSV的CRLF行尾和失败JUnit traceback的空白均按字节保留，未为了消除Git whitespace提示而改写冻结产物。

报告及publication receipt只向指定分支追加；没有main merge。之后停止：`STOP_FOR_INDEPENDENT_REVIEW`。
