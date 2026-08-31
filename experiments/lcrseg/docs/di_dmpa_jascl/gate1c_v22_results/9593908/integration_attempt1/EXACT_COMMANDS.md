# Exact execution

```sh
cd /root/LCRSeg/gate1c_v22_source/worktree_1cfd8235293e157afd6b40f0f091ce6bc6df9f9f/experiments/lcrseg
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/lib/x86_64-linux-gnu CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/root/LCRSeg/gate1c_v22_source/worktree_1cfd8235293e157afd6b40f0f091ce6bc6df9f9f/experiments/lcrseg /root/.venvs/lcrseg-py310/bin/python -m di_dmpa_gate1c_v2.runner run --code-commit 1cfd8235293e157afd6b40f0f091ce6bc6df9f9f --output /root/LCRSeg/runs/di_dmpa_gate1c_v22/9593908bd36f7f833e385a70b2b772b7a8c84d22/integration_attempt1 --tests /root/LCRSeg/gate1c_v22_tests/1cfd8235293e157afd6b40f0f091ce6bc6df9f9f_attempt1/pytest.xml --data-root /root/LCRSeg --input-contract v2.1 --execution-version v2.2 --scope integration
```
