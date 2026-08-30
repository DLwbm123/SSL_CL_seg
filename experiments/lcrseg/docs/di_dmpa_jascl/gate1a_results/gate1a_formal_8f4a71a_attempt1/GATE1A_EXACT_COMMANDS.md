# Gate 1A exact execution

```text
/root/SSL_CL_gate1/experiments/lcrseg
/root/.venvs/lcrseg-py310/bin/python /root/SSL_CL_gate1/experiments/lcrseg/di_dmpa_gate1/gate1a_runner.py run --code-commit 8f4a71a5ea8d145183a3007ccd398ab79387478e --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1 --tests /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_tests_8f4a71a_attempt1 --gpus 0,1 --workers 16 --planning-workers 8
```

GPU child shards call the same module with extract, the same output/data/code commit, shard indices 0..N-1 and --shards N. Each has CUDA_VISIBLE_DEVICES set to its listed physical GPU, BLAS/OpenMP threads=1, CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU geometry is float64. Full shard metadata/logs are retained.
