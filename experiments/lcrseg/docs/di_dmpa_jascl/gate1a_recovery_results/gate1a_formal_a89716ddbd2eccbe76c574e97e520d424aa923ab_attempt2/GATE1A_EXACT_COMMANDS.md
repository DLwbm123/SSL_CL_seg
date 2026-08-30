# Gate 1A exact execution

```text
/root/SSL_CL_gate1_recovery_code/experiments/lcrseg
/root/.venvs/lcrseg-py310/bin/python /root/SSL_CL_gate1_recovery_code/experiments/lcrseg/di_dmpa_gate1/gate1a_runner.py run --code-commit a89716ddbd2eccbe76c574e97e520d424aa923ab --output /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_a89716ddbd2eccbe76c574e97e520d424aa923ab_attempt2 --tests /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_recovery_tests_a89716d_attempt1 --localization /root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_known_failure_a89716d_attempt1/GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json --gpus 0,1 --workers 16
```

GPU child shards call the same module with extract, the same output/data/code commit, shard indices 0..N-1 and --shards N. Each has CUDA_VISIBLE_DEVICES set to its listed physical GPU, BLAS/OpenMP threads=1, CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU geometry is float64. Full shard metadata/logs are retained.
