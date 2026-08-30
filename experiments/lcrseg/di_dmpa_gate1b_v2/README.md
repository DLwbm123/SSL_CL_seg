# Gate 1B v2: offline null-aware transport only

The immutable contract is `docs/di_dmpa_jascl/DI_DMPA_GATE1B_V2_PREREGISTRATION.json`.
This package adds no training method, configuration or dependency. It reuses the frozen UNet loader, image reader,
null-aware features/spherical K2 oracle solver, SHA helpers and model-state guard.

`python -m pytest tests/di_dmpa_gate1b_v2/test_core.py` uses only synthetic tensors and frozen JSON inputs.
After exact code push/remote verification, the opt-in real integration requires `GATE1B_V2_CODE_COMMIT` and
`GATE1B_V2_INTEGRATION_OUTPUT` (a new directory), and runs `tests/di_dmpa_gate1b_v2/test_integration.py`.
It performs no transform fit. The exact-code combined test receipt, JUnit XML, stdout and integration JSON
must be archived before a formal attempt.

The create-only formal command is:

```sh
python -m di_dmpa_gate1b_v2.runner run --code-commit EXACT_CODE_SHA --output REGISTERED_ATTEMPT1_PATH --tests EXACT_CODE_TEST_DIRECTORY
```

Use the preregistered existing Python environment, both GPUs0/1 for extraction, and
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Six CPU float64 workers fit the six independent fixed1000-step T2 maps after the complete12-unit paired barrier.
Historical-val evaluator processes launch only after6000 updates and frozen model hashes. No per-seed result preview
changes execution. Undefined conditional statistics retain explicit support flags; full-support errors never discard nulls.

Runtime artifacts are remote-only; reports copy descriptors and hashes without raw feature arrays. Every output is
create-only. Scientific failure is valid; no extra steps, automatic retries, Gate1C, model optimization or main merge.
The report's Git identity is the first commit adding its exact bytes, recorded in a separate publication receipt.
