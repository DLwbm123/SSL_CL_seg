# PMGC exact execution commands

Completed audit record, not a rerun instruction. All phases are create-only and the sole real attempt is consumed. Source commit: `c3e044f7359c89e157561cbdfe6c9cab0ac46bb5`.

Server: zmic44, uid1006. The parent used `bash /home/jiangsuiyang/SSL_CL/with_nas_storage.sh` with NAS cache/TMP paths, CUDA_VISIBLE_DEVICES empty for CPU controllers, PYTHONPATH at the exact NAS checkout, OMP/MKL/OpenBLAS/NumExpr thread limits 1, CUBLAS_WORKSPACE_CONFIG=:4096:8 and LD_LIBRARY_PATH=/lib/x86_64-linux-gnu. Worker CUDA masks are the registered physical GPU. No environment was installed.

## archive

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/archive_completed_evidence.py
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/archiving/LAUNCH_REQUEST.json`; SHA256 `d73d3ece83eed5b7eee83757e436a602efa731004aaaaa75cc5d8864a0eb7db0`.

## audit_formal

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.audit --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/audit_formal --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/audit_formal/LAUNCH_REQUEST.json`; SHA256 `e27195f682665972219382dc57587e342d256c556b62090abba90044b84e814f`.

## audit_integration

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.audit --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/audit_integration --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/audit_integration/LAUNCH_REQUEST.json`; SHA256 `3226a6c4991ce64596f8cff641bdab60e25ad5fdf1bb2e48bc2fbbf5cbe10de9`.

## development_tests

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.testing --development --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/dev01 --code-commit 7dcac476d5ca84349d407e2f2f9ca2c8269f872e
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/dev01/LAUNCH_REQUEST.json`; SHA256 `94d3070d8af35bc2aa30d657aeb2229bcd0bd7ad0e168f46a49f05eeea6ca7b9`.

## exact_tests

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.testing --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/exact_c3e044f7359c --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/exact_c3e044f7359c/LAUNCH_REQUEST.json`; SHA256 `94bc888623b3e5bc52aeb18fa799d44a38a7460fabce32c56444f2315afbaec8`.

## formal

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/LAUNCH_REQUEST.json`; SHA256 `3df8ce59edc3e358f674ab0fb44bda0b38c61a0add0424a5bbba66491cc90cf7`.

## input_audit

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase input_audit --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/input_audit --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/input_audit/LAUNCH_REQUEST.json`; SHA256 `4df961beeac4353871b009dcaa8f3592a9fa7cf80d605f6e80a79289b21c366e`.

## integration

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/LAUNCH_REQUEST.json`; SHA256 `216a126a609d2155272e786df8b75145254ac212f000a27fd829bc11b3a7362a`.

## orchestration

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/phase_operator.py
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/orchestration/LAUNCH_REQUEST.json`; SHA256 `dcc37f1a648829962ca3813cb687a98427138089fe8df0d9251d998b623cffb2`.

## preparation

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`. Actual child exit: `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

Source receipt: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/LAUNCH_REQUEST.json`; SHA256 `1930d9efad248945c911b5d47655cec0175d6efbb36b4af66abbc215d1566a56`.

## preparation seed0_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed0_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed0_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## preparation seed0_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed0_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed0_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## preparation seed1_stage1

GPU `6`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed1_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed1_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## preparation seed1_stage2

GPU `7`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed1_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed1_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## preparation seed2_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed2_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed2_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## preparation seed2_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase preparation_unit --unit seed2_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/preparation/units/seed2_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed0_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed0_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed0_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed0_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed0_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed0_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed1_stage1

GPU `6`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed1_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed1_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed1_stage2

GPU `7`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed1_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed1_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed2_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed2_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed2_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## integration seed2_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase integration_unit --unit seed2_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/integration/units/seed2_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed0_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed0_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed0_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed0_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed0_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed0_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed1_stage1

GPU `6`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed1_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed1_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed1_stage2

GPU `7`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed1_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed1_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed2_stage1

GPU `4`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed2_stage1 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed2_stage1 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## formal seed2_stage2

GPU `5`, actual child exit `0`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m pmgc_v0_1.run --phase formal_unit --unit seed2_stage2 --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/formal/units/seed2_stage2 --code-commit c3e044f7359c89e157561cbdfe6c9cab0ac46bb5 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/control/PMGC_PUBLICATION_GATE.json
```

## Exact pytest collection

```text
-q -p no:cacheprovider tests/di_dmpa_gate1c_v2 tests/di_dmpa_gate1c_v3 tests/mmpr_gs_v0_1 tests/pmgc_v0_1 tests/gate0/test_config_protocol.py tests/gate0/test_report_compiler.py tests/gate0/test_official_model_contract.py tests/gate0/test_classifier_stochasticity.py tests/gate0/test_pas_probability.py --ignore=tests/di_dmpa_gate1c_v2/test_real.py --ignore=tests/di_dmpa_gate1c_v3/test_baseline.py --ignore=tests/gate0/test_model_checkpoint.py --ignore=tests/gate0/test_deterministic_supervised_smoke.py --ignore=tests/gate0/test_runner_resume_equivalence.py --ignore=tests/gate0/test_resume_v2.py --ignore=tests/gate0/test_manifest_adapter.py --deselect=tests/gate0/test_classifier_stochasticity.py::test_stochastic_classifier_draws_different_weights --deselect=tests/gate0/test_pas_probability.py::test_zero_valid_pixels_returns_graph_connected_zero --deselect=tests/gate0/test_pas_probability.py::test_teacher_receives_no_gradient --deselect=tests/gate0/test_pas_probability.py::test_prototypes_and_masks_receive_no_gradient --junitxml /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/exact_c3e044f7359c/pytest.xml --basetemp /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/pmgc_v0_1_feasibility_20260831/tests/exact_c3e044f7359c/scratch
```

The measured compiler runs in the same exact-test phase. Its dynamic trace/source hashes are in PMGC_CALL_GRAPH.json; exact test XML/output are retained verbatim. All live operators and source-only transfer evidence are retained privately on NAS.
