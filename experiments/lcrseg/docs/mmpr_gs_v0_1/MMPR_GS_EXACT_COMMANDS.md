# MMPR-GS exact execution commands

Audit record only. These completed create-only phases must not be rerun. No new attempt is authorized. Commands ran on zmic44 (uid1006), under the existing Python environment and pinned detached source bda0af8e25db492785ff09315b2722042e0174e0.

The launcher inherited the environment from `bash /home/jiangsuiyang/SSL_CL/with_nas_storage.sh`: NAS-only cache/TMP paths, `CUDA_VISIBLE_DEVICES=` for controllers, `PYTHONPATH` pointing at the exact NAS checkout, all OMP/MKL/OpenBLAS/NumExpr thread limits 1, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu`. GPU workers received only the registered physical device ID. The durable helper recorded real child and worker exits independently of SSH.

## exact synthetic tests

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.testing --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/receipts/tests_bda0af8_01 --code-commit bda0af8e25db492785ff09315b2722042e0174e0
```

Receipt: `EXACT_TEST_LAUNCH_REQUEST.json`.

## input_audit

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase input_audit --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/input_audit --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json
```

Receipt: `receipts/input_audit/LAUNCH_REQUEST.json`.

## validation

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase validation --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/validation --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json
```

Receipt: `receipts/validation/LAUNCH_REQUEST.json`.

## integration

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/integration --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json
```

Receipt: `receipts/integration/LAUNCH_REQUEST.json`.

## formal

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/formal --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json
```

Receipt: `receipts/formal/LAUNCH_REQUEST.json`.

## final_audit

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase final_audit --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/final_audit --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json
```

Receipt: `receipts/final_audit/LAUNCH_REQUEST.json`.

## private archive

Working directory: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/code/SSL_CL_seg/experiments/lcrseg`.

```sh
/home/jiangsuiyang/anaconda3/envs/py38/bin/python -B /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/archive_new_evidence.py
```

Receipt: `receipts/archive/LAUNCH_REQUEST.json`.

## integration worker commands

GPU 4, PID 2468158, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/integration --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 4
```

GPU 5, PID 2468159, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=5 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/integration --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 5
```

GPU 6, PID 2468160, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=6 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase integration --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/integration --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 6
```

## formal worker commands

GPU 4, PID 2468672, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/formal --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 4
```

GPU 5, PID 2468673, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=5 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/formal --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 5
```

GPU 6, PID 2468674, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=6 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/formal --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 6
```

GPU 7, PID 2468675, actual exit 0:

```sh
CUDA_VISIBLE_DEVICES=7 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -B -m mmpr_gs_v0_1.run --phase formal --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/runs/formal --gate /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/control/PUBLICATION_GATE.json --code-commit bda0af8e25db492785ff09315b2722042e0174e0 --worker --shard 7
```

## Test collection

`mmpr_gs_v0_1.testing` called pytest with these exact arguments under the no-update guard:

```text
-q -p no:cacheprovider tests/di_dmpa_gate1c_v2 tests/di_dmpa_gate1c_v3 tests/mmpr_gs_v0_1 --ignore=tests/di_dmpa_gate1c_v2/test_real.py --ignore=tests/di_dmpa_gate1c_v3/test_baseline.py --junitxml /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/receipts/tests_bda0af8_01/pytest.xml --basetemp /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/mmpr_gs_v0_1_feasibility_20260831/receipts/tests_bda0af8_01/scratch
```

No environment was created and no Torch/package installation occurred. The complete launcher request/response scripts and publication gate are preserved in the NAS evidence bundle; they contain no credentials.
