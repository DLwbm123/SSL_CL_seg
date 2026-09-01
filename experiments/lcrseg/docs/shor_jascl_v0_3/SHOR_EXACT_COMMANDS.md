# SHOR-JASCL V0.3 exact commands

The public record uses path variables to avoid publishing private server locations. The execution receipts retain the resolved argv and working directory.

```bash
CODE_COMMIT=4551d9311ab49927b55730c64085d4990a32fedc
SOURCE_REPO=<exact NAS source checkout>
SHOR_ROOT=<create-only SHOR protocol root>
PRIVATE_BUNDLE=<frozen V0.2.1 private bundle>
PYTHON=/home/jiangsuiyang/anaconda3/envs/py38/bin/python
WRAPPER=/home/jiangsuiyang/SSL_CL/with_nas_storage.sh
```

Admission suite:

```bash
cd "$SOURCE_REPO"
export PYTHONPATH="$SOURCE_REPO/experiments/lcrseg"
bash "$WRAPPER" "$PYTHON" -B -m pytest -q \
  experiments/lcrseg/tests/shor_jascl_v0_3 \
  experiments/lcrseg/tests/pres_dsr_sf_v0_2 \
  experiments/lcrseg/tests/pres_jascl_v0_1 \
  experiments/lcrseg/tests/gate0/test_official_model_contract.py \
  experiments/lcrseg/tests/gate0/test_pres_readonly_checkpoint_contract.py \
  --junitxml="$SHOR_ROOT/test_evidence/SHOR_JASCL_V0_3_TESTS_4551d93.xml"
```

Single durable formal launch:

```bash
cd "$SOURCE_REPO/experiments/lcrseg"
bash scripts/with_nas_storage.sh "$PYTHON" -B -m di_dmpa_gate1c_v3.durable launch \
  --output "$SHOR_ROOT/formal_01" --phase shor_jascl_v0_3 \
  --cwd "$SOURCE_REPO/experiments/lcrseg" \
  --env CUDA_VISIBLE_DEVICES= --env PYTHONHASHSEED=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  -- bash scripts/with_nas_storage.sh "$PYTHON" -B -m shor_jascl_v0_3.run \
  --output "$SHOR_ROOT/formal_01" --code-commit "$CODE_COMMIT" \
  --test-report "$SHOR_ROOT/test_evidence/SHOR_JASCL_V0_3_TEST_REPORT_4551d93.json" \
  --private-root "$PRIVATE_BUNDLE"
```

Post-stop read-only archive verification:

```bash
cd "$SOURCE_REPO/experiments/lcrseg"
export PYTHONPATH=.
bash scripts/with_nas_storage.sh "$PYTHON" -B - <<'PY'
from di_dmpa_gate1c_v3 import durable as d
from shor_jascl_v0_3.protocol import isolation_guard, verify_private_bundle
d.verify("<formal_01>", "PHASE_shor_jascl_v0_3_MANIFEST.json")
with isolation_guard():
    verify_private_bundle("<frozen V0.2.1 private bundle>")
PY
```
