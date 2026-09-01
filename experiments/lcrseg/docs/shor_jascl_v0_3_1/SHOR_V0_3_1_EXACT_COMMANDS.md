# SHOR-JASCL V0.3.1 exact commands

Public path variables replace private server paths. The durable private receipts preserve resolved argv and working directories.

```bash
CODE_COMMIT=b7f85c7deaccbb8dbeef1ef998ac0823db55e75d
SOURCE_REPO=<exact NAS source checkout>
PROTOCOL_ROOT=<create-only SHOR V0.3.1 protocol root>
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
  --junitxml="$PROTOCOL_ROOT/test_evidence/SHOR_V0_3_1_TESTS_b7f85c7.xml"
```

Scoped preflight:

```bash
cd "$SOURCE_REPO/experiments/lcrseg"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=
bash scripts/with_nas_storage.sh "$PYTHON" -B - <<'PY'
# Read only seed 0 train-memory domains 0 and 1.
# bootstrap_weights(seed=0, stage=1, replicate=0)
# -> reconstruct_oof(..., multiplicity=multiplicity)
# -> select_threshold(..., stage=1, domain=0, multiplicity=multiplicity)
# Assert positive inactive count, finite active OOF, inactive NaN sentinels,
# zero validation/GT/model-forward counts, then create SHOR_V0_3_1_PREFLIGHT.json.
PY
```

Single durable formal launch:

```bash
cd "$SOURCE_REPO/experiments/lcrseg"
bash scripts/with_nas_storage.sh "$PYTHON" -B -m di_dmpa_gate1c_v3.durable launch \
  --output "$PROTOCOL_ROOT/formal_01" --phase shor_jascl_v0_3 \
  --cwd "$SOURCE_REPO/experiments/lcrseg" \
  --env CUDA_VISIBLE_DEVICES= --env PYTHONHASHSEED=0 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  -- bash scripts/with_nas_storage.sh "$PYTHON" -B -m shor_jascl_v0_3.run \
  --output "$PROTOCOL_ROOT/formal_01" --code-commit "$CODE_COMMIT" \
  --test-report "$PROTOCOL_ROOT/SHOR_V0_3_1_TEST_REPORT.json" \
  --private-root "$PRIVATE_BUNDLE"
```

Postflight archive audit:

```bash
cd "$SOURCE_REPO/experiments/lcrseg"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=
bash scripts/with_nas_storage.sh "$PYTHON" -B -m shor_jascl_v0_3.postflight \
  --output "$PROTOCOL_ROOT/formal_01" --code-commit "$CODE_COMMIT" \
  --private-root "$PRIVATE_BUNDLE"
```
