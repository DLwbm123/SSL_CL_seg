# PRES-DSR-SF V0.2.1 exact commands

The private launch receipt preserves literal NAS paths. This public normalization replaces only the private protocol and source roots with `$PROTOCOL_ROOT` and `$SOURCE_ROOT`.

## Tests

```sh
python -B -m pytest -q -p no:cacheprovider \
  --junitxml "$PROTOCOL_ROOT/test_evidence/PRES_DSR_SF_V0_2_1_TESTS_58ee45b.xml" \
  experiments/lcrseg/tests/pres_dsr_sf_v0_2 \
  experiments/lcrseg/tests/pres_jascl_v0_1 \
  experiments/lcrseg/tests/gate0/test_official_model_contract.py \
  experiments/lcrseg/tests/gate0/test_pres_readonly_checkpoint_contract.py
```

## Durable validation child

```sh
bash scripts/with_nas_storage.sh python -B -m pres_dsr_sf_v0_2.run \
  --output "$PROTOCOL_ROOT/formal_01" \
  --code-commit 58ee45b12aae662c8fe61595dc4068094c783f7c \
  --test-report "$PROTOCOL_ROOT/test_evidence/PRES_DSR_SF_V0_2_1_TEST_REPORT_58ee45b.json"
```

The durable launch fixed one visible GPU, deterministic backend settings, `PYTHONHASHSEED=0`, and single-threaded CPU math. Both commands ran through the NAS storage wrapper. No package installation, optimizer, backward, training, or test-evaluation command ran.
