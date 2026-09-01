# PRES-DSR-SF V0.2 exact commands

The private launch receipt preserves the literal NAS paths. The public normalization below replaces only the private protocol root with `$PROTOCOL_ROOT`.

## Tests

```sh
python -m pytest -q -p no:cacheprovider \
  --junitxml "$PROTOCOL_ROOT/test_evidence/PRES_DSR_SF_TESTS_09f4600.xml" \
  experiments/lcrseg/tests/pres_dsr_sf_v0_2 \
  experiments/lcrseg/tests/pres_jascl_v0_1 \
  experiments/lcrseg/tests/gate0/test_official_model_contract.py \
  experiments/lcrseg/tests/gate0/test_pres_readonly_checkpoint_contract.py
```

## Durable validation child

```sh
bash scripts/with_nas_storage.sh python -B -m pres_dsr_sf_v0_2.run \
  --output "$PROTOCOL_ROOT/runs/formal_01" \
  --code-commit 09f4600348f8708ca9e865f7d5c925b6472cd013 \
  --test-report "$PROTOCOL_ROOT/test_evidence/PRES_DSR_SF_TEST_REPORT_09f4600.json"
```

The durable launch fixed one visible GPU, deterministic backend settings, `PYTHONHASHSEED=0`, and single-threaded CPU math. No package installation, optimizer, backward, training, or test-evaluation command ran.
