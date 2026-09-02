# RC-SHOR V0.5 exact commands

```sh
PYTHONPATH=experiments/lcrseg python -m pytest --import-mode=importlib experiments/lcrseg/tests/rc_shor_v0_5 experiments/lcrseg/tests/shor_jascl_v0_3 experiments/lcrseg/tests/shor_v0_4_test --junitxml=PYTEST_XML
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/rc_shor_v0_5.py --output NAS_CREATE_ONLY_ROOT --code-commit FREEZE_COMMIT --test-report TEST_REPORT --device cuda:0
```
