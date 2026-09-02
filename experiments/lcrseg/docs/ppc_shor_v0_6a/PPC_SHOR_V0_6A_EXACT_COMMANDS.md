# PPC-SHOR V0.6A exact commands

Private paths are represented by bound variables in the public record.

```bash
git switch codex/ppc-shor-v0-6a-development
python -m pytest -q experiments/lcrseg/tests/ppc_shor_v0_6a
CODE_COMMIT=$(git rev-parse HEAD)
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6a.py \
  --output <NAS_CREATE_ONLY_FORMAL_01> --code-commit "$CODE_COMMIT" \
  --test-report <NAS_TEST_REPORT> --device cuda:1
```

V0.4 `formal_03` was not an input.
