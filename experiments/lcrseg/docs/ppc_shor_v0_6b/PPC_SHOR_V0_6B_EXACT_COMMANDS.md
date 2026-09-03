# PPC-SHOR V0.6B exact commands

Qualification and formal execution use the registered source with the project NAS wrapper:

```bash
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6b.py --mode qualify --output QUALIFICATION_ROOT --test-report TEST_REPORT
bash experiments/lcrseg/scripts/with_nas_storage.sh python experiments/lcrseg/ppc_shor_v0_6b.py --mode formal --output FORMAL_ROOT --qualification-root QUALIFICATION_ROOT --code-commit QUALIFIED_FREEZE_COMMIT --device cuda:1
```

The qualified freeze remote SHA is verified by the formal source gate before materialization.
V0.4 `formal_03` is not an input.
