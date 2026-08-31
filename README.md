# SSL_CL_seg

Research code, frozen protocol specifications, tests, configurations, and
reproducibility metadata for semi-supervised continual segmentation experiments.

The active implementation is under `experiments/lcrseg/`.

## Current JASCL / DI-DMPA handoff

For the current method, start with the [experiment report and jiangsuiyang
migration handoff](experiments/lcrseg/docs/di_dmpa_jascl/GATE1C_V22_MIGRATION_HANDOFF_20260831.md).
It distinguishes the tested code, verified integration, interrupted full-run
observation, and missing private inputs. This is a source/results handoff,
not permission to rerun an attempt or start method training on another server.

## Source-only repository

This Git repository intentionally excludes raw datasets, checkpoints, runtime
logs, rendered media, local environments, caches, and full third-party source
mirrors. Frozen manifests and split definitions are tracked, while the
corresponding HDF5 data remain external.

Before running or changing an experiment, read:

- `experiments/lcrseg/AGENTS.md`
- `experiments/lcrseg/METHOD_SPEC_V0_1.md`
- `experiments/lcrseg/IMPLEMENTATION_CONTRACT_V0_1.md`
- `experiments/lcrseg/STATUS.md`

Run the project test boundary from the active experiment directory:

```bash
cd experiments/lcrseg
PYTHONPATH=.:tests python -m pytest -q tests
```
