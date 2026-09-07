# R1 reproducible execution

All server commands use the existing Python environment and the repository NAS wrapper. Source is transferred as a standard git bundle into a new checkout; no old checkout or result is overwritten. Physical CUDA visibility is 4,5. Cached diagnostics need no GPU forward. Private absolute output arguments remain in NAS parent receipts.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg python -m care_hr_v0_7_1.run_checks --output NEW_A1_OUTPUT
bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES=4,5 python -m care_hr_v0_7_1.parent_r1 --receipt NEW_PARENT_RECEIPT --log NEW_RUN_LOG -- python -m care_hr_v0_7_1.execute_r1 blind --output NEW_NAS_RUN --test-report EXACT_SOURCE_TEST_REPORT --admission PUBLISHED_A1_RELEASE
# Only after child exit=0, verified seal, and published seal digest:
bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES=4,5 python -m care_hr_v0_7_1.parent_r1 --receipt NEW_EVALUATOR_RECEIPT --log NEW_EVALUATOR_LOG -- python -m care_hr_v0_7_1.execute_r1 evaluate --output SAME_SEALED_NAS_RUN
```

The evaluator is a separate OS process. Its reservation uses exclusive creation and prevents duplicate completed execution. Frozen cache validation occurs before every real label access. GT HDF5 files are opened only for the admitted own-seed train_labeled rows.
