# R2 commands and boundaries

Local interpreter: /opt/miniconda3/bin/python. Server: existing /home/jiangsuiyang/anaconda3/envs/py38/bin/python. No environment installation or checkpoint reload. CPU-only CUDA_VISIBLE_DEVICES is empty. Every server child is launched via the repository NAS wrapper, with a create-only parent receipt and private log.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg python -m care_hr_v0_7_1.run_checks_r2 --output NEW_A1_OUTPUT
bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES= python -m care_hr_v0_7_1.parent_r1 --receipt NEW_PARENT_RECEIPT --log NEW_PRIVATE_LOG -- python -m care_hr_v0_7_1.execute_r2 qualify --output NEW_R2_NAS_RUN --qualification PUBLISHED_R2_QUALIFICATION
# Publish and anonymously verify the R2 dual-source seal before this separate process:
bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES= python -m care_hr_v0_7_1.parent_r1 --receipt NEW_PREFLIGHT_RECEIPT --log NEW_PRIVATE_LOG -- python -m care_hr_v0_7_1.execute_r2 preflight --output SAME_R2_NAS_RUN
# Require all198 stat-only admission, then launch a fresh evaluator process:
bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES= python -m care_hr_v0_7_1.parent_r1 --receipt NEW_EVALUATOR_RECEIPT --log NEW_PRIVATE_LOG -- python -m care_hr_v0_7_1.execute_r2 evaluate --output SAME_R2_NAS_RUN
```

The original parent_r1 utility is reused unchanged only to record actual new child exits; receipts and logs belong exclusively to the new R2 root. The R1 run and failed reservation remain read-only.
