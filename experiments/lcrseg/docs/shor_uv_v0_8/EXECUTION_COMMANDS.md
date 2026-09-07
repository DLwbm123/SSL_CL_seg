# Exact execution commands

Source is frozen by commit before either full qualification run. Final source SHA and actual parent receipts accompany the report. All real commands run from the clean server checkout through with_nas_storage.sh; CUDA_VISIBLE_DEVICES is empty.

Local regression:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg /opt/miniconda3/bin/python -m shor_uv_v0_8.run_checks --output /tmp/shor_uv_v08_local_qualification

Server regression, wrapped by care_hr_v0_7_1.parent_r1:

    bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES= /home/jiangsuiyang/anaconda3/envs/py38/bin/python -m shor_uv_v0_8.run_checks --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/shor_uv_v0_8_20260907_01/A1

After exact-source local/server qualification and public freeze verification:

    bash experiments/lcrseg/scripts/with_nas_storage.sh env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lcrseg CUDA_VISIBLE_DEVICES= /home/jiangsuiyang/anaconda3/envs/py38/bin/python -m shor_uv_v0_8.run_study --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/shor_uv_v0_8_20260907_01/run_01 --qualification /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/shor_uv_v0_8_20260907_01/QUALIFICATION.json

The fixed sequence runs separate prepare, targets, five train/deploy process pairs, all-prediction seal, then independent evaluation. Each child has a create-only parent receipt and private log. No automatic retry or post-outer tuning exists.
