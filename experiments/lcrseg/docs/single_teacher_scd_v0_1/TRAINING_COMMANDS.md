# Exact execution commands

All server commands use the existing environment and `bash experiments/lcrseg/scripts/with_nas_storage.sh` from the new source checkout. Output, scratch and cache remain on NAS. The attachment default GPU restriction is superseded by the user's direct GPU4/5/6/7 authorization.

Local qualification: `/opt/miniconda3/bin/python -m experiments.lcrseg.single_teacher_scd_v0_1.qualification --output /tmp/scd_source_qualification --reference /Users/bominwang/Desktop/codes/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE`.

Server qualification: `bash experiments/lcrseg/scripts/with_nas_storage.sh env CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jiangsuiyang/anaconda3/envs/py38/bin/python -m experiments.lcrseg.single_teacher_scd_v0_1.qualification --device cuda:0 --reference /home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE --data /home/jiangsuiyang/SSL_CL --output /data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/qualification`.

Formal parent: the same NAS wrapper and interpreter, module `experiments.lcrseg.single_teacher_scd_v0_1.execute`, `--data /home/jiangsuiyang/SSL_CL --reference` as above, `--qualification` the exact-source PASS receipt, and `--output .../run_01`. Fixed lanes after shared stage0: GPU4 S then D; GPU5 A then E; GPU6 B; GPU7 C. Every stage evaluates in a separate process; each child command, exit code and physical GPU mapping is recorded on NAS.

Reporting: module `experiments.lcrseg.single_teacher_scd_v0_1.report --run .../run_01 --output .../public_results`. It requires all 13 stages and exactly 39,800 ledger and checkpoint steps. It writes a create-only scientific adjudication once. No formal model initialization reads smoke weights.
