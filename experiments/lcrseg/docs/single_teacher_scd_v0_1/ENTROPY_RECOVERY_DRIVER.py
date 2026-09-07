import json,os,sys,time
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from experiments.lcrseg.single_teacher_scd_v0_1.freeze import verify
source=verify();base=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01");run=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/entropy_repair_run_01")
q=json.loads(Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/qualification_entropy_repair/tests.json").read_text())
assert q["source"]==source and q["status"]=="PASS" and q["real_two_case_smoke"]["loss_decreased"]
run.mkdir();(run/"E").mkdir()
write_json(run/"reservation.json",dict(source=source,arm="E",stages=[1,2],expected_updates=5300,
qualification="/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/qualification_entropy_repair/tests.json",common_parent="/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/run_01/common/stage0/student_latest.pt",
prior_failed_attempt_preserved="/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/run_01/E",discarded_prior_formal_updates=1,diagnostic_replay_updates=1,
change="KL zero-target continuous entropy extension; all scientific parameters unchanged",created_unix=time.time()))
reference="/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE"
module="experiments.lcrseg.single_teacher_scd_v0_1."
env=dict(os.environ,CUDA_VISIBLE_DEVICES="5",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1")
parent=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/run_01/common/stage0/student_latest.pt")
for stage in (1,2):
 root=run/"E"/f"stage{stage}"
 command=[sys.executable,"-m",module+"engine","--data","/home/jiangsuiyang/SSL_CL","--reference",reference,"--output",str(root),"--arm","E","--stage",str(stage),"--parent",str(parent)]
 child(command,root.parent/f"stage{stage}_train.log",root.parent/f"stage{stage}_train_exit.json",env)
 child([sys.executable,"-m",module+"evaluate","--data","/home/jiangsuiyang/SSL_CL","--reference",reference,"--checkpoint",str(root/"student_latest.pt"),"--output",str(root/"val.json")],root.parent/f"stage{stage}_eval.log",root.parent/f"stage{stage}_eval_exit.json",env)
 parent=root/"student_latest.pt"
child([sys.executable,"-m",module+"deploy","--checkpoint",str(parent),"--reference",reference],run/"E/deploy.log",run/"E/deploy_exit.json",env)
write_json(run/"parent_receipt.json",dict(status="COMPLETE",source=source,expected_updates=5300,stages=[1,2],arm="E"))
