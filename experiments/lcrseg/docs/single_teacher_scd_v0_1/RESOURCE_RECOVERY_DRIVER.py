import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.lcrseg.single_teacher_scd_v0_1.engine import write_json
from experiments.lcrseg.single_teacher_scd_v0_1.execute import child
from experiments.lcrseg.single_teacher_scd_v0_1.freeze import verify
base=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01");run=base/"run_01";source=verify()
probe=json.loads((base/"synthetic_resource_probe.json").read_text())
assert probe["status"]=="PASS" and probe["source"]==source
minimum=(probe["actual_max_reserved"]+probe["resource_headroom_bytes"]+1048575)//1048576
write_json(base/"resource_recovery_reservation.json",dict(source=source,scope="Only unstarted A/C/E, unchanged source/config/data and optimizer budgets",
original_free_mib_floor=8192,measured_free_mib_floor=minimum,probe="synthetic_resource_probe.json",
authorization="User allows GPU4/5/6/7 whenever enough memory remains; scientific gates unchanged",
initial_parent_receipt_preserved=True,training_retries=0,created_unix=time.time()))
reference="/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE"
module="experiments.lcrseg.single_teacher_scd_v0_1."
def admission(gpu):
 while True:
  rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,uuid","--format=csv,noheader,nounits"],text=True).splitlines()
  row=next(r.split(", ") for r in rows if int(r.split(",")[0])==gpu)
  if int(row[1])>=minimum:return dict(physical_index=gpu,free_mib=int(row[1]),uuid=row[2],measured_admission_mib=minimum)
  time.sleep(30)
def stage(arm,t,gpu,parent):
 root=run/arm/f"stage{t}"
 if root.exists():raise RuntimeError("recovery cannot overwrite or retry a started stage")
 resource=admission(gpu)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1")
 write_json(root.parent/f"stage{t}_resource.json",resource)
 command=[sys.executable,"-m",module+"engine","--data","/home/jiangsuiyang/SSL_CL","--reference",reference,"--output",str(root),"--arm",arm,"--stage",str(t),"--parent",str(parent)]
 child(command,root.parent/f"stage{t}_train.log",root.parent/f"stage{t}_train_exit.json",env)
 command=[sys.executable,"-m",module+"evaluate","--data","/home/jiangsuiyang/SSL_CL","--reference",reference,"--checkpoint",str(root/"student_latest.pt"),"--output",str(root/"val.json")]
 child(command,root.parent/f"stage{t}_eval.log",root.parent/f"stage{t}_eval_exit.json",env)
 if t==2:
  child([sys.executable,"-m",module+"deploy","--checkpoint",str(root/"student_latest.pt"),"--reference",reference],root.parent/"deploy.log",root.parent/"deploy_exit.json",env)
 return root/"student_latest.pt"
def lane(gpu,arms):
 results=[]
 for arm in arms:
  try:
   first=stage(arm,1,gpu,run/"common/stage0/student_latest.pt")
   stage(arm,2,gpu,first)
   results.append(dict(arm=arm,status="COMPLETE"))
  except BaseException as error:
   results.append(dict(arm=arm,status="INCOMPLETE_TRAINING_MATRIX",error=repr(error)))
 return results
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(lane,5,"AE"),pool.submit(lane,7,"C")]
 results=[x for f in futures for x in f.result()]
write_json(base/"resource_recovery_parent_receipt.json",dict(source=source,results=results,all_recovery_arms_complete=all(x["status"]=="COMPLETE" for x in results)))
if not all(x["status"]=="COMPLETE" for x in results):raise SystemExit(1)
