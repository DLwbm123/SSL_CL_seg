import json
from pathlib import Path
import torch
from experiments.lcrseg.single_teacher_scd_v0_1 import objectives as obj
from experiments.lcrseg.single_teacher_scd_v0_1.engine import model,optimizer_for,train_step,write_json,precision
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData,batches,load_batch
root=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/engineering_bracket_probe_01");root.mkdir()
old=Path("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/entropy_repair_run_01/E/stage1")
device=torch.device("cuda:0");reference="/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE"
payload=torch.load(old/"student_latest.pt",map_location="cpu")
assert payload["epoch"]==47 and payload["counters"]["optimizer_steps"]==1504
student=model(reference,device,payload.pop("student"));optimizer=optimizer_for(student);optimizer.load_state_dict(payload["optimizer"])
proto,support=payload["prototypes"].to(device),payload["supported"].to(device)
torch.set_rng_state(payload["cpu_rng"]);torch.cuda.set_rng_state(payload["cuda_rng"]);del payload
teacher_payload=torch.load(old/"prev_teacher.pt",map_location="cpu")
teacher=model(reference,device,teacher_payload.pop("state")).eval();del teacher_payload
for parameter in teacher.parameters():parameter.requires_grad_(False)
labeled=CurrentData("/home/jiangsuiyang/SSL_CL",1,"train_labeled")
unlabeled=CurrentData("/home/jiangsuiyang/SSL_CL",1,"train_unlabeled")
lb=batches(len(labeled),1,48,"labeled_order",32);ub=batches(len(unlabeled),1,48,"unlabeled_order",32)
c=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
original=obj.project
diagnostic={}
def wrapped(p,q,y,reliable,**kwargs):
 try:return original(p,q,y,reliable,**kwargs)
 except FloatingPointError:
  a=p-torch.nn.functional.one_hot(y,p.shape[-1]).double()
  conflict=reliable&(((p-q)*a).sum(-1)<0);ix=conflict.nonzero().flatten()
  aa=a[ix]/a[ix].abs().amax(-1)[:,None];b=(aa*p[ix]).sum(-1)
  rr=(q[ix].log()-float(2**60)*aa).softmax(-1);gap=(rr*aa).sum(-1)-b;bad=gap>0
  diagnostic.update(failed_pixels=int(bad.sum()),conflict_pixels=int(conflict.sum()),
   final_gap_min=float(gap[bad].min()),final_gap_max=float(gap[bad].max()),
   b_minus_min_a_min=float((b-aa.min(-1).values)[bad].min()),
   p_sum_min=float(p[ix][bad].sum(-1).min()),p_sum_max=float(p[ix][bad].sum(-1).max()))
  torch.save(dict(p=p[ix][bad].cpu(),q=q[ix][bad].cpu(),y=y[ix][bad].cpu(),a=aa[bad].cpu(),b=b[bad].cpu()),root/"private_failure_vectors.pt")
  raise
obj.project=wrapped
try:
 for j in range(4):
  for group in optimizer.param_groups:group["lr"]=obj.learning_rate(1504+j,3200)
  left=load_batch(labeled,lb[j],1,48,j,"labeled",device);right=load_batch(unlabeled,ub[j],1,48,j,"unlabeled",device)
  out=train_step(student,teacher,optimizer,left,right,proto,support,"E",1,48,j,c,diagnostic=False)
  write_json(root/f"replay_step{j}.json",out)
 write_json(root/"receipt.json",dict(status="NOT_REPRODUCED",counters=c))
except FloatingPointError as error:
 write_json(root/"receipt.json",dict(status="REPRODUCED",source="270e23985c8d78d0508fe6c4d43150f6ebffcc92",error=repr(error),checkpoint_updates=1504,failed_step=1508,formal_updates=0,engineering_replay_updates=c["optimizer_steps"],diagnostic=diagnostic))
 print((root/"receipt.json").read_text(),flush=True)

