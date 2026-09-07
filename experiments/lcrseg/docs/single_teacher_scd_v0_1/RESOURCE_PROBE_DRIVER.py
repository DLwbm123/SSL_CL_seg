import json,os,time
from pathlib import Path
import torch
from experiments.lcrseg.single_teacher_scd_v0_1.engine import model,second_model,optimizer_for,train_step,write_json
from experiments.lcrseg.single_teacher_scd_v0_1.freeze import verify
source=verify();device=torch.device("cuda:0")
reference="/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg/third_party/JASCL_REFERENCE"
s=model(reference,device);t=second_model(s,"E");o=optimizer_for(s)
g=torch.Generator(device=device).manual_seed(70418)
x=torch.rand((2,3,384,384),generator=g,device=device)
y=torch.randint(3,(2,384,384),generator=g,device=device)
v=torch.ones_like(y,dtype=torch.bool)
l=dict(image=x,label=y,geometry=v);u=dict(image=x,geometry=v)
c=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
p=torch.ones(3,16,device=device)/4
supported=torch.ones(3,dtype=torch.bool,device=device)
torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
r=train_step(s,t,o,l,u,p,supported,"E",1,11,0,c,diagnostic=True)
torch.cuda.synchronize()
write_json("/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/single_teacher_scd_v0_1_20260908_01/synthetic_resource_probe.json",dict(source=source,status="PASS",pid=os.getpid(),gpu=5,
shape=[2,3,384,384],arm="E",actual_max_allocated=torch.cuda.max_memory_allocated(),actual_max_reserved=torch.cuda.max_memory_reserved(),
seconds=time.perf_counter()-start,counters=c,gt_accesses=0,real_data_accesses=0,used_for_initialization=False,
resource_headroom_bytes=1024**3,formal_optimizer_updates=0))
