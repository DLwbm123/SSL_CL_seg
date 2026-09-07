"""Explicit R1 objectives, reusing the frozen V0.1 lifecycle and pure helpers."""
import argparse
import contextlib
import json
import time
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from experiments.lcrseg.single_teacher_scd_v0_1 import engine as old
from experiments.lcrseg.single_teacher_scd_v0_1 import objectives as obj
from experiments.lcrseg.single_teacher_scd_v0_1 import freeze as old_freeze
from . import solver

ARMS={"U0":dict(use_u=True,lu=False,kl=.5,ku=.5,project=False),
      "L05":dict(use_u=False,lu=False,kl=.5,ku=0.,project=False),
      "L10":dict(use_u=False,lu=False,kl=1.,ku=0.,project=False),
      "E_R1":dict(use_u=True,lu=True,kl=.5,ku=.5,project=True)}

def objective(sup,lu,kl,ku,arm,epoch):
    spec=ARMS[arm]
    actual=obj.lambda_u(epoch) if spec["lu"] else 0.
    kd=.5*(kl+ku) if spec["use_u"] else spec["kl"]*kl
    return sup+actual*lu+kd,actual

def projected_kd(logits,q_raw,y,reliable,valid):
    begin=time.perf_counter()
    lp=logits.double().movedim(1,-1).reshape(-1,logits.shape[1]).log_softmax(-1)
    p=lp.detach().exp()
    q0=q_raw.detach().double().movedim(1,-1).reshape_as(p);q0=q0/q0.sum(-1,keepdim=True)
    q=(1-obj.EPS_Q)*q0+obj.EPS_Q/p.shape[-1]
    ref=y.flatten().clamp(0,p.shape[-1]-1);v=valid.flatten().bool();rel=reliable.flatten().bool()&v
    target,conflict,info=solver.project(p,q,ref,rel)
    logtarget=target.log().masked_fill(target==0,0.)
    per=(target*(logtarget-lp)).sum(-1)
    loss=(per*v).sum()/v.sum().clamp_min(1)
    rows=[]
    for c in range(p.shape[-1]):
        j=v&(ref==c);active=j&conflict;finite=active&torch.isfinite(info["log_eta"])
        logetas=info["log_eta"][finite]
        rows.append(dict(class_id=c,pixels=int(j.sum()),reliable=int((j&rel).sum()),
            conflicts=int(active.sum()),projected=int(active.sum()),drop_kl_mass=0.,drop_gradient_l2_sum=0.,
            r_q_norm_sum=float(((target-q).norm(dim=-1)*active).sum()),r_p_norm_sum=float(((target-p).norm(dim=-1)*active).sum()),
            nondegenerate=int((active&((target-q).norm(dim=-1)>1e-8)&((target-p).norm(dim=-1)>1e-8)).sum()),
            log_eta_max=float(logetas.max()) if logetas.numel() else None,
            log_eta_logsumexp=float(torch.logsumexp(logetas,0)) if logetas.numel() else None,
            finite_positive_eta_count=int(finite.sum()),boundary_infinite_eta_count=int((active&info["boundary"]).sum()),
            analytical_brackets=int((active&info["analytic"]).sum()),iterations=solver.ITERATIONS,
            residual_max=float(info["residual"][j].max()) if j.any() else 0.,
            compensated_v_difference_max=float(info["compensated_v_difference"][j].max()) if j.any() else 0.,
            smoothing_l1_sum=float(((q-q0).abs().sum(-1)*j).sum())))
    return loss,dict(classes=rows,solver_seconds=time.perf_counter()-begin,bracket_max=0,
        target_temporary_bytes=sum(x.numel()*x.element_size() for x in (p,q0,q,target,lp,*[v for v in info.values() if isinstance(v,torch.Tensor)])))

@torch.no_grad()
def logit_diagnostics(z,valid):
    zz=z.detach().double().movedim(1,-1)[valid]
    if not len(zz):return dict(evaluable_pixels=0)
    p=zz.log_softmax(-1).exp();pm=p.max(-1).values
    return dict(evaluable_pixels=len(zz),logit_min=float(zz.min()),logit_max=float(zz.max()),
        pmax_quantiles=[float(x) for x in torch.quantile(pm,pm.new_tensor([.1,.5,.9,.99,.999]))],
        numerical_one_hot_fraction=float((pm==1.).double().mean()),
        predicted_class_fraction=[float((p.argmax(-1)==c).double().mean()) for c in range(z.shape[1])])

def train_step(student,teacher,optimizer,labeled,unlabeled,proto,supported,arm,stage,epoch,step,counters,*,diagnostic=False):
    spec=ARMS[arm];start=time.perf_counter();optimizer.zero_grad(set_to_none=True);key=(stage,epoch,step)
    zl,_=old.forward(student,labeled["image"],True,(*key,"supervised_head"),counters)
    sup=obj.supervised(zl,labeled["label"])
    begin=time.perf_counter()
    gas=torch.autograd.grad(sup,student.decoder.conv_logit.mu.weight,retain_graph=True)[0].detach().square()
    counters["gas_autograd"]+=1;gas_seconds=time.perf_counter()-begin
    lu=zl.sum()*0.;ku=zl.sum()*0.;pasrows=[];ds=[]
    if spec["use_u"]:
        if unlabeled is None:raise ValueError("U0/E_R1 require U images")
        xu,geo=unlabeled["image"],unlabeled["geometry"]
        with torch.no_grad():
            zw,feature=old.forward(student,xu,False,(*key,"weak_head"),counters)
            pseudo,mask,_=obj.pas(zw,feature,proto,supported,geo)
        zu,_=old.forward(student,old.strong(xu,stage,epoch,step),True,(*key,"unsupervised_head"),counters)
        lu=obj.ssl_ce(zu,pseudo,mask,geo)
        for c in range(3):
            j=pseudo==c;pasrows.append(dict(class_id=c,predicted_pixels=int((j&geo).sum()),
                accepted_pixels=int((j&mask).sum()),prototype_valid=bool(supported[c])))
    elif unlabeled is not None:
        raise PermissionError("label-only arm received U images")
    with torch.no_grad():
        ql=old.forward(teacher,labeled["image"],False,(*key,"teacher_labeled"),counters)[0].double().softmax(1)
        if spec["use_u"]:qu=old.forward(teacher,xu,False,(*key,"teacher_unlabeled"),counters)[0].double().softmax(1)
    vl=(labeled["label"]!=255)&labeled["geometry"]
    kd_fn=projected_kd if spec["project"] else lambda *args:obj.kd(*args,"D")
    kl,di=kd_fn(zl,ql,labeled["label"],vl,vl);ds.append(dict(branch="labeled",**di))
    if spec["use_u"]:
        ku,di=kd_fn(zu,qu,pseudo,mask,geo);ds.append(dict(branch="unlabeled",**di))
    loss,actual=objective(sup,lu,kl,ku,arm,epoch)
    grad={}
    if diagnostic:
        for name,l in (("supervised",sup),("raw_lu",lu),("labeled_kd",kl),("unlabeled_kd",ku)):
            grad[name]=old.grad_norm(l,student);counters["diagnostic_autograd"]+=1
    numerics={}
    if epoch%5==0:
        numerics["labeled"]=logit_diagnostics(zl,vl)
        if spec["use_u"]:numerics["unlabeled"]=logit_diagnostics(zu,geo)
    if not torch.isfinite(loss):raise FloatingPointError("nonfinite R1 loss")
    loss.backward();counters["backward"]+=1
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
        raise FloatingPointError("nonfinite R1 gradient")
    optimizer.step();counters["optimizer_steps"]+=1
    with torch.no_grad():student.decoder.conv_logit.grad_update.copy_(gas)
    return dict(loss=float(loss),sup=float(sup),ssl=float(lu),raw_lu=float(lu),weighted_lu=float(actual*lu),
        labeled_kd=float(kl),unlabeled_kd=float(ku),kd=float(spec["kl"]*kl+spec["ku"]*ku),
        lambda_u=actual,actual_lambda_u=actual,nominal_lambda_u=obj.lambda_u(epoch),
        labeled_kd_coefficient=spec["kl"],unlabeled_kd_coefficient=spec["ku"],gradients=grad,pas=pasrows,
        distillation=ds,numerics=numerics,gas_seconds=gas_seconds,step_seconds=time.perf_counter()-start)

@contextlib.contextmanager
def lifecycle(arm,verify):
    """Bind one process to one explicit R1 recipe without editing frozen V0.1 files."""
    spec=ARMS[arm]
    original_data,original_load,original_save=old.CurrentData,old.load_batch,old.save_checkpoint
    def data(*args,**kwargs):
        ds=original_data(*args,**kwargs)
        if not spec["use_u"] and ds.role=="train_unlabeled":
            # Retain only a count: neither an image path nor a hidden-label field survives.
            return CountOnly(len(ds),ds.stage)
        return ds
    def batch(ds,*args,**kwargs):
        return None if isinstance(ds,CountOnly) else original_load(ds,*args,**kwargs)
    def save(path,student,optimizer,proto,support,**metadata):
        original_save(path,student,optimizer,proto,support,**metadata,r1_arm=arm)
    def step(*args,**kwargs):
        args=list(args);args[7]=arm
        return train_step(*args,**kwargs)
    with contextlib.ExitStack() as stack:
        for module,name,value in [(old_freeze,"verify",verify),(old,"CurrentData",data),(old,"load_batch",batch),
            (old,"save_checkpoint",save),(old,"train_step",step)]:
            stack.enter_context(patch.object(module,name,value))
        if not spec["use_u"]:
            stack.enter_context(patch.object(old,"prototypes",lambda student,dataset,device,counters:
                (torch.zeros(3,16,device=device),torch.zeros(3,dtype=torch.bool,device=device))))
        yield

class CountOnly:
    def __init__(self,count,stage):
        self.count,self.stage,self.checked=count,stage,set()
    def __len__(self):return self.count
    def __getitem__(self,index):raise PermissionError("label-only U image sentinel")

def run_stage(*,arm,stage,parent,output,resume=False,**kwargs):
    if arm not in ARMS or stage not in (1,2):raise ValueError("unknown R1 arm/stage")
    if resume:
        payload=torch.load(Path(output)/"student_latest.pt",map_location="cpu",weights_only=False)
        if payload.get("r1_arm")!=arm:raise PermissionError("cross-arm R1 resume")
        logged=[json.loads(x) for x in (Path(output)/"steps.jsonl").read_text().splitlines()]
        if len(logged)!=payload["counters"]["optimizer_steps"]:
            raise RuntimeError("resume requires a checkpoint-aligned ledger; preserve divergent attempt")
        del payload
    elif stage==2:
        if Path(parent).parent.parent!=Path(output).parent:raise PermissionError("R1 stage2 must be own arm")
        payload=torch.load(parent,map_location="cpu",weights_only=False)
        if payload.get("r1_arm")!=arm:raise PermissionError("cross-arm R1 predecessor")
        del payload
    from .freeze import verify
    with lifecycle(arm,verify):
        result=old.run_stage(arm="E" if ARMS[arm]["project"] else "D",stage=stage,parent=parent,output=output,resume=resume,**kwargs)
    result["r1_arm"]=arm
    if result.get("status")=="COMPLETE":old.write_json(Path(output)/"r1_receipt.json",result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("data","reference","output","arm","parent"):p.add_argument("--"+x,required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--resume",action="store_true")
    a=vars(p.parse_args())
    try:run_stage(**a,device=torch.device("cuda:0"))
    except BaseException as error:
        root=Path(a["output"])
        if root.is_dir():old.write_json(root/"r1_failure.json",dict(status="ENGINEERING_FAILURE",arm=a["arm"],error=repr(error)))
        raise
