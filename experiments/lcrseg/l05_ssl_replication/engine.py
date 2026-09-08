"""One explicit recipe per process; frozen V0.1 loop with split-independent RNG adapter."""
import argparse,contextlib,json,time
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.single_teacher_scd_v0_1 import engine as old, data as data_module, objectives as obj, freeze as old_freeze
from experiments.lcrseg.single_teacher_scd_r1.engine import CountOnly,logit_diagnostics

RECIPES=json.loads(Path("experiments/lcrseg/docs/l05_ssl_replication/PROTOCOL.json").read_text())["recipes"]

def objective(sup,lu,kl,ku,arm,epoch):
    spec=RECIPES[arm]
    actual=obj.lambda_u(epoch) if spec["use_pseudo_ce"] else 0.
    return sup+actual*lu+spec["labeled_kd"]*kl,actual

def train_step(student,teacher,optimizer,labeled,unlabeled,proto,supported,arm,stage,epoch,step,counters,*,diagnostic=False):
    spec=RECIPES[arm];start=time.perf_counter();optimizer.zero_grad(set_to_none=True);key=(stage,epoch,step)
    zl,_=old.forward(student,labeled["image"],True,(*key,"supervised_head"),counters)
    sup=obj.supervised(zl,labeled["label"])
    begin=time.perf_counter()
    gas=torch.autograd.grad(sup,student.decoder.conv_logit.mu.weight,retain_graph=True)[0].detach().square()
    counters["gas_autograd"]+=1;gas_seconds=time.perf_counter()-begin
    lu=zl.sum()*0.;ku=zl.sum()*0.;pasrows=[];ds=[]
    if spec["use_unlabeled_images"]:
        if unlabeled is None:raise ValueError("L05_SSL requires current U images")
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
    vl=(labeled["label"]!=255)&labeled["geometry"]
    kl=zl.sum()*0.
    if spec["labeled_kd"]:
        if teacher is None:raise ValueError("missing predecessor")
        with torch.no_grad():
            ql=old.forward(teacher,labeled["image"],False,(*key,"teacher_labeled"),counters)[0].double().softmax(1)
        kl,di=obj.kd(zl,ql,labeled["label"],vl,vl,"D")
        ds.append(dict(branch="labeled",**di))
    elif teacher is not None:raise ValueError("S must have one model")
    loss,actual=objective(sup,lu,kl,ku,arm,epoch)
    grad={}
    if diagnostic:
        for name,l in (("supervised",sup),("raw_lu",lu),("labeled_kd",kl),("unlabeled_kd",ku)):
            grad[name]=old.grad_norm(l,student);counters["diagnostic_autograd"]+=1
    numerics={}
    if epoch%5==0:
        numerics["labeled"]=logit_diagnostics(zl,vl)
        if spec["use_unlabeled_images"]:numerics["unlabeled"]=logit_diagnostics(zu,geo)
    if not torch.isfinite(loss):raise FloatingPointError("nonfinite L05_SSL loss")
    loss.backward();counters["backward"]+=1
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
        raise FloatingPointError("nonfinite L05_SSL gradient")
    optimizer.step();counters["optimizer_steps"]+=1
    with torch.no_grad():student.decoder.conv_logit.grad_update.copy_(gas)
    return dict(loss=float(loss),sup=float(sup),ssl=float(lu),raw_lu=float(lu),weighted_lu=float(actual*lu),
        labeled_kd=float(kl),unlabeled_kd=float(ku),kd=float(spec["labeled_kd"]*kl+0.*ku),
        lambda_u=actual,actual_lambda_u=actual,nominal_lambda_u=obj.lambda_u(epoch),
        labeled_kd_coefficient=spec["labeled_kd"],unlabeled_kd_coefficient=0.,gradients=grad,pas=pasrows,
        distillation=ds,numerics=numerics,gas_seconds=gas_seconds,step_seconds=time.perf_counter()-start)


@contextlib.contextmanager
def lifecycle(arm,optimization_seed,verify=lambda:"qualification"):
    if arm not in RECIPES or optimization_seed not in (0,1,2):raise ValueError("fixed recipe/seed required")
    spec=RECIPES[arm]
    original_data,original_load,original_save=old.CurrentData,old.load_batch,old.save_checkpoint
    original_seed,original_batch,original_forward=old.stable_seed,data_module.batch_indices,old.forward
    def seed(first,*parts):
        if first!=0:raise ValueError("unexpected base random key")
        return original_seed(optimization_seed,*parts)
    def batch_indices(*args,seed_parts,**kwargs):
        if seed_parts[0]!=0:raise ValueError("unexpected order random key")
        return original_batch(*args,seed_parts=(optimization_seed,*seed_parts[1:]),**kwargs)
    def data(*args,**kwargs):
        ds=original_data(*args,**kwargs)
        return CountOnly(len(ds),ds.stage) if ds.role=="train_unlabeled" and not spec["use_unlabeled_images"] else ds
    def batch(ds,*args,**kwargs):return None if isinstance(ds,CountOnly) else original_load(ds,*args,**kwargs)
    def forward(network,x,stochastic,key,counters):
        counters.setdefault("teacher_u_forward",0)
        if "teacher_unlabeled" in key:raise PermissionError("teacher-U forward forbidden")
        if "teacher_labeled" in key:counters["teacher_l_forward"]=counters.get("teacher_l_forward",0)+1
        if "weak_head" in key or "unsupervised_head" in key:counters["student_u_forward"]=counters.get("student_u_forward",0)+1
        return original_forward(network,x,stochastic,key,counters)
    def save(path,student,optimizer,proto,support,**metadata):
        original_save(path,student,optimizer,proto,support,**metadata,recipe=arm,optimization_seed=optimization_seed,data_split_seed=0)
    def step(*args,**kwargs):
        args=list(args);args[7]=arm
        return train_step(*args,**kwargs)
    with contextlib.ExitStack() as stack:
        for module,name,value in ((old_freeze,"verify",verify),(old,"CurrentData",data),(old,"load_batch",batch),
            (old,"save_checkpoint",save),(old,"train_step",step),(old,"stable_seed",seed),(data_module,"stable_seed",seed),
            (data_module,"batch_indices",batch_indices),(old,"forward",forward)):
            stack.enter_context(patch.object(module,name,value))
        stack.enter_context(patch.object(obj,"project",side_effect=PermissionError("SCD projector forbidden")))
        if not spec["use_pseudo_ce"]:
            stack.enter_context(patch.object(old,"prototypes",lambda student,dataset,device,counters:
                (torch.zeros(3,16,device=device),torch.zeros(3,dtype=torch.bool,device=device))))
        yield

def parent_identity(payload,arm,stage,optimization_seed):
    if not payload.get("complete") or payload["stage"]!=stage-1:raise PermissionError("nonfinal/improper predecessor")
    if optimization_seed==0 and stage==1:
        if payload["arm"]!="common" or payload["source"]!="057ce07fa7bd02f8319e0c2bdd9d4adbceff23d8":
            raise PermissionError("seed0 requires original common")
    elif payload.get("optimization_seed")!=optimization_seed or payload.get("data_split_seed")!=0:
        raise PermissionError("cross-seed/split predecessor")
    if stage==1 and payload["arm"]!="common":raise PermissionError("stage1 requires same-seed common")
    if stage==2 and payload.get("recipe")!=arm:raise PermissionError("cross-recipe predecessor")

def run_stage(*,arm,optimization_seed,stage,output,parent=None,resume=False,**kwargs):
    if arm not in RECIPES or optimization_seed not in (0,1,2) or stage not in (0,1,2):raise ValueError("fixed task required")
    if stage==0 and (arm!="S" or optimization_seed==0):raise PermissionError("only new seed1/2 common")
    root=Path(output)
    if (root/"receipt.json").exists() or (root/"final_receipt.json").exists():raise FileExistsError("terminal directory cannot rerun")
    if resume:
        payload=torch.load(root/"student_latest.pt",map_location="cpu",weights_only=False)
        if payload.get("recipe")!=arm or payload.get("optimization_seed")!=optimization_seed or payload.get("data_split_seed")!=0:
            raise PermissionError("resume identity mismatch")
        lines=(root/"steps.jsonl").read_text().splitlines()
        if len(lines)!=payload["counters"]["optimizer_steps"]:raise RuntimeError("preserve checkpoint-divergent interrupted attempt")
        del payload
    elif stage:
        if stage==2 and Path(parent).parent.parent!=root.parent:raise PermissionError("stage2 must be this arm's stage1")
        payload=torch.load(parent,map_location="cpu",weights_only=False)
        parent_identity(payload,arm,stage,optimization_seed);del payload
    if optimization_seed:
        from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier
        _official_probabilistic_classifier(kwargs["reference"],upstream_path=old.UPSTREAM_PATH)
    from .freeze import verify
    with lifecycle(arm,optimization_seed,verify):
        result=old.run_stage(arm="common" if stage==0 else "S" if arm=="S" else "D",stage=stage,parent=parent,output=output,resume=resume,**kwargs)
    result.update(recipe=arm,optimization_seed=optimization_seed,data_split_seed=0)
    if result.get("status")=="COMPLETE":
        assert result["counters"].get("teacher_u_forward",0)==0
        old.write_json(root/"final_receipt.json",result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("data","reference","output","arm"):p.add_argument("--"+x,required=True)
    p.add_argument("--stage",type=int,required=True);p.add_argument("--optimization-seed",type=int,required=True)
    p.add_argument("--parent");p.add_argument("--resume",action="store_true")
    a=vars(p.parse_args())
    try:run_stage(**a,device=torch.device("cuda:0"))
    except BaseException as error:
        if Path(a["output"]).is_dir():old.write_json(Path(a["output"])/"failure.json",dict(status="ENGINEERING_FAILURE",error=repr(error)))
        raise
