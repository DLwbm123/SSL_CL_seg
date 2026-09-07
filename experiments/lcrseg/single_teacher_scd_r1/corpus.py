"""One authorized replay: exactly three successful updates, private full-vector capture."""
import argparse,json
from pathlib import Path
import torch
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from experiments.lcrseg.single_teacher_scd_v0_1 import engine as old,objectives as obj
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData,batches,load_batch
from .freeze import verify

def capture(old_root,output,data,reference):
    source=verify();output=Path(output);output.mkdir()
    root=Path(old_root)/"entropy_repair_run_01/E/stage1";device=torch.device("cuda:0")
    payload=torch.load(root/"student_latest.pt",map_location="cpu")
    assert payload["epoch"]==47 and payload["counters"]["optimizer_steps"]==1504
    s=old.model(reference,device,payload.pop("student"));opt=old.optimizer_for(s);opt.load_state_dict(payload["optimizer"])
    proto,support=payload["prototypes"].to(device),payload["supported"].to(device)
    torch.set_rng_state(payload["cpu_rng"]);torch.cuda.set_rng_state(payload["cuda_rng"]);del payload
    payload=torch.load(root/"prev_teacher.pt",map_location="cpu");t=old.model(reference,device,payload.pop("state")).eval();del payload
    for p in t.parameters():p.requires_grad_(False)
    old.audit_live_models(2)
    l=CurrentData(data,1,"train_labeled");u=CurrentData(data,1,"train_unlabeled")
    lb=batches(len(l),1,48,"labeled_order",32);ub=batches(len(u),1,48,"unlabeled_order",32)
    c=dict(forward=0,gas_autograd=0,diagnostic_autograd=0,backward=0,optimizer_steps=0)
    files=[];original=obj.project
    def wrapped(p,q,y,reliable):
        path=output/f"vectors_{len(files):02d}.pt"
        torch.save(dict(p=p.cpu(),q=q.cpu(),y=y.cpu(),reliable=reliable.cpu()),path)
        record=dict(file=path.name,rows=len(p),sha256=sha256(path),old_status="PENDING")
        files.append(record)
        try:
            result=original(p,q,y,reliable);record["old_status"]="PASS"
            return result
        except FloatingPointError as error:
            record["old_status"]=str(error);raise
    obj.project=wrapped
    prior=[json.loads(x) for x in (root/"steps.jsonl").read_text().splitlines()][-3:]
    matched=[]
    try:
        for j in range(4):
            for g in opt.param_groups:g["lr"]=obj.learning_rate(1504+j,3200)
            a=load_batch(l,lb[j],1,48,j,"labeled",device);b=load_batch(u,ub[j],1,48,j,"unlabeled",device)
            info=old.train_step(s,t,opt,a,b,proto,support,"E",1,48,j,c)
            matched.append(all(info[k]==prior[j][k] for k in ("loss","sup","ssl","kd")))
    except FloatingPointError as error:
        assert str(error)=="SCD root not bracketed" and c["optimizer_steps"]==3 and all(matched)
        assert len(files)==7
        old.write_json(output/"manifest.json",dict(source=source,status="COMPLETE_PRIVATE_CORPUS",files=files,
            engineering_updates=3,formal_updates=0,maximum_authorized_diagnostic_updates=8,
            replayed_scalar_losses_equal=matched,total_vectors=sum(x["rows"] for x in files),
            captured_all_projection_calls=True,counters=c,hidden_gt_accesses=0))
        return
    finally:obj.project=original
    raise RuntimeError("frozen failure not reproduced")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for x in ("old-root","output","data","reference"):p.add_argument("--"+x,required=True)
    capture(**vars(p.parse_args()))
