"""Student-only final bundle and a teacher-hidden deployment equality check."""
import argparse
import gc
from pathlib import Path
import torch

from .engine import model, write_json, audit_live_models
from .evaluate import infer


def load_student(path,reference,device):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    state=payload.pop("student")
    del payload
    return model(reference,device,state).eval()


def export(checkpoint,reference,device=None):
    checkpoint=Path(checkpoint);root=checkpoint.parent
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload["stage"]!=2 or not payload["complete"]:raise ValueError("final complete student required")
    student=payload.pop("student")
    bundle={k:payload[k] for k in ("arm","stage","complete","student_hash","source")}
    del payload
    target=root/"student_deploy.pt"
    if target.exists():raise FileExistsError(target)
    torch.save(dict(student=student,**bundle),target)
    del student;gc.collect()
    device=torch.device("cuda:0") if device is None else device
    synthetic=torch.linspace(0,1,3*32*32,device=device).reshape(1,3,32,32)
    network=load_student(checkpoint,reference,device)
    audit_live_models(1)
    before=infer(network,synthetic).cpu()
    del network;gc.collect()
    teacher=root/"prev_teacher.pt";hidden=root/"teacher_disabled_during_deploy_check.pt"
    if teacher.exists():teacher.rename(hidden)
    try:
        network=load_student(target,reference,device)
        audit_live_models(1)
        after=infer(network,synthetic).cpu()
        if not torch.equal(before,after):raise RuntimeError("student deployment predictions differ")
    finally:
        if hidden.exists():hidden.rename(teacher)
    write_json(root/"deployment.json",dict(status="PASS",student_hash=bundle["student_hash"],
        bundle_fields=["student",*bundle],models=1,teacher_hidden=True,
        equal_synthetic_prediction=True,gt_accesses=0,posterior_forward_calls=2,optimizer_updates=0))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--reference",required=True)
    a=p.parse_args();export(a.checkpoint,a.reference)
