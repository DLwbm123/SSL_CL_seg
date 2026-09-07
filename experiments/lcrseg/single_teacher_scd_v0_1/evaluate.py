"""Independent last-student evaluator. Only seen-domain val may expose GT."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from experiments.lcrseg.care_hr_v0_7_1.scoring_r1 import case_metrics
from .data import CurrentData, DOMAINS
from .engine import model, write_json, state_hash


@torch.no_grad()
def infer(student, image):
    return student(image,stochastic_classifier=False)[0].argmax(1)


def aggregate(rows):
    keys=("macro_fg_dice","background_dice","rim_dice","cup_dice","mean_iou")
    valid=[r for r in rows if r["has_evaluable_gt"]]
    return {**{k:float(np.mean([r[k] for r in rows])) for k in keys},
        "cases":len(rows),"evaluable_cases":len(valid),
        "evaluable_macro_fg_dice":float(np.mean([r["macro_fg_dice"] for r in valid])) if valid else None,
        "empty_support_evidence":bool(not valid)}


def evaluate(*,data,reference,checkpoint,output,device,expected=None,shape=(384,384)):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if not payload["complete"]: raise ValueError("only final stage checkpoint can be evaluated")
    stage,arm,expected_hash=payload["stage"],payload["arm"],payload["student_hash"]
    state=payload.pop("student")
    del payload  # Inference never restores optimizer, prototypes or a teacher.
    student=model(reference,device,state).eval()
    if state_hash(student)!=expected_hash: raise RuntimeError("evaluator student identity mismatch")
    rows=[]
    for domain in range(stage+1):
        args={} if expected is None else dict(expected=expected)
        dataset=CurrentData(data,stage,"val",purpose="evaluate",domain=domain,shape=shape,**args)
        cases=[]
        for i in range(len(dataset)):
            item=dataset[i]
            prediction=infer(student,item["image"][None].to(device))[0].cpu().numpy()
            cases.append(case_metrics(prediction,item["label"].numpy()))
        rows.append(dict(arm=arm,stage=stage,domain=DOMAINS[domain],domain_index=domain,
                         **aggregate(cases)))
    result=dict(status="COMPLETE",student_hash=expected_hash,rows=rows,
        inference_full_models=1,teacher_accesses=0,prototype_accesses=0,optimizer_restored=False,
        test_accesses=0,future_domain_gt_accesses=0)
    write_json(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("data","reference","checkpoint","output"):parser.add_argument("--"+name,required=True)
    args=vars(parser.parse_args())
    evaluate(**args,device=torch.device("cuda:0"))
