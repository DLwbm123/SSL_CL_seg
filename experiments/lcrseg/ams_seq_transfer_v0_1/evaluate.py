"""Fresh process, one final student, only the task's seen-domain validation."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from . import core as c
from .contract import verify,admit,read
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,case_metrics,METRICS
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

@torch.no_grad()
def evaluate(base,task_id,data,reference,device,fixture=None):
    source=verify() if fixture is None else 'SYNTHETIC'
    task=admit(base,task_id,source) if fixture is None else fixture['task']
    root=Path(base)/'tasks'/task_id;out=root/'evaluation';out.mkdir()
    receipt=read(root/'receipt.json')
    if receipt['status']!='TRAINING_COMPLETE' or receipt['source']!=source:raise PermissionError('unfinished trainer')
    payload=torch.load(root/'deploy_student.pt',map_location=device,weights_only=False)
    if payload['task']!=task or payload['head_mode']!=c.LINEAR or payload['source']!=source or not payload['complete']:raise PermissionError('deploy identity')
    model=c.from_state(reference,device,payload.pop('student'),c.LINEAR).eval();del payload
    assert c.state_hash(model)==receipt['student_hash'] and audit_live_models(1)==1
    domains=[task['domain']] if task['source_task_id'] is None else [task['source_domain'],task['domain']]
    kw={} if fixture is None else dict(expected=fixture['expected'],shape=fixture['shape'])
    rows=[];summary=[];started=time.time();count=0
    for domain in domains:
        ds=c.DomainData(data,domain,'val',evaluator=True,**kw);scores=[]
        for i in range(len(ds)):
            item=ds[i];logits,_=model(item['image'][None].to(device),stochastic_classifier=False)
            pred=logits.argmax(1)[0].cpu().numpy();metrics=case_metrics(pred,item['label'].numpy());count+=1
            row=dict(task_id=task_id,domain=domain,patient_index=i,**metrics);scores.append(row);rows.append(row)
        summary.append(dict(task_id=task_id,seed=task['seed'],order=task['order'],arm=task['arm'],domain=domain,role='source' if task['source_task_id'] is None else ('old' if domain==task['source_domain'] else 'current'),n_patients=len(scores),n_evaluable=sum(r['has_evaluable_gt'] for r in scores),**{k:float(np.mean([r[k] for r in scores])) for k in METRICS}))
    csv_write(out/'private_patient_metrics.csv',rows);csv_write(out/'metrics.csv',summary)
    result=dict(status='COMPLETE',source=source,task_id=task_id,student_hash=receipt['student_hash'],models=1,seen_domains=domains,patient_forwards=count,GT_samples=count,optimizer_updates=0,backward=0,training_model_released_before_eval=True,seconds=time.time()-started)
    c.write_json(out/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','task_id','data','reference'):p.add_argument('--'+k.replace('_','-'),required=True)
    a=vars(p.parse_args());root=Path(a['base'])/'tasks'/a['task_id']
    with Operations(root.parent/(root.name+'_eval_operations')):evaluate(**a,device=torch.device('cuda:0'))
