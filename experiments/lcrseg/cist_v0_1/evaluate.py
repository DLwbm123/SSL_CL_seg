"""Fixed final student evaluation; aggregate mechanisms without saving patient Q/b."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,case_metrics,METRICS
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from . import core as c,contract as ct

@torch.no_grad()
def evaluate(base,task_id,data,reference,device):
    source=ct.verify();task=ct.admit(base,task_id,source);b=Path(base);root=b/'tasks'/task_id;r=ct.read(root/'receipt.json')
    if r['source']!=source or r['task']!=task or r['status']!='TRAINING_COMPLETE':raise PermissionError('unfinished task')
    if str(Path(data).resolve())!=ct.read(b/'private_inputs.json')['data']:raise PermissionError('data identity')
    p=torch.load(root/'deploy_student.pt',map_location=device,weights_only=False);tensors,entry=ct.load_basis(b,task)
    if p['source']!=source or p['task']!=task or not p['complete'] or p['head_mode']!=c.LINEAR:raise PermissionError('deploy identity')
    core=c.from_state(reference,device,p.pop('core'),c.LINEAR);tr=c.Transport(tensors,task['arm'],task['seed'],task['source_domain']).to(device);tr.load_state_dict(p.pop('controller'))
    model=c.Student(core,tr).eval();assert c.state_hash(model)==p['student_hash']==r['student_hash'] and c.state_hash(core)==entry['student_hash'];del p
    assert audit_live_models(1)==1
    out=root/'evaluation';out.mkdir();rows=[];summary=[];mechanisms=[];started=time.time()
    for domain in (task['source_domain'],task['domain']):
        ds=c.DomainData(data,domain,'val',evaluator=True);scores=[];qsum=torch.zeros(8,8,dtype=torch.float64);bsum=torch.zeros(8,dtype=torch.float64);q2=b2=0.;stats=[]
        cls_before=np.zeros(3,dtype=np.int64);cls_after=np.zeros(3,dtype=np.int64);conf_before=conf_after=pixels=0
        for i in range(len(ds)):
            item=ds[i];logits,(h,hp,detail)=model(item['image'][None].to(device));pred=logits.argmax(1)[0].cpu().numpy()
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite evaluation logits')
            metrics=case_metrics(pred,item['label'].numpy());row=dict(task_id=task_id,domain=domain,patient_index=i,**metrics);scores.append(row);rows.append(row)
            stats.append(c.mechanism(h,hp,detail));q=detail['Q'].double().cpu()[0];bias=detail['b'].double().cpu()[0];qsum+=q;bsum+=bias;q2+=float(q.square().sum());b2+=float(bias.square().sum())
            # One extra readout-only diagnostic; deployment itself uses one original image/core/controller/head.
            original=model.readout(h,item['image'].shape[-2:]);c.count('diagnostic_readout_forward')
            for z,target in [(original,cls_before),(logits,cls_after)]:
                pp=z.softmax(1);valid=item['geometry'].to(device);target+=torch.bincount(pp.argmax(1)[0][valid],minlength=3).cpu().numpy()
            mask=item['geometry'].to(device);conf_before+=float(original.softmax(1).max(1).values[0][mask].sum());conf_after+=float(logits.softmax(1).max(1).values[0][mask].sum());pixels+=int(mask.sum())
        n=len(scores);summary.append(dict(task_id=task_id,seed=task['seed'],order=task['order'],arm=task['arm'],domain=domain,role='old' if domain==task['source_domain'] else 'current',n_patients=n,n_evaluable=sum(x['has_evaluable_gt'] for x in scores),**{k:float(np.mean([x[k] for x in scores])) for k in METRICS}))
        mechanisms.append(dict(task_id=task_id,domain=domain,images=n,Q_between_image_variance=max(0.,q2/n-float((qsum/n).square().sum())),b_between_image_variance=max(0.,b2/n-float((bsum/n).square().sum())),
                               distance_relative_max=max(x['distance_relative_max'] for x in stats),distance_relative_mean=float(np.mean([x['distance_relative_mean'] for x in stats])),
                               Q_norm_mean=float(np.mean([x['Q_norm'] for x in stats])),b_norm_mean=float(np.mean([x['b_norm'] for x in stats])),
                               prediction_fraction_source=(cls_before/pixels).tolist(),prediction_fraction_adapted=(cls_after/pixels).tolist(),confidence_source=conf_before/pixels,confidence_adapted=conf_after/pixels))
    csv_write(out/'private_patient_metrics.csv',rows);csv_write(out/'metrics.csv',summary);c.write_json(out/'MECHANISMS.json',mechanisms)
    result=dict(status='COMPLETE',source=source,task_id=task_id,student_hash=r['student_hash'],models=1,seen_domains=[task['source_domain'],task['domain']],
                patient_forwards=len(rows),GT_samples=len(rows),optimizer_updates=0,backward=0,EMA_models=0,normal_deploy_core_forwards_per_image=1,normal_deploy_controller_forwards_per_image=1,
                diagnostic_extra_readout_forwards=len(rows),per_patient_Qb_saved=False,seconds=time.time()-started)
    c.write_json(out/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','task_id','data','reference'):p.add_argument('--'+n.replace('_','-'),required=True)
    a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id']
    with Operations(root.parent/(root.name+'_eval_operations')):evaluate(**a,device=torch.device('cuda:0'))
