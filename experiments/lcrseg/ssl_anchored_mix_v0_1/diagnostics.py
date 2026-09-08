"""Original-image validation only; no counterfactual gradient grids or mixed-image scoring."""
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from .core import *
from .telemetry import flush
from experiments.lcrseg.ssl_head_control_v0_1.diagnostics import image_only
from experiments.lcrseg.ssl_head_control_v0_1.probability import probability_rows,correctness_strata
from experiments.lcrseg.ssl_head_control_v0_1.audit import aggregate
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,quality,aggregate_quality,rng_hash,METRICS,case_metrics

@torch.no_grad()
def snapshot(student,teacher,labeled,*,data,reference,device,domain,seed,arm,epoch,output,proto,support,counters,expected=None,shape=(384,384),export_deployment=False,optimizer=None):
    if proto is not None or support is not None:raise PermissionError('no prototypes')
    root=Path(output);root.mkdir();start=time.time();prior=dict(counters)
    before=(state_hash(student),state_hash(teacher),rng_hash(),optimizer_hash(optimizer))
    modes=[(m,m.training) for n in (student,teacher) for m in n.modules()];student.eval();teacher.eval()
    scores=[];probs=[];qualities=[];strata=[];seals=[];predictions=[]
    try:
        val=DomainData(data,domain,'val',evaluator=True,shape=shape,**({} if expected is None else dict(expected=expected)));images=image_only(val)
        for i in range(len(images)):
            item=images[i];x=item['image'][None].to(device);geo=item['geometry'][None].to(device);pending=[]
            for rep in range(5):
                key=(seed,domain,epoch,i,'diagnostic',rep)
                z,_=fwd(student,x if rep==0 else noisy(x,key),False,key,counters,'diagnostic_student')
                zt,_=fwd(teacher,x,False,key,counters,'diagnostic_ema');p=z.softmax(1);q=zt.softmax(1);mask=geo&(q.max(1)[0]>.7)
                h=hashlib.sha256()
                for a in (z,q,geo,mask):h.update(a.cpu().numpy().tobytes())
                seals.append(dict(patient=i,repeat=rep,sha256=h.hexdigest(),GT_accessed=False));pending.append((rep,z,zt,p,q,mask))
            gt=val[i]['label'].numpy()
            for rep,z,zt,p,q,mask in pending:
                identity=dict(epoch=epoch,domain=domain,arm=arm,seed=seed,patient=i,repeat=rep,permutation=0,mode='posterior_mean_clean' if rep==0 else 'training_like')
                ys=p.argmax(1)[0].cpu().numpy();yt=q.argmax(1)[0].cpu().numpy()
                if rep==0:
                    for name,pred in [('student',ys),('ema',yt)]:scores.append(dict(patient=i,model=name,**case_metrics(pred,gt)))
                    if export_deployment:predictions.append(ys)
                for model,zv,pv,pred,other in [('student',z,p,ys,yt),('teacher',zt,q,yt,ys)]:
                    pr,_=probability_rows(zv,pv,gt);probs.append(dict(**identity,model=model,**pr))
                    for filter_,m in [('raw',geo),('teacher',mask)]:
                        qualities.extend(dict(**identity,library='none',filter=filter_,model=model,**r) for r in quality(pred,other,gt,m[0].cpu().numpy()))
                strata.append(dict(**identity,**correctness_strata(p,q,gt,mask)))
            del pending
            flush('diagnostic_patient_complete')
        csv_write(root/'private_metrics.csv',scores)
        summary=[]
        for name in ('student','ema'):
            rr=[r for r in scores if r['model']==name];ev=[r for r in rr if r['has_evaluable_gt']]
            summary.append(dict(epoch=epoch,domain=domain,arm=arm,seed=seed,model=name,head_mode=LINEAR,n_cases=len(rr),n_evaluable=len(ev),**{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
        csv_write(root/'metrics.csv',summary)
        for name,rr,keys in [('probability',probs,('epoch','domain','arm','seed','mode','model')),('strata',strata,('epoch','domain','arm','seed','mode'))]:
            csv_write(root/f'{name}_aggregate.csv',aggregate(rr,keys))
        csv_write(root/'quality_aggregate.csv',aggregate_quality(qualities));write_json(root/'private_prediction_seals.json',seals)
        if export_deployment:np.savez_compressed(root/'evaluator_only_deployment_predictions.npz',predictions=np.stack(predictions))
    finally:
        for m,mode in modes:m.training=mode
    after=(state_hash(student),state_hash(teacher),rng_hash(),optimizer_hash(optimizer));assert before==after
    write_json(root/'receipt.json',dict(status='COMPLETE',state_preserved=True,student_hash=after[0],ema_hash=after[1],rng_hash=after[2],optimizer_hash=after[3],head_mode=LINEAR,models=2,val_cases=len(val),val_opens=val.opens,image_only_opens=images.opens,prediction_seals=len(seals),optimizer_updates=0,additional_backward=0,counters_delta={k:counters[k]-prior.get(k,0) for k in counters},seconds=time.time()-start))
    flush('diagnostic_complete')
