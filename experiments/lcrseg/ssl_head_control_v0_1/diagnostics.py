"""All masks/draws sealed before per-case GT access; no metric feeds training."""
import time
from pathlib import Path
import numpy as np
import torch
from .core import *
from .audit import matched_masks,evaluate_mass,aggregate
from .probability import probability_rows,correctness_strata
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,quality,aggregate_quality,rng_hash,proto_hash,METRICS,case_metrics

def image_only(ds):
    clone=copy.copy(ds);clone._dataset=copy.copy(ds._dataset);clone._dataset.role='train_unlabeled'
    clone._dataset.rows=[{k:v for k,v in r.items() if k in ('case_id','image_h5_relpath','image_sha256')} for r in ds._dataset.rows]
    clone._dataset.checked=set();clone.opens=0;return clone

@torch.no_grad()
def snapshot(student,teacher,labeled,*,data,reference,device,domain,seed,arm,epoch,output,proto,support,counters,expected=None,shape=(384,384),export_deployment=False,audit_only=False):
    root=Path(output);root.mkdir();started=time.time();before=(state_hash(student),state_hash(teacher),rng_hash(),proto_hash(proto,support));counts_before=dict(counters)
    modes=[(m,m.training) for net in (student,teacher) for m in net.modules()];student.eval();teacher.eval()
    raw=[];scores=[];prob_rows=[];bin_rows=[];mass=[];grad=[];strata=[];seals=[];predictions=[]
    try:
        if audit_only:
            if proto is None or support is None:raise ValueError('actual prototype required for P0')
            libraries=[('actual_training',proto,support)];spec=base.config(arm)
        else:
            spec=config(arm);fresh,fsupport=prototypes(student,labeled,device,counters,seed,epoch,'diagnostic_prototype')
            libraries=[('fresh_diagnostic',fresh,fsupport)]
            if spec['pas'] and proto is not None:libraries.append(('actual_training',proto,support))
        val=DomainData(data,domain,'val',evaluator=True,shape=shape,**({} if expected is None else dict(expected=expected)));images=image_only(val)
        for i in range(len(images)):
            item=images[i];x=item['image'][None].to(device);geom=item['geometry'][None].to(device);pending=[]
            for rep in range(5):
                mode='posterior_mean_clean' if rep==0 else 'training_like';key=(seed,domain,epoch,i,'diagnostic',rep)
                zs,fs=fwd(student,x if rep==0 else noisy(x,key),spec['gas'] and rep>0,key,counters,'diagnostic_student')
                zt,ft=fwd(teacher,x,spec['gas'] and rep>0,key,counters,'diagnostic_ema')
                ps,pt=zs.softmax(1),zt.softmax(1);ys,yt=ps.argmax(1)[0].cpu().numpy(),pt.argmax(1)[0].cpu().numpy();views=[]
                for library,p,s in libraries:
                    conf,pas=masks(ps,pt,fs,ft,p,s,geom)
                    matched=None
                    if library=='actual_training' and (audit_only or spec['pas']):
                        matched=matched_masks(conf[0].cpu().numpy(),pas[0].cpu().numpy(),ps.max(1)[0][0].cpu().numpy(),pt.max(1)[0][0].cpu().numpy(),yt,key)
                        seals.append(dict(patient=i,repeat=rep,library=library,mask_sha256=matched[2],K=matched[1],GT_accessed_before_seal=False))
                    views.append((library,conf,pas,matched))
                pending.append((rep,mode,zs,zt,ps,pt,ys,yt,views))
            # Only after all five prediction/mask payloads exist does GT enter this evaluator.
            gt=val[i]['label'].numpy()
            for rep,mode,zs,zt,ps,pt,ys,yt,views in pending:
                identity=dict(epoch=epoch,domain=domain,arm=arm,seed=seed,patient=i,repeat=rep,permutation=0,mode=mode)
                if rep==0:
                    for model,pred in [('student',ys),('ema',yt)]:scores.append(dict(patient=i,model=model,**case_metrics(pred,gt)))
                    if export_deployment:predictions.append(ys)
                if not audit_only:
                    for model,z,p in [('student',zs,ps),('teacher',zt,pt)]:
                        values,bins=probability_rows(z,p,gt);prob_rows.append(dict(**identity,model=model,**values))
                        bin_rows.extend(dict(**identity,model=model,**v) for v in bins)
                for library,conf,pas,matched in views:
                    for filter_name,mask in ([('raw',geom),('confidence',conf),('PAS',pas)] if library=='fresh_diagnostic' else [('PAS',pas)]):
                        for model,pred,other in [('teacher',yt,ys),('student',ys,yt)]:
                            raw.extend(dict(**identity,library=library,filter=filter_name,model=model,**r) for r in quality(pred,other,gt,mask[0].cpu().numpy()))
                        if not audit_only:
                            grad.extend(dict(**identity,library=library,filter=filter_name,**r) for r in class_loss_stats(ps,pt,mask,weight(epoch)))
                            strata.append(dict(**identity,library=library,filter=filter_name,**correctness_strata(ps,pt,gt,mask)))
                    if matched is not None:
                        for r in evaluate_mass(*matched,conf[0].cpu().numpy(),ys,yt,gt):
                            ident=dict(identity);ident.pop('permutation');mass.append(dict(**ident,library=library,**r))
            del pending
        csv_write(root/'private_quality.csv',raw);csv_write(root/'quality_aggregate.csv',aggregate_quality(raw));csv_write(root/'private_metrics.csv',scores)
        summary=[]
        for model in ('student','ema'):
            rr=[r for r in scores if r['model']==model];ev=[r for r in rr if r['has_evaluable_gt']]
            summary.append(dict(epoch=epoch,domain=domain,arm=arm,seed=seed,head_mode=getattr(student,'head_mode',NORMAL),model=model,n_cases=len(rr),n_evaluable=len(ev),**{k:float(np.mean([r[k] for r in rr])) for k in METRICS},**{'evaluable_only_'+k:float(np.mean([r[k] for r in ev])) if ev else None for k in METRICS}))
        csv_write(root/'metrics.csv',summary)
        for name,records,keys in [
            ('mass',mass,('epoch','domain','arm','seed','mode','library','method','model','class_id')),
            ('probability',prob_rows,('epoch','domain','arm','seed','mode','model')),
            ('reliability',bin_rows,('epoch','domain','arm','seed','mode','model','class_id','bin')),
            ('gradient_logit',grad,('epoch','domain','arm','seed','mode','library','filter','class_id')),
            ('correctness_strata',strata,('epoch','domain','arm','seed','mode','library','filter'))]:
            if records:csv_write(root/f'private_{name}.csv',records);csv_write(root/f'{name}_aggregate.csv',aggregate(records,keys))
        write_json(root/'private_mask_seals.json',seals)
        if export_deployment:np.savez_compressed(root/'evaluator_only_deployment_predictions.npz',predictions=np.stack(predictions))
    finally:
        for m,mode in modes:m.training=mode
    after=(state_hash(student),state_hash(teacher),rng_hash(),proto_hash(proto,support))
    if before!=after:raise RuntimeError('read-only diagnostic mutated training state')
    delta={k:counters[k]-counts_before.get(k,0) for k in counters}
    write_json(root/'receipt.json',dict(status='COMPLETE',models=2,state_preserved=True,student_hash=after[0],ema_hash=after[1],rng_hash=after[2],training_proto_hash=after[3],actual_training_library='NOT_STARTED' if spec['pas'] and proto is None else 'AVAILABLE' if proto is not None else 'NOT_USED',val_cases=len(val),val_opens=val.opens,image_only_opens=images.opens,seconds=time.time()-started,counters_delta=delta,GT_free_mask_seals=len(seals),matched_mass_completed=bool(mass),P0=audit_only,sample_model_forwards=delta.get('diagnostic_student_images',0)+delta.get('diagnostic_ema_images',0),optimizer_updates=0,additional_backward=0))
