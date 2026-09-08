"""Read-only evaluator; private patient rows never enter training or public tables."""
import csv, hashlib, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from experiments.lcrseg.care_hr_v0_7_1.scoring_r1 import case_metrics
from .core import DomainData, batch, fwd, masks, prototypes, noisy, state_hash, config, write_json, tensor_bytes

METRICS=('background_dice','rim_dice','cup_dice','macro_fg_dice')
RATIOS={'precision':('correct','accepted'),'prediction_coverage':('accepted','predicted'),'image_coverage':('accepted','valid_pixels'),'accepted_correct_recall':('correct','gt_pixels'),'joint_valid_coverage':('joint_valid','valid_pixels'),'disagreement_rate':('disagreement','valid_pixels'),'accepted_disagreement_rate':('accepted_disagreement','joint_valid')}

def csv_write(path,rows,fields=None):
    path=Path(path);rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def quality(pred,other,gt,mask):
    valid=gt!=255;m=mask&valid;rows=[]
    for c in range(3):
        counts=dict(class_id=c,predicted=int(((pred==c)&valid).sum()),gt_pixels=int((gt==c).sum()),accepted=int(((pred==c)&m).sum()),correct=int(((pred==c)&(gt==c)&m).sum()),valid_pixels=int(valid.sum()),joint_valid=int(m.sum()),disagreement=int(((pred!=other)&valid).sum()),empty_set=int(not m.any()),accepted_disagreement=int(((pred!=other)&m).sum()))
        for k,(num,den) in RATIOS.items():
            counts[k]=counts[num]/counts[den] if counts[den] else None
            counts[k+'_status']='DEFINED' if counts[den] else 'UNDEFINED_NO_SUPPORT'
        rows.append(counts)
    return rows

def aggregate_quality(rows):
    # First average four repeats within patient, then equal-weight patients. Never n=4N.
    keys=('epoch','domain','arm','seed','mode','library','filter','model','class_id')
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,values in groups.items():
        patients=defaultdict(list)
        for r in values:patients[r['patient']].append(r)
        row=dict(zip(keys,key));row['n_patients']=len(patients);row['n_replicate_rows']=len(values)
        for field in ('predicted','gt_pixels','accepted','correct','valid_pixels','joint_valid','disagreement','empty_set','accepted_disagreement'):
            row['pooled_'+field]=sum(r[field] for r in values)
            row['patient_mean_'+field]=float(np.mean([np.mean([r[field] for r in rs]) for rs in patients.values()]))
        row['pooled_counts_unit']='diagnostic_draw_pixels; four draws are not independent patients'
        for metric,(num,den) in RATIOS.items():
            means=[float(np.mean([r[metric] for r in rs if r[metric] is not None])) for rs in patients.values() if any(r[metric] is not None for r in rs)]
            row['patient_mean_'+metric]=float(np.mean(means)) if means else None
            row['n_patients_defined_'+metric]=len(means)
            row['n_draws_defined_'+metric]=sum(r[metric] is not None for r in values)
            row['pooled_'+metric]=row['pooled_'+num]/row['pooled_'+den] if row['pooled_'+den] else None
            row[metric+'_status']='DEFINED' if row['pooled_'+den] else 'UNDEFINED_NO_SUPPORT'
        out.append(row)
    return out

def rng_hash():
    h=hashlib.sha256();h.update(torch.get_rng_state().numpy().tobytes());h.update(repr(random.getstate()).encode());h.update(repr(np.random.get_state()).encode())
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():h.update(s.cpu().numpy().tobytes())
    return h.hexdigest()

def proto_hash(proto,support):
    if proto is None:return None
    return hashlib.sha256(proto.detach().cpu().numpy().tobytes()+support.detach().cpu().numpy().tobytes()).hexdigest()

@torch.no_grad()
def snapshot(student,teacher,labeled,*,data,reference,device,domain,seed,arm,epoch,output,proto,support,counters,expected=None,shape=(384,384),export_deployment=False):
    root=Path(output);root.mkdir();started=time.time()
    before=(state_hash(student),state_hash(teacher),rng_hash(),proto_hash(proto,support))
    modes=[(m,m.training) for net in (student,teacher) for m in net.modules()]
    student.eval();teacher.eval();raw=[];scores=[];mean_predictions=[]
    counts_before=dict(counters)
    try:
        fresh,fsupport=prototypes(student,labeled,device,counters,seed,epoch,'diagnostic_prototype')
        kw={} if expected is None else dict(expected=expected)
        val=DomainData(data,domain,'val',evaluator=True,shape=shape,**kw)
        # Patient identity remains evaluator-only. Fixed manifest proves one case per patient.
        for i in range(len(val)):
            item=val[i];x=item['image'][None].to(device);gt=item['label'].numpy();geom=item['geometry'][None].to(device)
            for rep in range(5):
                mode='posterior_mean_clean' if rep==0 else 'training_like'
                key=(seed,domain,epoch,i,'diagnostic',rep)
                xs=x if rep==0 else noisy(x,key)
                gas=config(arm)['gas'] and rep>0
                zs,feat_s=fwd(student,xs,gas,key,counters,'diagnostic_student')
                zt,feat_t=fwd(teacher,x,gas,key,counters,'diagnostic_ema')
                ps,pt=zs.softmax(1),zt.softmax(1)
                ys,yt=ps.argmax(1)[0].cpu().numpy(),pt.argmax(1)[0].cpu().numpy()
                if rep==0:
                    for name,pred in [('student',ys),('ema',yt)]:scores.append(dict(patient=i,model=name,**case_metrics(pred,gt)))
                    if export_deployment:mean_predictions.append(ys)
                libraries=[('fresh_diagnostic',fresh,fsupport)]
                if config(arm)['pas'] and proto is not None:libraries.append(('actual_training',proto,support))
                for library,p,s in libraries:
                    conf,pas=masks(ps,pt,feat_s,feat_t,p,s,geom)
                    filters=[('raw',geom),('confidence',conf),('PAS',pas)] if library=='fresh_diagnostic' else [('PAS',pas)]
                    for name,mask in filters:
                        for model,pred,other in [('teacher',yt,ys),('student',ys,yt)]:
                            for row in quality(pred,other,gt,mask[0].cpu().numpy()):
                                raw.append(dict(epoch=epoch,domain=domain,arm=arm,seed=seed,patient=i,repeat=rep,mode=mode,library=library,filter=name,model=model,**row))
        csv_write(root/'private_quality.csv',raw)
        csv_write(root/'quality_aggregate.csv',aggregate_quality(raw))
        csv_write(root/'private_metrics.csv',scores)
        summary=[]
        for name in ('student','ema'):
            allrows=[r for r in scores if r['model']==name];ev=[r for r in allrows if r['has_evaluable_gt']]
            summary.append(dict(epoch=epoch,domain=domain,arm=arm,seed=seed,model=name,n_cases=len(allrows),n_evaluable=len(ev),**{k:float(np.mean([r[k] for r in allrows])) for k in METRICS},**{'evaluable_only_'+k:float(np.mean([r[k] for r in ev])) if ev else None for k in METRICS}))
        csv_write(root/'metrics.csv',summary)
        if export_deployment:np.savez_compressed(root/'evaluator_only_deployment_predictions.npz',predictions=np.stack(mean_predictions))
    finally:
        for m,mode in modes:m.training=mode
    after=(state_hash(student),state_hash(teacher),rng_hash(),proto_hash(proto,support))
    if before!=after:raise RuntimeError('read-only diagnostic changed training state')
    write_json(root/'receipt.json',dict(status='COMPLETE',models=2,state_preserved=True,student_hash=after[0],ema_hash=after[1],rng_hash=after[2],training_proto_hash=after[3],actual_training_library='NOT_STARTED' if config(arm)['pas'] and proto is None else 'AVAILABLE' if proto is not None else 'NOT_USED',val_cases=len(val),val_opens=val.opens,seconds=time.time()-started,counters_delta={k:counters[k]-counts_before.get(k,0) for k in counters},fresh_prototype_bytes=tensor_bytes((fresh,fsupport))))
    # No metric value returns to the training loop.
