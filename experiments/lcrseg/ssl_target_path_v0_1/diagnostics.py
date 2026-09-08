"""Prediction-only sealing precedes GT. Analytic diagnostics never call backward."""
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from .core import *
from experiments.lcrseg.ssl_head_control_v0_1.diagnostics import image_only
from experiments.lcrseg.ssl_head_control_v0_1.probability import probability_rows
from experiments.lcrseg.ssl_head_control_v0_1.audit import aggregate
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import quality,aggregate_quality,rng_hash,csv_write,case_metrics,METRICS
NEAR_L1=1e-6

def seal_prediction(z,q,geometry):
    p,q,j,t,values=objectives(z,q,geometry)
    h=hashlib.sha256()
    for x in (z,q,geometry,j,t):h.update(x.detach().cpu().numpy().tobytes())
    return p,q,j,t,h.hexdigest()

def path_rows(p,q,g,j,t,gt,lam):
    # FP64 scalar accumulation of already-fixed FP32 probabilities; no new inference.
    p=p[0].detach().cpu().numpy().astype(float);q=q[0].detach().cpu().numpy().astype(float)
    g,j,t=(x[0].cpu().numpy() for x in (g,j,t));v=gt!=255;ys=p.argmax(0);yt=q.argmax(0);delta=p-q;distance=np.abs(delta).sum(0)
    strata=dict(both_correct=(ys==gt)&(yt==gt),teacher_correct_student_wrong=(yt==gt)&(ys!=gt),teacher_wrong_student_correct=(yt!=gt)&(ys==gt),both_wrong_same_class=(ys!=gt)&(yt!=gt)&(ys==yt),both_wrong_different_class=(ys!=gt)&(yt!=gt)&(ys!=yt))
    gtg=p.copy()
    for c in range(3):gtg[c]-=(gt==c)
    gg=np.sqrt((gtg**2).sum(0));raws=dict(MSE=2*p*(delta-(p*delta).sum(0)),SCE=delta);paths=[];grads=[]
    for grouping,labels in [('teacher_predicted_class',yt),('true_class',gt)]:
        for cls in range(3):
            for name,st in strata.items():
                support=v&g&(labels==cls)&st;raw_n=int(support.sum());tn=int((support&t).sum());jn=int((support&j).sum())
                paths.append(dict(grouping=grouping,class_id=cls,stratum=name,raw_support=raw_n,T_accepted=tn,J_accepted=jn,teacher_rejected=raw_n-tn,student_further_rejected=tn-jn,teacher_reject_rate=(raw_n-tn)/raw_n if raw_n else None,student_further_reject_rate=(tn-jn)/tn if tn else None,T_retention=tn/raw_n if raw_n else None,J_retention=jn/raw_n if raw_n else None,same_argmax_different_distribution=int((support&(ys==yt)&(distance>NEAR_L1)).sum()),near_equal_distribution=int((support&(distance<=NEAR_L1)).sum()),probability_L1_mean=float(distance[support].mean()) if raw_n else None))
                for candidate in ARMS[1:]:
                    mask=j if candidate[0]=='J' else t;loss=candidate.split('_')[1];raw=raws[loss];selected=support&mask;n=int(selected.sum());den=max(int(mask.sum()),1);weighted=raw*mask[None]*lam/den
                    norm=np.sqrt((raw**2).sum(0));dot=(raw*gtg).sum(0);valid_cos=selected&(norm>0)&(gg>0)
                    grads.append(dict(grouping=grouping,class_id=cls,stratum=name,candidate=candidate,accepted=n,geometry_loss_denominator=den,lambda_cons=lam,raw_pixel_logit_norm_mean=float(norm[selected].mean()) if n else None,weighted_local_logit_norm=float(np.sqrt((weighted[:,support]**2).sum())),raw_dot_GT_mean=float(dot[selected].mean()) if n else None,weighted_dot_GT_sum=float((weighted*gtg).sum(0)[support].sum()),raw_cosine_GT_mean=float((dot[valid_cos]/(norm[valid_cos]*gg[valid_cos])).mean()) if valid_cos.any() else None,weighted_cosine_GT_mean=float((dot[valid_cos]/(norm[valid_cos]*gg[valid_cos])).mean()) if valid_cos.any() and lam>0 else None,raw_cosine_defined=int(valid_cos.sum()),weighted_cosine_defined=int(valid_cos.sum()) if lam>0 else 0,direction_conflict_count=int((selected&(dot<0)).sum())))
    return paths,grads

@torch.no_grad()
def snapshot(student,teacher,labeled,*,data,reference,device,domain,seed,arm,epoch,output,proto,support,counters,expected=None,shape=(384,384),export_deployment=False,optimizer=None):
    if proto is not None or support is not None:raise PermissionError('no prototype diagnostics')
    root=Path(output);root.mkdir();start=time.time();prior=dict(counters)
    before=(state_hash(student),state_hash(teacher),rng_hash(),student.head_mode,optimizer_hash(optimizer));modes=[(m,m.training) for net in (student,teacher) for m in net.modules()];student.eval();teacher.eval()
    scores=[];path=[];grad=[];prob=[];qual=[];seals=[];predictions=[]
    try:
        val=DomainData(data,domain,'val',evaluator=True,shape=shape,**({} if expected is None else dict(expected=expected)));images=image_only(val)
        for i in range(len(images)):
            item=images[i];x=item['image'][None].to(device);geo=item['geometry'][None].to(device);pending=[]
            for rep in range(5):
                key=(seed,domain,epoch,i,'diagnostic',rep)
                z,_=fwd(student,x if rep==0 else noisy(x,key),False,key,counters,'diagnostic_student');zt,_=fwd(teacher,x,False,key,counters,'diagnostic_ema')
                p,q,j,t,seal=seal_prediction(z,zt.softmax(1).detach(),geo);pending.append((rep,z,zt,p,q,j,t));seals.append(dict(patient=i,repeat=rep,sha256=seal,GT_accessed=False))
            gt=val[i]['label'].numpy()
            for rep,z,zt,p,q,j,t in pending:
                identity=dict(epoch=epoch,domain=domain,arm=arm,seed=seed,patient=i,repeat=rep,permutation=0,mode='posterior_mean_clean' if rep==0 else 'training_like')
                pp,qq=p.argmax(1)[0].cpu().numpy(),q.argmax(1)[0].cpu().numpy()
                if rep==0:
                    for model,pred in [('student',pp),('ema',qq)]:scores.append(dict(patient=i,model=model,**case_metrics(pred,gt)))
                    if export_deployment:predictions.append(pp)
                pr,gr=path_rows(p,q,geo,j,t,gt,weight(epoch));path.extend(dict(**identity,**r) for r in pr);grad.extend(dict(**identity,**r) for r in gr)
                for model,zz,ps,pred,other in [('student',z,p,pp,qq),('teacher',zt,q,qq,pp)]:
                    values,_=probability_rows(zz,ps,gt);prob.append(dict(**identity,model=model,**values))
                    for name,mask in [('raw',geo),('teacher',t),('joint',j)]:qual.extend(dict(**identity,library='none',filter=name,model=model,**r) for r in quality(pred,other,gt,mask[0].cpu().numpy()))
            del pending
        csv_write(root/'private_metrics.csv',scores)
        summary=[]
        for model in ('student','ema'):
            rr=[r for r in scores if r['model']==model];ev=[r for r in rr if r['has_evaluable_gt']]
            summary.append(dict(epoch=epoch,domain=domain,arm=arm,seed=seed,head_mode=LINEAR,model=model,n_cases=len(rr),n_evaluable=len(ev),**{k:float(np.mean([r[k] for r in rr])) for k in METRICS},**{'evaluable_only_'+k:float(np.mean([r[k] for r in ev])) if ev else None for k in METRICS}))
        csv_write(root/'metrics.csv',summary)
        for name,records,keys in [('correction',path,('epoch','domain','arm','seed','mode','grouping','class_id','stratum')),('gradient',grad,('epoch','domain','arm','seed','mode','grouping','class_id','stratum','candidate')),('probability',prob,('epoch','domain','arm','seed','mode','model'))]:
            csv_write(root/f'private_{name}.csv',records);csv_write(root/f'{name}_aggregate.csv',aggregate(records,keys))
        csv_write(root/'quality_aggregate.csv',aggregate_quality(qual));write_json(root/'private_prediction_seals.json',seals)
        if export_deployment:np.savez_compressed(root/'evaluator_only_deployment_predictions.npz',predictions=np.stack(predictions))
    finally:
        for m,mode in modes:m.training=mode
    after=(state_hash(student),state_hash(teacher),rng_hash(),student.head_mode,optimizer_hash(optimizer))
    assert before==after
    write_json(root/'receipt.json',dict(status='COMPLETE',state_preserved=True,optimizer_hash=after[4],head_mode=LINEAR,student_hash=after[0],ema_hash=after[1],rng_hash=after[2],val_cases=len(val),val_opens=val.opens,image_only_opens=images.opens,prediction_seals=len(seals),models=2,optimizer_updates=0,additional_backward=0,counters_delta={k:counters[k]-prior.get(k,0) for k in counters},seconds=time.time()-start))
