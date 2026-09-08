"""Evaluator-only probability scale/calibration and analytic local logit gradients."""
import numpy as np
import torch
from .core import class_loss_stats,weight
QUANTILES=(0.,.25,.5,.75,.9,.99,1.)

def probability_rows(logits,prob,truth):
    z=logits[0].detach().cpu().numpy();p=prob[0].detach().cpu().numpy();valid=truth!=255;pred=p.argmax(0);confidence=p.max(0)
    top=np.sort(z,axis=0)[-2:];stats={'logit':z[:,valid].ravel(),'logit_range':(z.max(0)-z.min(0))[valid],'margin':(top[1]-top[0])[valid],'pmax':confidence[valid]}
    n=int(valid.sum());row=dict(valid_pixels=n,pmax_above_099=int(((confidence>.99)&valid).sum()))
    for name,value in stats.items():
        for q in QUANTILES:row[f'{name}_q{int(q*100):02d}']=float(np.quantile(value,q)) if value.size else None
    row['fraction_pmax_above_099']=row['pmax_above_099']/n if n else None
    t=truth[valid].astype(int);pp=p[:,valid];zz=z[:,valid]
    if n:
        # Log-softmax from logits avoids assigning finite NLL to underflowed probability.
        maxima=zz.max(0);lse=maxima+np.log(np.exp(zz-maxima).sum(0));row['NLL']=float(np.mean(lse-zz[t,np.arange(n)]))
        onehot=np.eye(3)[t].T;row['multiclass_Brier']=float(np.mean(((pp-onehot)**2).sum(0)))
    else:row.update(NLL=None,multiclass_Brier=None)
    fgerr=(pred>0)&(pred!=truth)&valid;row['wrong_foreground_count']=int(fgerr.sum());row['wrong_foreground_confidence']=float(confidence[fgerr].mean()) if fgerr.any() else None
    bins=[]
    for cls in range(3):
        for b in range(10):
            m=valid&(pred==cls)&(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else (confidence<=1));num=int(m.sum());correct=int(((pred==truth)&m).sum());cs=float(confidence[m].astype(float).sum())
            bins.append(dict(class_id=cls,bin=b,lower=b/10,upper=(b+1)/10,bin_count=num,confidence_sum=cs,correct_count=correct,mean_confidence=cs/num if num else None,accuracy=correct/num if num else None))
    row['ECE_10_predicted_class_bins']=sum(abs(x['confidence_sum']-x['correct_count']) for x in bins)/n if n else None
    return row,bins

def correctness_strata(ps,pt,truth,mask):
    ys=ps.argmax(1)[0].cpu().numpy();yt=pt.argmax(1)[0].cpu().numpy();v=truth!=255;m=mask[0].cpu().numpy()&v
    return dict(both_correct=int((m&(ys==truth)&(yt==truth)).sum()),student_only_correct=int((m&(ys==truth)&(yt!=truth)).sum()),teacher_only_correct=int((m&(ys!=truth)&(yt==truth)).sum()),both_wrong_same=int((m&(ys!=truth)&(yt!=truth)&(ys==yt)).sum()),both_wrong_different=int((m&(ys!=truth)&(yt!=truth)&(ys!=yt)).sum()),evaluable_accepted=int(m.sum()))
