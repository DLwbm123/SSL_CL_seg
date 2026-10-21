"""Binary whole-prostate volume metrics at native stored spacing."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage

def binary_phase_metrics(p,y,spacing):
    assert p.shape[1] == 2 and np.isin(y, [0, 1]).all()
    pred=p.argmax(1).numpy();y=np.asarray(y);rows=[]
    assert pred.shape==y.shape and len(spacing)==3 and min(spacing)>0
    penalty=float(np.linalg.norm((np.array(y.shape)-1)*np.array(spacing)))
    for k in (1,):
        a=pred==k;b=y==k;tp=int((a&b).sum());pn=int(a.sum());gn=int(b.sum())
        dice=2*tp/(pn+gn) if pn+gn else None;hd=None
        if pn and gn:
            structure=ndimage.generate_binary_structure(3,1)
            sa=a&~ndimage.binary_erosion(a,structure=structure,border_value=0)
            sb=b&~ndimage.binary_erosion(b,structure=structure,border_value=0)
            da=ndimage.distance_transform_edt(~sb,sampling=spacing)[sa]
            db=ndimage.distance_transform_edt(~sa,sampling=spacing)[sb]
            hd=float(np.percentile(np.r_[da,db],95))
        rows.append(dict(class_index=k,GT=gn,predicted=pn,TP=tp,FP=pn-tp,FN=gn-tp,Dice=dice,
          HD95_mm=hd,HD95_penalized_mm=hd if hd is not None else penalty if gn or pn else None,
          empty_prediction_failure=gn>0 and pn==0,GT_absent=gn==0))
    yt=torch.from_numpy(y.astype(np.int64));nll=0.;brier=0.
    for i in range(len(y)):
        q=p[i].double();target=yt[i]
        nll+=float(-q.clamp_min(1e-30).log().gather(0,target[None]).mean())
        one=F.one_hot(target,2).permute(2,0,1);brier+=float((q-one).square().sum(0).mean())
    return dict(classes=rows,NLL=nll/len(y),Brier=brier/len(y))

