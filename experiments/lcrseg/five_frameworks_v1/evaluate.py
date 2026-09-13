"""Student-only synthetic deployment evaluator and explicit segmentation metrics."""
import numpy as np
from scipy import ndimage
import torch
from .parent_bridge import SyntheticParentBridge
from .model import Deployment


def binary_metrics(pred,target,spacing=None):
    p=np.asarray(pred,dtype=bool);t=np.asarray(target,dtype=bool)
    if p.shape!=t.shape:raise ValueError('metric shape mismatch')
    npix,nt=p.sum(),t.sum();both_empty=(npix+nt)==0
    dice=1. if both_empty else 2*(p&t).sum()/(npix+nt)
    if both_empty:hd=assd=0.
    elif not npix or not nt:hd=assd=float('inf')
    else:
        ps=p&~ndimage.binary_erosion(p);ts=t&~ndimage.binary_erosion(t)
        a=ndimage.distance_transform_edt(~ts,sampling=spacing)[ps]
        b=ndimage.distance_transform_edt(~ps,sampling=spacing)[ts]
        hd=float(np.percentile(np.concatenate((a,b)),95));assd=float((a.sum()+b.sum())/(len(a)+len(b)))
    def topology(mask):
        components=ndimage.label(mask)[1]
        holes=ndimage.label(ndimage.binary_fill_holes(mask)&~mask)[1]
        return components,holes
    pc,ph=topology(p);tc,th=topology(t)
    return {'Dice':float(dice),'HD95':hd,'ASSD':assd,'distance_unit':'pixels' if spacing is None else 'supplied_spacing_unit',
            'empty_case':'both' if both_empty else ('one' if not npix or not nt else 'neither'),
            'pred_components':pc,'target_components':tc,'pred_holes':ph,'target_holes':th,
            'area_absolute_error_pixels':int(abs(int(npix)-int(nt)))}


def segmentation_metrics(pred,target,valid=None):
    pred,target=np.asarray(pred),np.asarray(target)
    if pred.shape!=target.shape or pred.ndim!=2:raise ValueError('expected equal 2D label maps')
    if not np.isin(target,[0,1,2,255]).all():raise ValueError('unknown target class')
    if not np.isin(pred,[0,1,2]).all():raise ValueError('unknown prediction class')
    if valid is not None and (np.asarray(valid).shape!=target.shape or np.asarray(valid).dtype!=bool):
        raise ValueError('valid mask must be boolean and match labels')
    support=(target!=255) if valid is None else ((target!=255)&valid)
    result={}
    for name,classes in [('rim',[1]),('cup',[2]),('disc_union',[1,2])]:
        p=np.isin(pred,classes)&support;t=np.isin(target,classes)&support
        if support.all():
            result[name]=binary_metrics(p,t)
        else:
            denom=p.sum()+t.sum()
            result[name]={'Dice':None if not support.any() else (1. if denom==0 else float(2*(p&t).sum()/denom)),
                          'HD95':None,'ASSD':None,'pred_components':None,'target_components':None,
                          'pred_holes':None,'target_holes':None,'area_absolute_error_pixels':None,
                          'distance_unit':'pixels','auxiliary_status':'NOT_EVALUABLE_IGNORE_REQUIRES_NATIVE_EVALUATOR',
                          'empty_case':'NO_VALID_SUPPORT' if not support.any() else ('both' if denom==0 else ('one' if not p.any() or not t.any() else 'neither'))}
        result[name]['valid_pixels']=int(support.sum())
    return result


def load_synthetic_student(path):
    v=torch.load(path,map_location='cpu',weights_only=True)
    if v.get('synthetic_only') is not True:raise ValueError('synthetic evaluator cannot load real checkpoints')
    model=Deployment(SyntheticParentBridge(v['d'],v['rank']),torch.eye(v['d']))
    model.load_state_dict(v['student']);return model.eval().requires_grad_(False)
