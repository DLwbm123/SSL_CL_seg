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


def segmentation_metrics(pred,target):
    return {name:binary_metrics(p,t) for name,p,t in [
        ('rim',pred==1,target==1),('cup',pred==2,target==2),
        ('disc_union',np.isin(pred,[1,2]),np.isin(target,[1,2]))]}


def load_synthetic_student(path):
    v=torch.load(path,map_location='cpu',weights_only=True)
    if v.get('synthetic_only') is not True:raise ValueError('synthetic evaluator cannot load real checkpoints')
    model=Deployment(SyntheticParentBridge(v['d'],v['rank']),torch.eye(v['d']))
    model.load_state_dict(v['student']);return model.eval().requires_grad_(False)
