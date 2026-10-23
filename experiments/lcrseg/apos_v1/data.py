"""Only source train-L before adaptation; target process enforces a target-L allowlist."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from ..main_style_ssl_v1.data import Accessor
from ..di_dmpa_gate1.binding import safe_asset
from ..five_frameworks_v1.recipes import generator
from ..main_style_ssl_v1.runtime import write


def fields(y):
    values=[];patterns=[]
    for item in y.detach().cpu().numpy():
        present=[bool((item==c).any()) for c in (1,2,3)]
        patterns.append(sum(int(v)<<i for i,v in enumerate(present)))
        channels=[]
        for c,exists in zip((1,2,3),present):
            mask=item==c
            a=np.clip((distance_transform_edt(~mask)-distance_transform_edt(mask))/48,-1,1) if exists else np.ones_like(item,dtype=np.float32)
            channels.append(a.astype(np.float32))
        values.append(np.stack(channels))
    t=F.interpolate(torch.from_numpy(np.stack(values)),size=(32,32),mode='bilinear',align_corners=False)
    return t.to(y.device),torch.tensor(patterns,device=y.device)


def allow_only(access):
    import h5py
    allowed={str(safe_asset(access.data,r[k])).strip() for r in access.rows for k in ('image_h5_relpath','label_h5_relpath')}
    original=h5py.File
    def checked(path,*args,**kwargs):
        if str(Path(path).resolve()) not in allowed:raise PermissionError('training HDF5 outside current train-L')
        return original(path,*args,**kwargs)
    h5py.File=checked

class Current:
    def __init__(self,cfg):
        self.cfg=cfg;self.arm=cfg['arm'];self.l_reads=self.u_reads=0
        self.access=Accessor(cfg['data'],'mnms','Siemens' if cfg['stage']=='source' else 'Philips','train_labeled')
        assert len(self.access.patients)==(10 if cfg['stage']=='source' else 13)
        allow_only(self.access)
    def labeled(self,step):
        self.l_reads+=1;g=generator(20271601,1,0 if self.cfg['stage']=='source' else 1,step,'APOS_L')
        selected=torch.randperm(len(self.access.patients),generator=g)[:2].tolist();xx=[];yy=[];ids=[]
        for pos in selected:
            patient=self.access.patients[pos];ii=self.access.by_patient[patient]
            index=ii[int(torch.randint(len(ii),(),generator=g))];z=int(torch.randint(self.access.depths[index],(),generator=g))
            item=self.access.read(index,z);xx.append(item['image']);yy.append(item['label']);ids.append(patient)
        return torch.stack(xx).cuda(),torch.stack(yy).cuda(),tuple(ids)


def fit_prior(cfg):
    from .runtime import guard,save
    access=Accessor(cfg['data'],'mnms','Siemens','train_labeled');allow_only(access)
    assert len(access.patients)==10 and sum(access.depths)==230
    matrices=[];patterns=[]
    for i,depth in enumerate(access.depths):
        guard(cfg)
        item=access.read(i)
        t,p=fields(item['label']);matrices.append(t.flatten(1).double());patterns.append(p)
    x=torch.cat(matrices);p=torch.cat(patterns);assert x.shape==(230,3072)
    def fit(a):
        mean=a.mean(0);c=a-mean
        if len(a)>1:
            vals,vec=torch.linalg.eigh(c@c.T/len(a));take=min(8,len(a)-1)
            ix=torch.argsort(vals,descending=True)[:take];e=vals[ix].clamp_min(0)
            factor=c.T@vec[:,ix]/len(a)**.5
            # c^T eigenvectors /sqrt(n) already equals U sqrt(eigenvalue).
            factor[:,e<1e-12]=0
        else:factor=torch.zeros(3072,0,dtype=torch.double)
        factor=F.pad(factor,(0,8-factor.shape[1]))
        tau=(c.square().mean(0)-factor.square().sum(1)).clamp(.01,1)
        return mean,factor,tau
    global_fit=fit(x);means=[];factors=[];taus=[];active=[];counts=[]
    for k in range(8):
        a=x[p==k];counts.append(len(a));active.append(len(a)>0)
        if len(a)>1:m,f,tau=fit(a)
        elif len(a)==1:m=a[0].clone();f=global_fit[1].clone();tau=global_fit[2].clone()
        else:m,f,tau=(v.clone() for v in global_fit)
        for c in range(3):
            if not(k&(1<<c)):m[c*1024:(c+1)*1024]=1;f[c*1024:(c+1)*1024]=0
        means.append(m);factors.append(f);taus.append(tau)
    prior=dict(mu=torch.stack(means).float(),factor=torch.stack(factors).float(),tau2=torch.stack(taus).float(),active=torch.tensor(active),counts=counts)
    assert all(torch.isfinite(prior[k]).all() for k in ('mu','factor','tau2')) and prior['active'].any()
    save(cfg,prior,Path(cfg['root'])/'prior.private.pt')
    write(Path(cfg['root'])/'PRIOR_FIT.json',dict(status='PASS',source_train_slices=230,label_field_reads=230,optimizer=0,model_forward=0,rank=8,grid=32,components_active=sum(active),component_counts=counts,per_patient_artifact_saved=False))
