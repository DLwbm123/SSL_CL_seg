"""Strict current-L provider and once-only patient-balanced occupancy prior."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..main_style_ssl_v1.data import Accessor,metadata
from ..five_frameworks_v1.recipes import generator
from ..di_dmpa_gate1.binding import safe_asset
from ..main_style_ssl_v1.runtime import write
from .runtime import guard,save

def occupancy(y):return F.avg_pool2d(F.one_hot(y,4).permute(0,3,1,2).float()[:,1:],12)
def patterns(y):return sum(((y==c).flatten(1).any(1).long()<<(c-1)) for c in (1,2,3))
class Current:
    def __init__(self,cfg):
        import h5py
        self.cfg=cfg;self.arm=cfg['arm'];self.l_reads=self.u_reads=0
        self.access=Accessor(cfg['data'],'mnms','Siemens' if cfg['stage']=='source' else 'Philips','train_labeled')
        assert len(self.access.patients)==(10 if cfg['stage']=='source' else 13)
        allowed={str(safe_asset(self.access.data,r[k])) for r in self.access.rows for k in ('image_h5_relpath','label_h5_relpath')};original=h5py.File
        def checked(path,*a,**kw):
            if str(Path(path).resolve()) not in allowed:raise PermissionError('HDF5 outside current training L')
            return original(path,*a,**kw)
        h5py.File=checked
    def labeled(self,step,diagnostic=False):
        self.l_reads+=1;g=generator(20271612 if diagnostic else 20271611,1,0 if self.cfg['stage']=='source' else 1,step,'APOS_V2_L')
        xx=[];yy=[];ids=[]
        for pos in torch.randperm(len(self.access.patients),generator=g)[:2].tolist():
            patient=self.access.patients[pos];ii=self.access.by_patient[patient];i=ii[int(torch.randint(len(ii),(),generator=g))];z=int(torch.randint(self.access.depths[i],(),generator=g))
            a=self.access.read(i,z);xx.append(a['image']);yy.append(a['label']);ids.append(patient)
        return torch.stack(xx).cuda(),torch.stack(yy).cuda(),tuple(ids)

def fit_prior(cfg):
    import h5py
    rows=[r for r in metadata(cfg['data'],'mnms') if r['site_or_vendor']=='Siemens' and r['primary_20pct_split']=='train_labeled']
    patients=sorted({r['patient_id'] for r in rows});assert len(patients)==10 and len(rows)==20
    xx=[];ww=[];pp=[];patients_of=[]
    for r in rows:
        guard(cfg)
        with h5py.File(safe_asset(Path(cfg['data']),r['label_h5_relpath']),'r') as f:y=torch.from_numpy(f['label'][...].astype(np.int64))
        xx.append(occupancy(y).flatten(1).double());pp.append(patterns(y));ww.extend([1/(10*2*len(y))]*len(y));patients_of.extend([patients.index(r['patient_id'])]*len(y))
    x=torch.cat(xx);mode=torch.cat(pp);weights=torch.tensor(ww,dtype=torch.double);people=torch.tensor(patients_of);assert x.shape==(230,3072)
    eigcalls=0
    def fitted(values,w):
        nonlocal eigcalls
        w=w/w.sum();mu=(w[:,None]*values).sum(0);Z=(values-mu)*w.sqrt()[:,None]
        eig,V=torch.linalg.eigh(Z@Z.T);eigcalls+=1;ix=torch.argsort(eig,descending=True)[:8]
        factor=Z.T@V[:,ix];factor[:,eig[ix]<1e-12]=0
        lengths=factor.square().sum(0).sqrt();basis=factor/lengths.clamp_min(1e-12);basis[:,lengths<1e-10]=0
        return mu,factor,basis
    def bank(keep):
        vals=x[keep];base=weights[keep];base=base/base.sum();mods=mode[keep];pe=people[keep]
        global_fit=fitted(vals,base);out=[];alphas=[]
        for k in range(8):
            take=mods==k;n=len(torch.unique(pe[take]));alpha=n/(n+2);alphas.append(alpha)
            if not n:out.append(global_fit)
            else:
                conditional=base*take;conditional/=conditional.sum();out.append(fitted(vals,alpha*conditional+(1-alpha)*base))
        return global_fit,out,alphas
    global_fit,full,alphas=bank(torch.ones(230,dtype=torch.bool));global_squared=torch.zeros(3072,dtype=torch.double);conditional_squared=torch.zeros(8,3072,dtype=torch.double);mass=torch.zeros(8,dtype=torch.double)
    for patient in range(10):
        guard(cfg);keep=people!=patient;held=~keep;gf,bf,_=bank(keep);values=x[held];w=weights[held]
        def square(fit):
            mu,_,basis=fit;res=values-mu;return (res-(res@basis)@basis.T).square()
        global_squared+=(w[:,None]*square(gf)).sum(0)
        for k in range(8):
            wk=w*(mode[held]==k);conditional_squared[k]+=(wk[:,None]*square(bf[k])).sum(0);mass[k]+=wk.sum()
    sigma=[]
    for k in range(8):
        conditional=conditional_squared[k]/mass[k] if mass[k]>0 else global_squared
        sigma.append((alphas[k]*conditional+(1-alphas[k])*global_squared).clamp_min(1e-6))
    freq=torch.tensor([float(weights[mode==k].sum()) for k in range(8)])+.05;freq/=freq.sum()
    prior=dict(mu=torch.stack([v[0] for v in full]).float(),factor=torch.stack([v[1] for v in full]).float(),sigma2=torch.stack(sigma).float(),mode_probs=freq.float())
    assert eigcalls<=99 and all(torch.isfinite(v).all() for v in prior.values())
    save(cfg,prior,Path(cfg['root'])/'prior.private.pt')
    write(Path(cfg['root'])/'PRIOR_FIT.json',dict(status='PASS',source_label_slices=230,source_patients=10,eigh_calls=eigcalls,patient_balanced=True,LOPO=True,mode_counts=[int((mode==k).sum()) for k in range(8)],mode_shrinkage_alpha=alphas,variance_floor_fraction=float((prior['sigma2']<=1.000001e-6).float().mean()),image_array_reads=0,model_samples=0,optimizer=0))
