"""Frozen bandit action, quality, state and controller mathematics."""
import copy,math
import torch
from torch import nn

ACTIONS=('SKIP','COARSE_DISC_BG','FINE_BG_RIM_CUP')
ADAPTIVE=('REG','NC_RL','RL')


def coarse(p):return torch.stack((p[:,0],p[:,1]+p[:,2]),1)


def u_loss(p,q,valid,action,selection=None):
    if action not in range(3):raise ValueError('action outside registry')
    if p.shape!=q.shape or p.ndim!=4 or p.shape[1]!=3 or valid.shape!=p.shape[:1]+p.shape[2:]:raise ValueError('action geometry')
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():raise FloatingPointError('nonfinite probabilities')
    with torch.autocast(p.device.type,enabled=False):
        p=p.float();q=q.detach().float();valid=valid.bool()
        coverage={str(a):float(((coarse(q) if a==1 else q).max(1).values>.7)[valid].float().mean()) if valid.any() else 0. for a in (1,2)}
        if action==0:return p.sum()*0,coverage
        if action==1:p,q=coarse(p),coarse(q)
        admitted=valid&(q.max(1).values>.7)
        if selection is not None:
            if selection.shape!=valid.shape or selection.dtype!=torch.bool:raise ValueError('selection geometry/dtype')
            admitted=admitted&selection.detach()
        kl=(q*(q.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(1)
        return (kl*admitted).sum()/valid.sum().clamp_min(1),coverage


def quality(p,y):
    if p.ndim!=4 or p.shape[1]!=3 or y.shape!=p.shape[:1]+p.shape[2:]:raise ValueError('feedback geometry')
    if not torch.isfinite(p).all() or not ((y==255)|((y>=0)&(y<3))).all():raise ValueError('feedback nonfinite or invalid labels')
    with torch.autocast(p.device.type,enabled=False):
        p=p.float();values=[]
        for b in range(len(p)):
            valid=y[b]!=255
            if not valid.any():continue
            supported=torch.unique(y[b][valid]);nll=torch.stack([-p[b,int(c)][y[b]==c].clamp_min(1e-8).log().mean() for c in supported]).mean()
            dice=[]
            for c in (1,2):
                pred=p[b,c][valid];truth=(y[b][valid]==c).float();dice.append(1-(2*(pred*truth).sum()+1)/(pred.sum()+truth.sum()+1))
            values.append(-(nll+(dice[0]+dice[1])/2))
        return torch.stack(values).mean() if values else None


def state_vector(q,p,valid,lset,t,lr_ratio):
    q=q.detach().float();p=p.detach().float();v=valid.bool()
    if not v.any():raise ValueError('empty valid geometry')
    ent=lambda x:-(x*x.clamp_min(1e-8).log()).sum(1)
    qc=coarse(q);disc=q[:,1]+q[:,2];cond=q[:,1:]/disc[:,None].clamp_min(1e-8)
    mass=disc[v].sum();unsupported=bool(mass<=1e-8)
    conditional=(ent(cond)[v]*disc[v]).sum()/mass/math.log(2) if not unsupported else q.new_tensor(1.)
    mix=(q+p)/2;js=((q*(q.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1)+(p*(p.clamp_min(1e-8).log()-mix.clamp_min(1e-8).log())).sum(1))/2/math.log(2)
    vals=[q.new_tensor(t/1200),q.new_tensor(lr_ratio)]
    vals += [(q.argmax(1)[v]==c).float().mean() for c in range(3)]
    vals += [q[:,c][v].mean() for c in range(3)]
    vals += [ent(q)[v].mean()/math.log(3),ent(qc)[v].mean()/math.log(2),conditional,(q.max(1).values[v]>.7).float().mean(),(qc.max(1).values[v]>.7).float().mean(),js[v].mean(),lset.detach()/(1+lset.detach()),(q.argmax(1)[v]!=p.argmax(1)[v]).float().mean()]
    z=torch.stack(vals).detach();assert z.shape==(16,) and torch.isfinite(z).all()
    return z,dict(conditional_unsupported=unsupported)
