"""Observed-label chart of a Gaussian logit proposal; generator unchanged."""
import torch
from torch.nn import functional as F
from sc_bayeseg.split_v1.model import Model


def transform(u,label):
    z=-u;ell=-F.softplus(z)
    for k in (1,2,3):
        winner=u[:,k-1:k];zz=winner-u
        zz=torch.cat([winner if j==k-1 else zz[:,j:j+1] for j in range(3)],1)
        sw=F.softplus(winner);ss=F.softplus(zz)
        out=torch.cat([sw if j==k-1 else sw-ss[:,j:j+1] for j in range(3)],1)
        mask=(label==k)[:,None];z=torch.where(mask,zz,z);ell=torch.where(mask,out,ell)
    logdet=F.logsigmoid(z).sum(tuple(range(1,z.ndim)))
    return ell,logdet


def inverse(ell,label):
    inv=lambda x:x+torch.log(-torch.expm1(-x))
    u=-inv(-ell)
    for k in (1,2,3):
        win=ell[:,k-1:k];zk=inv(win)
        out=torch.cat([zk if j==k-1 else zk-inv(win-ell[:,j:j+1]) for j in range(3)],1)
        u=torch.where((label==k)[:,None],out,u)
    return u


class LabelChartModel(Model):
    def __init__(self):
        super().__init__();self.chart_enabled=True;self.chart_label=None;self.chart_stats={}

    def proposal(self,image,shifts=None,sample=True):
        if not self.chart_enabled or self.chart_label is None:return super().proposal(image,shifts,sample)
        assert shifts is None,'local shifted inference outside this audit'
        self.calls['proposal_images']+=len(image)
        v=self.shape_encoder(image).chunk(2,1)[0]
        x=self.gaussian(self.physical_encoder(image),sample);m=self.gaussian(self.appearance_encoder(image),sample)
        base=self.gaussian(self.logit_encoder(v),sample)
        ell,logdet=transform(base[0],self.chart_label)
        c=self.gaussian(self.code_encoder(ell),sample)
        h=dict(x=x[0],m=m[0],ell=ell,c=c[0]);logq=x[1]+m[1]+base[1]-logdet+c[1]
        z=torch.cat([torch.zeros_like(ell[:,:1]),ell],1);top=z.max(1).values
        self.chart_stats=dict(argmax_correct_fraction=float((z.detach().argmax(1)==self.chart_label).float().mean()),
                              tied_top_fraction=float(((z.detach()==top[:,None]).sum(1)>1).float().mean()),
                              logdet_per_pixel=float(logdet.detach().mean()/self.size**2))
        self.proposal_stats={}
        return h,logq

    def bound(self,image,label=None):
        self.chart_label=label if self.chart_enabled else None
        try:return super().bound(image,label)
        finally:self.chart_label=None
