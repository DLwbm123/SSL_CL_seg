"""Normalized categorical model with an independent Gaussian spatial residual."""
import torch
from torch.nn import functional as F
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.model import normal_log_prob,student_log_prob,sar
import math


class SpatialResidualModel(DiscreteLabeledModel):
    @staticmethod
    def residual_kl(q):
        mu,logsd=q
        return .5*(mu.square()+torch.exp(2*logsd)-1-2*logsd).flatten(1).sum(1)

    def spatial_logits(self,c,residual):
        base=F.interpolate(F.relu(self.anatomy[0](c)).reshape(-1,16,4,4),(self.size,self.size),mode='bilinear',align_corners=False)
        feature=(base+F.interpolate(residual,(self.size,self.size),mode='bilinear',align_corners=False)).clamp_min(0)
        a=self.anatomy[2](F.relu(self.anatomy[1](feature)))
        return torch.cat([torch.zeros_like(a[:,:1]),a],1)

    def conditional_joint(self,image,h,residual):
        z=h['z'];onehot=F.one_hot(z,4).permute(0,3,1,2).to(image.dtype);logits=self.spatial_logits(h['c'],residual)
        terms=dict(code=normal_log_prob(h['c'],torch.zeros_like(h['c']),torch.zeros_like(h['c'])),
                   appearance=normal_log_prob(h['m'],torch.zeros_like(h['m']),torch.full_like(h['m'],math.log(.5))),
                   image=student_log_prob(image-h['x']-h['m'],.1),
                   shape=student_log_prob(sar(h['x']-self.structure_mean(onehot)),.2)+self.logdet,
                   categories=-F.cross_entropy(logits,z,reduction='none').sum((1,2)))
        return sum(terms.values()),terms,logits.softmax(1)

    def bound(self,image,label=None,residual_q=None):
        if label is None:raise NotImplementedError('real U and spatial recognition are not implemented')
        if residual_q is None:raise ValueError('explicit local Gaussian residual posterior required')
        mu,logsd=residual_q;assert mu.shape==logsd.shape==(len(image),16,16,16)
        h,logq=self.labeled_proposal(image,label);residual=mu+logsd.exp()*torch.randn_like(mu)
        joint,terms,p=self.conditional_joint(image,h,residual);kl=self.residual_kl(residual_q)
        return joint-logq-kl,terms,dict(logq=logq,residual_KL=kl,normalization_error=float((p.sum(1)-1).detach().abs().max()))
