"""A distinct categorical-semantics generative model, labeled boundary only."""
import math
import torch
from torch.nn import functional as F
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.model import normal_log_prob,student_log_prob,sar


class DiscreteLabeledModel(Model):
    def recognition_logits(self,image):
        v=self.shape_encoder(image).chunk(2,1)[0]
        ell=self.logit_encoder(v).chunk(2,1)[0]
        return torch.cat([torch.zeros_like(ell[:,:1]),ell],1)

    def labeled_proposal(self,image,label,sample=True):
        self.calls['proposal_images']+=len(image)
        x=self.gaussian(self.physical_encoder(image),sample)
        m=self.gaussian(self.appearance_encoder(image),sample)
        onehot=F.one_hot(label,4).permute(0,3,1,2).to(image.dtype)
        c=self.gaussian(self.code_encoder(onehot[:,1:]),sample)
        return dict(x=x[0],m=m[0],c=c[0],z=label),x[1]+m[1]+c[1]

    def log_joint(self,image,h,label=None,**kwargs):
        if 'z' not in h:raise ValueError('discrete model requires explicit categorical Z; no legacy ell scoring')
        z=h['z']
        if label is not None:assert torch.equal(z,label)
        probabilities=F.one_hot(z,4).permute(0,3,1,2).to(image.dtype)
        a=self.anatomy_mean(h['c']);logits=torch.cat([torch.zeros_like(a[:,:1]),a],1)
        terms=dict(code=normal_log_prob(h['c'],torch.zeros_like(h['c']),torch.zeros_like(h['c'])),
                   appearance=normal_log_prob(h['m'],torch.zeros_like(h['m']),torch.full_like(h['m'],math.log(.5))),
                   image=student_log_prob(image-h['x']-h['m'],.1),
                   shape=student_log_prob(sar(h['x']-self.structure_mean(probabilities)),.2)+self.logdet,
                   categories=-F.cross_entropy(logits,z,reduction='none').sum((1,2)))
        return sum(terms.values()),terms,logits.softmax(1)

    def bound(self,image,label=None):
        if label is None:raise NotImplementedError('real unlabeled discrete inference is outside this labeled prototype')
        h,logq=self.labeled_proposal(image,label);joint,terms,p=self.log_joint(image,h,label)
        self.calls['density_images']+=len(image)
        return joint-logq,terms,dict(logq=logq,categorical_normalization_error=float((p.sum(1)-1).detach().abs().max()))

    def prediction(self,image,corrected=True,**kwargs):
        if corrected:raise NotImplementedError('no reuse of continuous-logit correction in discrete model')
        return super().prediction(image,corrected=False,**kwargs)
