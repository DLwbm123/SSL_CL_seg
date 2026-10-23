"""Semantic anchoring with exact Gaussian S/b marginalization under the SAR prior."""
import math
import torch
from torch.nn import functional as F
from ..distributions import positive, normal_log_prob, randn
from ..spatial_gaussian import sar_eigenvalues
from ..paired_semantic_v19.method import balanced_risk


class ImageEvidence:
    def __init__(self, model, a):
        c=model.config;self.model=model;self.c=c
        self.gamma,self.beta,self.xi=a.reshape(3,c.channels)
        eig=sar_eigenvalues(c.height,c.width,a,c.alpha)
        self.spectrum=self.gamma[:,None,None].mul(2).exp()*c.sigma_s**2/eig.square()+self.xi[:,None,None].mul(2).exp()
        self.W=model.basis*c.tau_b
        self.DW=self.solve(self.W[None].expand(c.channels,-1,-1,-1))
        eye=torch.eye(len(self.W),dtype=a.dtype,device=a.device)
        self.chol=torch.linalg.cholesky(eye+torch.einsum('khw,clhw->ckl',self.W,self.DW))
        self.logdet=self.spectrum.log().sum((-2,-1))+2*self.chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)

    def solve(self,x):
        # First axis is channel, next axis may enumerate bias columns or images.
        return torch.fft.ifft2(torch.fft.fft2(x)/self.spectrum[:,None]).real

    def score(self,x,prob,theta):
        mean=self.gamma.exp()[None,:,None,None]*self.model.structure_mean(prob,theta)+self.beta[None,:,None,None]
        r=(x-mean).transpose(0,1);Dr=self.solve(r)
        v=torch.einsum('khw,cbhw->cbk',self.W,Dr)
        correction=(v*torch.cholesky_solve(v.transpose(-1,-2),self.chol).transpose(-1,-2)).sum(-1)
        quad=(r*Dr).sum((-2,-1))-correction
        return -.5*(quad+self.logdet[:,None]+self.c.height*self.c.width*math.log(2*math.pi)).sum(0)


def sample_semantics(s,x,theta,a,rng):
    context=s.phi.context(x,theta,a);z={};logq=x.new_zeros(len(x));logp=logq.clone()
    for name in ('c','R'):
        mu,raw=s.phi.parameters_given(name,context,z).chunk(2,1);sd=positive(raw,s.config.scale_floor)
        z[name]=mu+sd*randn(mu.shape,x,rng)
        logq=logq+normal_log_prob(z[name],mu,sd).flatten(1).sum(1)
        logp=logp+normal_log_prob(z[name],x.new_zeros(()),x.new_ones(())).flatten(1).sum(1)
    return s.model.class_probabilities(z,theta),logp-logq


class Objective:
    def __init__(self,state,refiner,use_u):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;s.assert_active();assert sampling_weights is None
        pixels=s.config.height*s.config.width;fit=[];records={k:[] for k in ('task','anchor','image_L','image_U','structure_margin')}
        for outer in range(s.config.global_mc):
            theta=s.q.rsample(generator);a=s.current.rsample(theta,generator);evidence=ImageEvidence(s.model,a);value=theta.sum()*0
            for role,batch in [('L',L),('U',U)]:
                if role=='U' and not self.use_u:continue
                assert batch is not None;s.check_batch(batch,role);x=batch['x']
                # U does not advance L-local or global random streams.
                rng=torch.Generator().manual_seed(2200000+100*s.steps+10*outer+(role=='U'))
                if role=='L':
                    truth=batch['observations']['y_obs'].to(x.dtype)
                    assert batch['observations']['obs_mask'].all() and (truth.sum(1)==1).all()
                    positive_score=evidence.score(x,truth,theta)/pixels
                    wrong=torch.stack([evidence.score(x,torch.roll(truth,8,dim),theta)/pixels for dim in (-1,-2)])
                    anchor=F.softplus(wrong-positive_score).mean()
                    value=value+anchor
                    records['anchor'].append(anchor.detach());records['structure_margin'].append((positive_score-wrong.mean(0)).mean().detach())
                for _ in range(s.config.local_mc):
                    prob,local_prior_minus_q=sample_semantics(s,x,theta,a,rng)
                    self.counter.add('local_mc_samples',len(x));self.counter.add('joint_forwards',len(x))
                    image_elbo=(evidence.score(x,prob,theta)+local_prior_minus_q).mean()/pixels
                    term=-.01*(counts[role]/counts['L'])*image_elbo
                    records['image_'+role].append(image_elbo.detach())
                    if role=='L':
                        task=balanced_risk(prob,truth).mean();term=term+task;records['task'].append(task.detach())
                    value=value+term/s.config.local_mc
            fit.append(value)
        gkl=s.q.kl_to(s.prior);dkl=s.current.expected_kl(s.q);mean_fit=torch.stack(fit).mean();loss=mean_fit+(gkl+dkl)/(counts['L']*pixels)
        if not torch.isfinite(loss):raise FloatingPointError('semantic evidence objective')
        self.counter.add('global_mc_samples',s.config.global_mc)
        return loss,dict(scaled=loss.detach(),fit=mean_fit.detach(),global_kl=gkl.detach(),domain_kl=dkl.detach(),**{k:torch.stack(v).mean() if v else loss.detach()*0 for k,v in records.items()})
