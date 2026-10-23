"""Prediction-matched L ranking and U semantic-only image guidance.

This routed generalized-Bayes risk is not an ELBO: the U local KL is omitted,
and direct U gradients to image-scoring parameters are stopped.
"""
import torch
from torch.nn import functional as F
from ..semantic_evidence_v22.method import ImageEvidence, sample_semantics
from ..paired_semantic_v19.method import balanced_risk
from .candidates import pairs


def ranking(evidence,x,theta,prob,truth,step):
    losses=[];margins=[];valids=[]
    for pos,neg,support,valid in pairs(prob,truth,step):
        margin=(evidence.score(x,pos,theta)-evidence.score(x,neg,theta))/support
        losses.append(F.softplus(-margin)*valid)
        margins.append(margin.detach()*valid);valids.append(valid)
    n=torch.stack(valids).sum().clamp_min(1)
    return torch.stack(losses).sum()/n,torch.stack(margins).sum()/n,torch.stack(valids).double().mean()


def semantic_score(s,x,prob,theta,a):
    # Stop only the scoring inputs. Pi still carries phi/G/domain gradients.
    return ImageEvidence(s.model,a.detach()).score(x,prob,theta.detach())


class Objective:
    def __init__(self,state,refiner,use_u):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;s.assert_active();assert sampling_weights is None
        pixels=s.config.height*s.config.width;fit=[]
        records={k:[] for k in ('task','anchor','image_L','image_U','structure_margin','valid_candidate_fraction','U_local_prior_minus_q')}
        for outer in range(s.config.global_mc):
            theta=s.q.rsample(generator);a=s.current.rsample(theta,generator)
            evidence=ImageEvidence(s.model,a);value=theta.sum()*0
            for role,batch in [('L',L),('U',U)]:
                if role=='U' and not self.use_u:continue
                assert batch is not None;s.check_batch(batch,role);x=batch['x']
                rng=torch.Generator().manual_seed(2200000+100*s.steps+10*outer+(role=='U'))
                probs=[]
                if role=='L':
                    truth=batch['observations']['y_obs'].to(x.dtype)
                    assert batch['observations']['obs_mask'].all() and (truth.sum(1)==1).all()
                for _ in range(s.config.local_mc):
                    prob,local_prior_minus_q=sample_semantics(s,x,theta,a,rng);probs.append(prob)
                    self.counter.add('local_mc_samples',len(x));self.counter.add('joint_forwards',len(x))
                    if role=='L':
                        image=(evidence.score(x,prob,theta)+local_prior_minus_q).mean()/pixels
                        task=balanced_risk(prob,truth).mean()
                        term=task-.01*image;records['task'].append(task.detach())
                    else:
                        image=semantic_score(s,x,prob,theta,a).mean()/pixels
                        term=-.01*(counts['U']/counts['L'])*image
                        records['U_local_prior_minus_q'].append(local_prior_minus_q.detach().mean()/pixels)
                    records['image_'+role].append(image.detach());value=value+term/s.config.local_mc
                if role=='L':
                    anchor,margin,valid=ranking(evidence,x,theta,torch.stack(probs).mean(0),truth,s.steps)
                    value=value+anchor
                    records['anchor'].append(anchor.detach());records['structure_margin'].append(margin)
                    records['valid_candidate_fraction'].append(valid)
            fit.append(value)
        gkl=s.q.kl_to(s.prior);dkl=s.current.expected_kl(s.q)
        mean_fit=torch.stack(fit).mean();loss=mean_fit+(gkl+dkl)/(counts['L']*pixels)
        if not torch.isfinite(loss):raise FloatingPointError('semantic alignment objective')
        self.counter.add('global_mc_samples',s.config.global_mc)
        return loss,dict(scaled=loss.detach(),fit=mean_fit.detach(),global_kl=gkl.detach(),domain_kl=dkl.detach(),
            **{k:torch.stack(v).mean() if v else loss.detach()*0 for k,v in records.items()})
