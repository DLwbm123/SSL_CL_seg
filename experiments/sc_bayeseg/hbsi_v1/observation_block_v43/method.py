"""Remove observation score and ranking, retain the original local variational term."""
from contextlib import nullcontext
import torch
from ..semantic_evidence_v22.method import ImageEvidence, sample_semantics
from ..semantic_alignment_v23.method import ranking
from ..paired_semantic_v19.method import balanced_risk


class Objective:
    def __init__(self,state,refiner):
        self.s=state;self.counter=refiner.counter

    def components(self,L,counts,generator,preflight=False):
        s=self.s;s.assert_active();s.check_batch(L,'L')
        pixels=s.config.height*s.config.width;x=L['x']
        truth=L['observations']['y_obs'].to(x.dtype)
        assert L['observations']['obs_mask'].all() and (truth.sum(1)==1).all()
        task_values=[];local_values=[];scores=[];anchors=[];margins=[];valids=[];fits=[];full_fits=[]
        for outer in range(s.config.global_mc):
            theta=s.q.rsample(generator);a=s.current.rsample(theta,generator)
            rng=torch.Generator().manual_seed(2200000+100*s.steps+10*outer)
            # The removed deterministic block is still computed for logs. Only
            # the implementation check builds its graph; training never does.
            with nullcontext() if preflight else torch.no_grad():evidence=ImageEvidence(s.model,a)
            value=theta.sum()*0;full_value=theta.sum()*0;probs=[]
            for _ in range(s.config.local_mc):
                prob,local=sample_semantics(s,x,theta,a,rng);probs.append(prob)
                self.counter.add('local_mc_samples',len(x));self.counter.add('joint_forwards',len(x))
                task=balanced_risk(prob,truth).mean();regularizer=local.mean()/pixels
                with nullcontext() if preflight else torch.no_grad():score=evidence.score(x,prob,theta).mean()/pixels
                value=value+(task-.01*regularizer)/s.config.local_mc
                full_value=full_value+(task-.01*(score+regularizer))/s.config.local_mc
                task_values.append(task);local_values.append(regularizer);scores.append(score)
            with nullcontext() if preflight else torch.no_grad():
                anchor,margin,valid=ranking(evidence,x,theta,torch.stack(probs).mean(0),truth,s.steps)
            fits.append(value);full_fits.append(full_value+anchor)
            anchors.append(anchor);margins.append(margin);valids.append(valid)
        avg=lambda v:torch.stack(v).mean()
        gkl=s.q.kl_to(s.prior);dkl=s.current.expected_kl(s.q);kl=(gkl+dkl)/(counts['L']*pixels)
        loss=avg(fits)+kl
        if not torch.isfinite(loss):raise FloatingPointError('retained objective')
        self.counter.add('global_mc_samples',s.config.global_mc)
        records=dict(scaled=loss,fit=avg(fits),task=avg(task_values),local_prior_minus_q=avg(local_values),score_x=avg(scores),anchor=avg(anchors),global_kl=gkl,domain_kl=dkl,full_objective_same_samples=avg(full_fits)+kl,structure_margin=avg(margins),valid_candidate_fraction=avg(valids))
        return loss,records

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        assert sampling_weights is None
        loss,records=self.components(L,counts,generator)
        return loss,{k:v.detach() for k,v in records.items()}
