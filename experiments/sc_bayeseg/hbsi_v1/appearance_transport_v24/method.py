"""True-label posterior risk under L- or U-supplied appearance transport.

Original-image evidence and ranking remain shared. No U likelihood, pseudo-label,
consistency target, or synthetic-image generative likelihood is optimized.
"""
import torch
from ..semantic_evidence_v22.method import ImageEvidence,sample_semantics
from ..semantic_alignment_v23.method import ranking
from ..paired_semantic_v19.method import balanced_risk
from .transport import transport


class Objective:
    def __init__(self,state,refiner,use_u):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;s.assert_active();assert sampling_weights is None
        s.check_batch(L,'L');assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        x=L['x'];truth=L['observations']['y_obs'].to(x.dtype)
        assert L['observations']['obs_mask'].all() and (truth.sum(1)==1).all()
        styled=transport(x,U['x']);pixels=s.config.height*s.config.width;fits=[]
        records={k:[] for k in ('task','task_original','task_transported','anchor','image_L','structure_margin','valid_candidate_fraction')}
        for outer in range(s.config.global_mc):
            theta=s.q.rsample(generator);a=s.current.rsample(theta,generator)
            evidence=ImageEvidence(s.model,a);value=theta.sum()*0;probs=[]
            rng=torch.Generator().manual_seed(2200000+100*s.steps+10*outer)
            for _ in range(s.config.local_mc):
                saved=rng.get_state()
                prob,lterm=sample_semantics(s,x,theta,a,rng);probs.append(prob)
                paired=torch.Generator();paired.set_state(saved)
                ps,_=sample_semantics(s,styled,theta,a,paired)
                original=balanced_risk(prob,truth).mean();augmented=balanced_risk(ps,truth).mean()
                task=(original+augmented)/2
                image=(evidence.score(x,prob,theta)+lterm).mean()/pixels
                value=value+(task-.01*image)/s.config.local_mc
                for k,v in [('task',task),('task_original',original),('task_transported',augmented),('image_L',image)]:records[k].append(v.detach())
                self.counter.add('local_mc_samples',2*len(x));self.counter.add('joint_forwards',len(x))
            anchor,margin,valid=ranking(evidence,x,theta,torch.stack(probs).mean(0),truth,s.steps)
            value=value+anchor;records['anchor'].append(anchor.detach());records['structure_margin'].append(margin);records['valid_candidate_fraction'].append(valid)
            fits.append(value)
        gkl=s.q.kl_to(s.prior);dkl=s.current.expected_kl(s.q);fit=torch.stack(fits).mean()
        loss=fit+(gkl+dkl)/(counts['L']*pixels)
        if not torch.isfinite(loss):raise FloatingPointError('appearance transport objective')
        self.counter.add('global_mc_samples',s.config.global_mc)
        return loss,dict(scaled=loss.detach(),fit=fit.detach(),global_kl=gkl.detach(),domain_kl=dkl.detach(),
            transport_rms=(styled-x).square().mean().sqrt(),donor_U=x.new_tensor(float(self.use_u)),
            **{k:torch.stack(v).mean() for k,v in records.items()})
