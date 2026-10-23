"""Semantic supervision before and after image-only local refinement."""
import torch
from .semantic_initialization import SemanticCurriculum
from .objective import SequentialObjective


class RefinedSemanticJoint(SemanticCurriculum):
    def __init__(self,state,refiner,transition=1200):
        super().__init__(state,refiner,warm_steps=10**9,boundary=transition,balanced_ce=True)
        self.transition=transition

    def estimate(self,batchL,batchU,counts,generator,sampling_weights=None):
        semantic,parts=super().estimate(batchL,batchU,counts,generator,sampling_weights)
        if self.state.steps<self.transition:return semantic,parts
        s=self.state;x=batchL['x'];theta=s.q.mean;a=s.current.mean(theta).detach()
        # Labels supervise the resulting prediction but never enter local inference.
        context,offsets=self.refiner.refine(x,theta,a,observations=None,K=3,training=True,generator=generator)
        noise=s.phi.noise(len(x),x,generator)
        z,_=s.phi.rsample_and_log_prob(context,noise,offsets,s.model,theta)
        prob=s.model.class_probabilities(z,theta);truth=batchL['observations']['y_obs'].to(prob.dtype)
        n=truth.flatten(2).sum(2);present=n>0
        terms=-(truth*prob.clamp_min(1e-12).log()).flatten(2).sum(2)
        ce=(((terms/n.clamp_min(1))*present).sum(1)/present.sum(1)).mean()
        fg,y=prob[:,1],truth[:,1]
        dice=(1-(2*(fg*y).flatten(1).sum(1)+1e-6)/(fg.flatten(1).sum(1)+y.flatten(1).sum(1)+1e-6)).mean()
        refined=ce+dice
        elbo,probparts=SequentialObjective.estimate(self,batchL,batchU,counts,generator,sampling_weights)
        denominator=sum(counts.values())*s.config.height*s.config.width
        if denominator<=0:raise ValueError('empty training population')
        total=.5*(semantic+refined)+elbo/denominator
        if not torch.isfinite(total):raise FloatingPointError('refined semantic objective')
        return total,dict(parts,scaled=total.detach(),refined_semantic=refined.detach(),normalized_elbo=(elbo/denominator).detach(),joint_phase=1,**{'prob/'+k:v for k,v in probparts.items()})
