"""Context-only recognition scored on a held-out Gaussian image marginal."""
import math
from types import SimpleNamespace
import torch
from ..spatial_gaussian import sar_eigenvalues,sar,dct_basis
from ..semantic_evidence_v22.method import sample_semantics
from ..semantic_alignment_v23.method import Objective as BaseObjective


def indices(x,step):
    assert x.shape[-2:]==(128,128)
    cy,cx=((32,32),(32,96),(96,32),(96,96))[step%4]
    y=torch.arange(cy-8,cy+8,device=x.device)
    z=torch.arange(cx-8,cx+8,device=x.device)
    return (y[:,None]*128+z[None,:]).flatten()


def hide(x,index):
    flat=x.flatten(2);visible=flat.clone();visible[:,:,index]=0
    fill=visible.sum(-1,keepdim=True)/(flat.shape[-1]-len(index))
    visible[:,:,index]=fill
    return visible.reshape_as(x)


class Marginal:
    def __init__(self,model,a,index):
        # Only Pi gets score gradients; the original L objective still learns F/a.
        self.model=model;self.index=index;c=model.config;a=a.detach()
        self.gamma,self.beta,xi=a.reshape(3,c.channels)
        eig=sar_eigenvalues(c.height,c.width,a,c.alpha)
        spectrum=self.gamma[:,None,None].mul(2).exp()*c.sigma_s**2/eig.square()+xi[:,None,None].mul(2).exp()
        kernel=torch.fft.ifft2(spectrum).real
        y=index//c.width;z=index%c.width
        cov=kernel[:,(y[:,None]-y[None,:])%c.height,(z[:,None]-z[None,:])%c.width]
        w=(model.basis*c.tau_b).flatten(1)[:,index]
        self.cov=cov+(w.T@w)[None]
        self.chol=torch.linalg.cholesky(self.cov)
        self.logdet=2*self.chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)

    def score(self,x,prob,theta):
        mu=self.gamma.exp()[None,:,None,None]*self.model.structure_mean(prob,theta.detach())+self.beta[None,:,None,None]
        r=(x-mu).flatten(2)[:,:,self.index]
        solved=torch.cholesky_solve(r.permute(1,2,0),self.chol).permute(2,0,1)
        score=-.5*((r*solved).sum(-1)+self.logdet[None]+len(self.index)*math.log(2*math.pi)).sum(-1)
        return score,dict(heldout_residual_mse=r.detach().square().mean(),heldout_logdet_per_pixel=self.logdet.detach().mean()/len(self.index))


def auxiliary(s,x):
    index=indices(x,s.steps);masked=hide(x,index)
    rng=torch.Generator().manual_seed(34000000+101*s.steps)
    losses=[];scores=[];stats=[]
    for _ in range(2):
        theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
        evidence=Marginal(s.model,a,index)
        for _ in range(2):
            prob,_=sample_semantics(s,masked,theta,a,rng)
            score,record=evidence.score(x,prob,theta)
            scores.append(score.mean()/len(index));stats.append(record)
            losses.append(-.01*score.mean()/len(index))
    loss=torch.stack(losses).mean()
    return loss,dict(heldout_logscore_per_pixel=torch.stack(scores).detach().mean(),heldout_fraction=loss.new_tensor(len(index)/16384),**{k:torch.stack([v[k] for v in stats]).mean() for k in stats[0]})


def check_fixture():
    c=SimpleNamespace(height=4,width=4,channels=1,alpha=.9,sigma_s=.2,tau_b=.3)
    model=SimpleNamespace(config=c,basis=dct_basis(4,4,2),structure_mean=lambda p,t:p[:,:1]*t[0])
    a=torch.tensor([.1,.2,-.3],dtype=torch.float64,requires_grad=True);index=torch.tensor([0,1,4,5]);theta=torch.tensor([1.2],dtype=a.dtype,requires_grad=True)
    e=Marginal(model,a,index)
    operator=sar(torch.eye(16,dtype=torch.float64).reshape(16,1,4,4),c.alpha).flatten(1).T
    inv=torch.linalg.inv(operator);w=(model.basis*c.tau_b).flatten(1).T
    dense=(2*a[0]).exp()*c.sigma_s**2*inv@inv.T+(2*a[2]).exp()*torch.eye(16,dtype=a.dtype)+w@w.T
    expected=dense[index][:,index];assert torch.allclose(e.cov[0],expected,atol=1e-12,rtol=1e-12)
    x=torch.arange(16,dtype=a.dtype).reshape(1,1,4,4)/16;p=torch.full_like(x,.4,requires_grad=True)
    value,_=e.score(x,p,theta);mu=(a[0].exp()*p*theta[0]+a[1]).flatten()[index]
    target=x.flatten()[index];reference=torch.distributions.MultivariateNormal(mu,covariance_matrix=expected).log_prob(target)
    assert torch.allclose(value[0],reference,atol=1e-12,rtol=1e-12)
    gp,gt,ga=torch.autograd.grad(value.sum(),(p,theta,a),allow_unused=True)
    assert torch.isfinite(gp).all() and gp.norm()>0 and gt is None and ga is None
    altered=x.clone().flatten(2);altered[:,:,index]+=9
    assert torch.equal(hide(x,index),hide(altered.reshape_as(x),index))
    return dict(SAR_dense_covariance_match=True,marginal_logprob_match=True,heldout_input_independent=True,finite_probability_gradient=True,direct_score_theta_a_stopped=True,fixture_reverse_calls=1)


class Objective:
    def __init__(self,state,refiner,use_u):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u
        self.base=BaseObjective(state,refiner,False)

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        aux,stats=auxiliary(s,U['x']);loss=base+aux
        if not torch.isfinite(loss):raise FloatingPointError('masked image marginal risk')
        self.counter.add('masked_global_draws',2);self.counter.add('masked_semantic_forwards',4*len(U['x']));self.counter.add('heldout_covariance_factorizations',2)
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),masked_risk=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)
