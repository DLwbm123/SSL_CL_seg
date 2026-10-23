"""Fixed class-labelled Normal-Inverse-Gamma intensity observation factor."""
import math
import torch
import torch.nn.functional as F
from ..data_adapter import leaf_observations
from ..stream_state import digest


def posterior(n,total,squares):
    # Fixed prior: mu0=0, kappa0=1, alpha0=2, beta0=1.
    kappa=1+n;mu=total/kappa;alpha=2+n/2
    beta=1+(squares-total.square()/kappa)/2
    assert torch.all(beta>0)
    return dict(df=2*alpha,loc=mu,scale=(beta*(kappa+1)/(alpha*kappa)).sqrt())


def log_density(x,parameters):
    v={k:t.to(x).reshape(1,-1,1,1) for k,t in parameters.items()}
    return torch.distributions.StudentT(**v).log_prob(x)


def mixture_risk(p,loglik):
    return -torch.logsumexp(p.clamp_min(1e-12).log()+loglik.detach(),dim=1).mean()


@torch.no_grad()
def fit(s,data,path):
    before=digest(s.payload());n=torch.zeros(2,dtype=torch.float64,device=s.q.mean.device)
    total=torch.zeros_like(n);squares=torch.zeros_like(n)
    assert data.counts['L']==80 and data.counts['U']==271
    assert s.config.height==s.config.width==128 and s.config.classes==2
    for record in data.records['L']:
        x=data.read(record,'image').to(n)[None,None]
        x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False)[:,0]
        raw=data.read(record,'label')[None,None].double().to(x.device)
        y=F.interpolate(raw,(128,128),mode='nearest')[:,0].long()
        obs=leaf_observations(y,s.config.classes,s.config.label_groups)
        assert obs['obs_mask'].all() and (obs['y_obs'].sum(1)==1).all()
        label=obs['y_obs'].long().argmax(1)
        assert torch.isfinite(x).all()
        for c in range(2):
            z=x[label==c];n[c]+=z.numel();total[c]+=z.sum();squares[c]+=z.square().sum()
    assert n.sum()==80*128*128 and (n>0).all()
    parameters=posterior(n,total,squares)
    assert digest(s.payload())==before
    with path.open('xb') as f:torch.save(dict(parameters={k:v.cpu() for k,v in parameters.items()},counts=n.cpu(),source_payload=before,evidence=data.evidence),f)
    return parameters,dict(status='COMPLETE',L_image_reads=80,L_label_reads=80,U_label_reads=0,model_forward_calls=0,source_updates=0,foreground_pixels=int(n[1]),background_pixels=int(n[0]),source_payload_unchanged=True,finite_parameters=all(bool(torch.isfinite(v).all()) for v in parameters.values()),fitted_parameters_public=False)


def check_fixture():
    z=torch.tensor([.1,.4,.9],dtype=torch.float64)
    n=z.new_tensor([3.]);p=posterior(n,z.sum()[None],z.square().sum()[None])
    mean=z.mean();expected_beta=1+((z-mean).square().sum()+3/4*mean.square())/2
    assert torch.allclose(p['scale'].square(),expected_beta*5/(3.5*4))
    # Prior predictive at n=0: t_4(0,sqrt(1)); also handles absent classes.
    zero=posterior(n*0,n*0,n*0)
    assert zero['df'].item()==4 and zero['loc'].item()==0 and zero['scale'].item()==1
    value=log_density(z[None,None,None,:],p)
    df=p['df'];scale=p['scale'];loc=p['loc']
    manual=torch.lgamma((df+1)/2)-torch.lgamma(df/2)-.5*torch.log(df*math.pi)-scale.log()-(df+1)/2*torch.log1p(((z-loc)/scale).square()/df)
    assert torch.allclose(value.flatten(),manual,atol=1e-12,rtol=1e-12)
    logits=torch.tensor([0.,0.],dtype=z.dtype).reshape(1,2,1,1).requires_grad_()
    ll=torch.tensor([0.,2.],dtype=z.dtype).reshape(1,2,1,1)
    prob=logits.softmax(1);loss=mixture_risk(prob,ll)
    g=torch.autograd.grad(loss,logits)[0]
    assert g[0,1,0,0]<0 and torch.isfinite(g).all()
    assert torch.allclose(loss,mixture_risk(prob.flip(1),ll.flip(1)))
    assert torch.allclose(mixture_risk(prob,torch.ones_like(ll)),z.new_tensor(-1.))
    return dict(conjugate_variance_formula=True,prior_predictive=True,student_t_logdensity_formula=True,class_permutation_invariant=True,gradient_toward_supported_class=True,identical_density_constant_risk=True,fixture_reverse_calls=1)
