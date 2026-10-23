"""Local adversarial smoothness of a common-noise posterior predictive mean."""
import json
import torch
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..semantic_evidence_v22.method import sample_semantics
from ..posterior_teacher_v25.method import parameter_groups,AUDIT_STEPS

XI=1e-3
EPSILON=1.0

def unit(x):
    norm=x.flatten(1).norm(dim=1).reshape(-1,1,1,1)
    return x/norm.clamp_min(torch.finfo(x.dtype).tiny)


def kl(reference,prediction):
    target=reference.detach()
    return (target*(target.clamp_min(1e-12).log()-prediction.clamp_min(1e-12).log())).sum(1).mean()


def predictive(s,x):
    rng=torch.Generator().manual_seed(36000000+101*s.steps);probs=[]
    for _ in range(2):
        theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
        for _ in range(2):probs.append(sample_semantics(s,x,theta,a,rng)[0])
    return torch.stack(probs).mean(0)


def auxiliary(s,x):
    assert x.shape[1:]==(1,128,128)
    with torch.no_grad():target=predictive(s,x)
    rng=torch.Generator().manual_seed(36100000+101*s.steps)
    d=unit(torch.randn(x.shape,dtype=x.dtype,generator=rng).to(x.device)).detach().requires_grad_()
    probe=predictive(s,x+XI*d)
    probe_loss=kl(target,probe)
    gradient=torch.autograd.grad(probe_loss,d,create_graph=False)[0]
    perturbation=(EPSILON*unit(gradient)).detach()
    prediction=predictive(s,x+perturbation)
    loss=kl(target,prediction)
    return loss,dict(probe_kl=probe_loss.detach(),direction_gradient_norm=gradient.detach().flatten(1).norm(dim=1).mean(),perturbation_norm=perturbation.flatten(1).norm(dim=1).mean(),zero_direction_fraction=(gradient.flatten(1).norm(dim=1)==0).to(x.dtype).mean(),reference_foreground_mean=target[:,1].mean(),prediction_floor_fraction=(prediction<=1e-12).to(x.dtype).mean(),minimum_probability=prediction.detach().min())


def check_fixture():
    x=torch.tensor([.3,-.2],dtype=torch.float64).reshape(1,1,1,2)
    weight=torch.tensor([1.,3.],dtype=x.dtype).reshape_as(x).requires_grad_()
    def pred(z):
        logit=(weight*z).flatten(1).sum(1).reshape(1,1,1,1)
        return torch.cat((torch.zeros_like(logit),logit),1).softmax(1)
    target=pred(x).detach();assert float(kl(target,pred(x)))==0
    d=unit(torch.ones_like(x)).requires_grad_();loss=kl(target,pred(x+XI*d))
    g=torch.autograd.grad(loss,d)[0];r=EPSILON*unit(g).detach()
    assert torch.allclose(r.flatten().norm(),x.new_tensor(EPSILON))
    assert abs(float(torch.nn.functional.cosine_similarity(g.flatten(),weight.detach().flatten(),dim=0)))>1-1e-10
    value=kl(target,pred(x+r));dw=torch.autograd.grad(value,weight)[0]
    assert torch.isfinite(dw).all() and dw.norm()>0 and not r.requires_grad
    assert torch.equal(unit(torch.zeros_like(x)),torch.zeros_like(x))
    return dict(zero_perturbation_zero_kl=True,power_direction_matches_rank_one_hessian=True,unit_radius=True,zero_direction_safe=True,detached_reference_and_direction=True,parameter_gradient_finite=True,fixture_reverse_calls=2)


class Objective:
    def __init__(self,state,refiner,use_u,audit_path=None):
        self.s=state;self.counter=refiner.counter;self.use_u=use_u
        self.base=BaseObjective(state,refiner,False);self.audit_path=audit_path;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;assert U is not None and 'observations' not in U
        if self.use_u:s.check_batch(U,'U')
        else:assert U['role']=='appearance_donor_L' and U['stage']==s.active['id']
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        aux,stats=auxiliary(s,U['x']);loss=base+aux
        if not torch.isfinite(loss):raise FloatingPointError('posterior local smoothness risk')
        self.counter.add('smoothness_global_draws',6);self.counter.add('smoothness_semantic_forwards',12*len(U['x']));self.counter.add('smoothness_direction_reverse_calls')
        if self.audit_path is not None and s.steps in AUDIT_STEPS:
            groups=parameter_groups(s);params=[p for group in groups.values() for p in group]
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gu=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
            row=dict(step=s.steps,auxiliary_pool='U' if self.use_u else 'L',extra_reverse_calls=2,groups={});offset=0
            for name,pp in groups.items():
                b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gb[offset:offset+len(pp)])])
                v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gu[offset:offset+len(pp)])]);offset+=len(pp)
                nb,nv=float(b.norm()),float(v.norm())
                row['groups'][name]=dict(base_gradient_norm=nb,auxiliary_gradient_norm=nv,auxiliary_over_base=nv/nb if nb else None,cosine=float(torch.dot(b,v)/(b.norm()*v.norm())) if nb and nv else None)
            self.pending=(row,{name:[p.detach().clone() for p in pp] for name,pp in groups.items()})
        return loss,dict(record,scaled=loss.detach(),base_objective=base.detach(),smoothness_risk=aux.detach(),auxiliary_pool_U=loss.new_tensor(float(self.use_u)),**stats)

    def after_update(self,step):
        if self.pending is None:return
        row,before=self.pending;assert step==row['step']+1
        for name,params in parameter_groups(self.s).items():
            row['groups'][name]['actual_combined_Adam_displacement']=float(torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,before[name])]).norm())
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None
