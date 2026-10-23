"""Expected conditional risk under the same nested student draws; fixed N target."""
import json
import torch
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..semantic_evidence_v22.method import sample_semantics

AUDIT_STEPS=(1600,1799,1999)


from ..posterior_teacher_v25.method import teacher_risk


def student_draws(s,x):
    # Fixed nontrivial view; flip alignment is exact, no augmentation search.
    flip=s.steps%2==0
    gain,bias=(.95,.025) if s.steps%2==0 else (1.05,-.025)
    view=gain*x+bias
    if flip:view=view.flip(-1)
    probs=[];rng=torch.Generator().manual_seed(25000000+101*s.steps)
    for _ in range(s.config.global_mc):
        theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
        for _ in range(s.config.local_mc):probs.append(sample_semantics(s,view,theta,a,rng)[0])
    samples=torch.stack(probs)
    return samples.flip(-1) if flip else samples


def expected_risk(target,samples,weight):
    risks=[]
    for student in samples:
        risk,details=teacher_risk(target,student,weight);risks.append(risk)
    expected=torch.stack(risks).mean()
    marginal,_=teacher_risk(target,samples.mean(0),weight)
    # The marginal risk is telemetry only; autograd uses expected risk.
    details.update(marginal_U_risk=marginal.detach(),Jensen_gap=(expected-marginal).detach(),sample_probability_min=samples.detach().min(),probability_floor_fraction=(samples.detach()<1e-12).double().mean(),nested_probability_variance=samples.detach().var(0,unbiased=False).sum(1).mean())
    return expected,details


def fixture():
    target=torch.tensor([[[[.25]],[[.75]]]],dtype=torch.float64)
    samples=torch.tensor([[[[[.1]],[[.9]]]],[[[[.8]],[[.2]]]]],dtype=torch.float64)
    weight=torch.ones_like(target[:,0]);value,info=expected_risk(target,samples,weight)
    direct=(target*(target.log()-samples.log())).sum(2).mean()
    assert torch.allclose(value,direct) and info['Jensen_gap']>0
    same=target.expand(4,-1,-1,-1,-1).clone();v,d=expected_risk(target,same,weight)
    assert abs(float(v))<1e-12 and abs(float(d['Jensen_gap']))<1e-12
    return dict(expected_KL_identity=True,Jensen_positive_on_disagreeing_samples=True,identical_samples_equal_marginal=True,fixture_reverse_calls=0)


def parameter_groups(s):
    return dict(q_mean=[s.q.mean],q_scale=[s.q.raw_scale],phi_encoder=list(s.phi.enc.parameters())+list(s.phi.down.parameters())+list(s.phi.up.parameters()),phi_cR=list(s.phi.c_head.parameters())+list(s.phi.r_head.parameters()),domain=list(s.current.parameters()))


class Objective:
    def __init__(self,state,refiner,arm,audit_path=None):
        self.s=state;self.base=BaseObjective(state,refiner,False);self.arm=arm
        self.counter=refiner.counter;self.audit_path=audit_path;self.pending=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        u=base*0;details={}
        if self.arm!='A':
            s.check_batch(U,'U');target=U['teacher_mean'].to(L['x'].device)
            w=U['teacher_weight'].to(L['x'].device) if self.arm=='C' else torch.ones_like(target[:,0])
            samples=student_draws(s,U['x']);u,details=expected_risk(target,samples,w)
            self.counter.add('local_mc_samples',s.config.global_mc*s.config.local_mc*len(U['x']))
            self.counter.add('global_mc_samples',s.config.global_mc)
        loss=base+u
        if not torch.isfinite(loss):raise FloatingPointError('fixed teacher risk')
        if self.audit_path is not None and s.steps in AUDIT_STEPS:
            groups=parameter_groups(s);params=[p for group in groups.values() for p in group]
            gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            gu=torch.autograd.grad(u,params,retain_graph=True,allow_unused=True) if self.arm!='A' else [None]*len(params)
            row=dict(step=s.steps,arm=self.arm,extra_reverse_calls=2 if self.arm!='A' else 1,groups={});offset=0
            for name,pp in groups.items():
                b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gb[offset:offset+len(pp)])])
                v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(pp,gu[offset:offset+len(pp)])]);offset+=len(pp)
                nb,nv=float(b.norm()),float(v.norm())
                row['groups'][name]=dict(base_gradient_norm=nb,U_gradient_norm=nv,U_over_base=nv/nb if nb else None,cosine=float(torch.dot(b,v)/(b.norm()*v.norm())) if nb and nv else None)
            self.pending=(row,{name:[p.detach().clone() for p in pp] for name,pp in groups.items()})
        return loss,dict(record,scaled=loss.detach(),U_risk=u.detach(),**{k:v.detach() for k,v in details.items()})

    def after_update(self,step):
        if self.pending is None:return
        row,before=self.pending;assert step==row['step']+1
        for name,params in parameter_groups(self.s).items():
            row['groups'][name]['actual_combined_Adam_displacement']=float(torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,before[name])]).norm())
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None
