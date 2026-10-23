"""Asymmetric whole-vector projection of U gradients against the full base risk."""
import json
import torch
from ..semantic_alignment_v23.method import Objective as BaseObjective
from ..posterior_teacher_v25.method import teacher_risk,student_probability


def project(base,u):
    norm2=base.square().sum();dot=torch.dot(base,u)
    coefficient=dot.clamp_max(0)/norm2 if float(norm2)>0 else dot.new_zeros(())
    result=u-coefficient*base
    assert torch.isfinite(result).all()
    assert float(torch.dot(base,result))>=-1e-10*(1+float(base.norm()*u.norm()))
    return result,coefficient


def fixture():
    b=torch.tensor([2.,0.],dtype=torch.float64)
    u=torch.tensor([-3.,4.],dtype=torch.float64)
    p,c=project(b,u);assert torch.equal(p,torch.tensor([0.,4.],dtype=torch.float64))
    v=torch.tensor([3.,4.],dtype=torch.float64);assert torch.equal(project(b,v)[0],v)
    assert torch.equal(project(torch.zeros_like(b),u)[0],u)
    assert torch.equal(project(b,torch.zeros_like(b))[0],torch.zeros_like(b))
    return dict(conflict_removed=True,aligned_unchanged=True,zero_base_preserves_U=True,zero_U_finite=True,fixture_reverse_calls=0)


class Objective:
    def __init__(self,state,refiner,arm='P',audit_path=None):
        assert arm=='P';self.s=state;self.counter=refiner.counter
        self.base=BaseObjective(state,refiner,False);self.audit_path=audit_path;self.pending=None
        self.expected=None

    def estimate(self,L,U,counts,generator,sampling_weights=None):
        s=self.s;s.check_batch(U,'U')
        base,record=self.base.estimate(L,None,counts,generator,sampling_weights)
        target=U['teacher_mean'].to(L['x'].device)
        student=student_probability(s,U['x']);u,details=teacher_risk(target,student,torch.ones_like(target[:,0]))
        params=[p for group in s.optimizer.param_groups for p in group['params'] if p.requires_grad]
        gb=torch.autograd.grad(base,params,allow_unused=True)
        gu=torch.autograd.grad(u,params,allow_unused=True)
        self.counter.add('projection_model_reverse_calls',2)
        self.counter.add('local_mc_samples',s.config.global_mc*s.config.local_mc*len(U['x']))
        self.counter.add('global_mc_samples',s.config.global_mc)
        b=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,gb)])
        v=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(params,gu)])
        projected,coefficient=project(b,v);combined=(b+projected).detach();self.expected=combined
        # Forward value reports the original risks; backward supplies the frozen projected vector.
        loss=(base+u).detach();offset=0
        for p,g,h in zip(params,gb,gu):
            desired=combined[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
            if g is not None or h is not None:loss=loss+((p-p.detach())*desired).sum()
        assert torch.isfinite(loss)
        if self.audit_path is not None and s.steps in (1600,1799,1999):
            self.pending=(s.steps,params,[p.detach().clone() for p in params],b.detach(),v.detach(),projected.detach())
        return loss,dict(record,scaled=loss.detach(),U_risk=u.detach(),base_objective=base.detach(),projection_applied=(coefficient<0).double(),projection_coefficient=coefficient.detach(),base_gradient_norm=b.norm(),U_gradient_norm=v.norm(),projected_U_gradient_norm=projected.norm(),base_dot_U=torch.dot(b,v),base_dot_projected_U=torch.dot(b,projected),**{k:t.detach() for k,t in details.items()})

    def after_update(self,step):
        if self.pending is None:return
        start,params,before,b,u,p=self.pending;assert step==start+1
        dx=torch.cat([(value.detach()-old).reshape(-1) for value,old in zip(params,before)])
        row=dict(step=start,arm='P',extra_reverse_calls=0,combined_Adam_displacement_norm=float(dx.norm()),base_gradient_dot_actual_displacement=float(torch.dot(b,dx)),U_gradient_dot_actual_displacement=float(torch.dot(u,dx)),projected_U_dot_actual_displacement=float(torch.dot(p,dx)))
        with self.audit_path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending=None
