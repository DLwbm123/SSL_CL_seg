"""Bounded hierarchy RKD checks; no model or patient data."""
import torch
from .relation import relation_terms


def check(physical):
    torch.manual_seed(171)
    labels=torch.tensor([0,1,2,0,1,2]).view(1,1,6).expand(2,4,6).clone()
    vectors=torch.tensor([[-1.,0.,0.,0.],[.2,.8,.1,0.],[.2,-.8,.1,0.]],dtype=torch.double)
    teacher=(vectors[labels].permute(0,3,1,2)+torch.randn(2,4,4,6,dtype=torch.double)*.03).requires_grad_(True)
    student=(teacher.detach()+torch.randn_like(teacher)*.3).requires_grad_(True)
    terms,stats=relation_terms(student,teacher,labels,[0,1,1]);value=sum(terms.values())
    assert stats['relation_edges']==4 and stats['relation_triplets']==1 and value>0
    grad,tgrad=torch.autograd.grad(value,(student,teacher),allow_unused=True);assert tgrad is None and torch.isfinite(grad).all() and grad.abs().sum()>0
    same,_=relation_terms(teacher.detach(),teacher,labels,[0,1,1]);assert sum(same.values())==0
    rot=torch.tensor([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.,0.]],dtype=torch.double)
    rotated=torch.einsum('ij,bjhw->bihw',rot,student.detach())
    renamed=torch.tensor([2,0,1])[labels]
    for a,b,y,m in [(rotated,teacher,labels,[0,1,1]),(student*7,teacher*3,labels,[0,1,1]),
                    (student,teacher,renamed,[1,1,0]),(student.unsqueeze(2),teacher.unsqueeze(2),labels.unsqueeze(1),[0,1,1]),
                    (student.repeat(2,1,1,1),teacher.repeat(2,1,1,1),labels.repeat(2,1,1),[0,1,1])]:
        other,_=relation_terms(a,b,y,m)
        for k in terms:assert torch.allclose(terms[k],other[k],atol=1e-12,rtol=1e-12)
    four=torch.tensor([0,1,2,3,0,1]).view(1,1,6).expand(2,4,6).clone()
    alternate,info=relation_terms(student,teacher,four,[0,0,1,1]);assert info['relation_edges']==7 and info['relation_triplets']==2
    eps=1e-6;hi=student.detach().clone();lo=hi.clone();hi[0,0,0,0]+=eps;lo[0,0,0,0]-=eps
    fd=(sum(relation_terms(hi,teacher,labels,[0,1,1])[0].values())-sum(relation_terms(lo,teacher,labels,[0,1,1])[0].values()))/(2*eps)
    # Second VJP checks the complete live prototype/normalization computation.
    again,_=relation_terms(student,teacher,labels,[0,1,1]);fdgrad,=torch.autograd.grad(sum(again.values()),student)
    assert torch.allclose(fdgrad[0,0,0,0],fd,atol=1e-8,rtol=1e-5)
    for a,y in [(student,torch.full_like(labels,255)),(torch.zeros_like(student,requires_grad=True),labels),
                (student,torch.ones_like(labels)),(torch.ones_like(student,requires_grad=True),labels)]:
        v,_=relation_terms(a,teacher,y,[0,1,1]);g,=torch.autograd.grad(sum(v.values()),a)
        assert torch.count_nonzero(g)==0 and torch.isfinite(g).all()
    ignored=labels.clone();ignored[:,:,:2]=255
    v,_=relation_terms(student,teacher,ignored,[0,1,1])
    # Ignored feature perturbations cannot change any mean/relation.
    changed=student.detach().clone();changed[:,:,:,:2]+=3
    vv,_=relation_terms(changed,teacher,ignored,[0,1,1])
    for k in v:assert torch.equal(v[k],vv[k])
    invalid=0
    for a,b,y,m in [(student,teacher,labels,[]),(student,teacher,labels,[0,2,2]),(student,teacher,labels,[0,True,1]),
                     (student,teacher,labels.float(),[0,1,1]),(student,teacher,labels+3,[0,1,1]),
                     (student[:,:,:,:3],teacher,labels,[0,1,1]),(student,teacher*float('nan'),labels,[0,1,1]),
                     (torch.full_like(student,1e308),teacher,labels,[0,1,1]),(student.half(),teacher.half(),labels,[0,1,1]),
                     (student,teacher,labels[:1],[0,1,1])]:
        try:relation_terms(a,b,y,m)
        except (ValueError,FloatingPointError):invalid+=1
        else:raise AssertionError('invalid relation input accepted')
    assert invalid==10
    fit=torch.nn.Parameter(student.detach().clone());opt=torch.optim.Adam([fit],lr=.05)
    start=float(sum(relation_terms(fit,teacher,labels,[0,1,1])[0].values()).detach())
    for i in range(32):
        opt.zero_grad();v,_=relation_terms(fit,teacher,labels,[0,1,1]);loss=sum(v.values());loss.backward()
        assert torch.isfinite(fit.grad).all();physical.call(i+1);opt.step()
    end=float(sum(relation_terms(fit,teacher,labels,[0,1,1])[0].values()).detach());assert end<start*.8
    return dict(status='PASS',cpu_optimizer_updates=32,extra_cpu_gradient_vjps=6,finite_difference_error=float(abs(fdgrad[0,0,0,0]-fd)),fit_initial=start,fit_final=end,invalid_rejections=invalid,
                checks=['teacher detach; identity zero; student orthogonal rotation/positive scaling; leaf renaming/4leaf/2D3D/batch replication',
                        'live prototypes and student distance mean fixed-teacher finite difference; all-ignore/zero/singleton/duplicate zero gradients',
                        'ignored features excluded; strict invalid/nonfinite/unsupported precision rejection; bounded two-case relation fit'])
