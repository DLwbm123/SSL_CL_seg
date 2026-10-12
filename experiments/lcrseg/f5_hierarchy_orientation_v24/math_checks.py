"""Bounded orientation checks, one registered two-case fit; no model or patient data."""
import torch
from .orientation import orientation_terms


def check(physical):
    torch.manual_seed(171)
    labels=torch.tensor([0,1,2,0,1,2]).view(1,1,6).expand(2,4,6).clone()
    vectors=torch.tensor([[-1.,0.,0.,0.],[.2,.8,.1,0.],[.2,-.8,.1,0.]],dtype=torch.double)
    teacher=(vectors[labels].permute(0,3,1,2)+torch.randn(2,4,4,6,dtype=torch.double)*.03).requires_grad_(True)
    student=(teacher.detach()+torch.randn_like(teacher)*.3).requires_grad_(True)
    terms,stats=orientation_terms(student,teacher,labels,[0,1,1]);value=terms['orientation_loss']
    assert stats['orientation_leaf_nodes']==3 and stats['orientation_parent_nodes']==1 and value>0
    grad,tgrad=torch.autograd.grad(value,(student,teacher),allow_unused=True);assert tgrad is None and torch.isfinite(grad).all() and grad.abs().sum()>0
    same,_=orientation_terms(teacher.detach(),teacher,labels,[0,1,1]);assert abs(same['orientation_loss'])<1e-14
    rot=torch.tensor([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.,0.]],dtype=torch.double)
    rotate=lambda x:torch.einsum('ij,bjhw->bihw',rot,x)
    renamed=torch.tensor([2,0,1])[labels]
    for a,b,y,m in [(rotate(student),rotate(teacher),labels,[0,1,1]),(student*7,teacher*3,labels,[0,1,1]),
                    (student,teacher,renamed,[1,1,0]),(student.unsqueeze(2),teacher.unsqueeze(2),labels.unsqueeze(1),[0,1,1]),
                    (student.repeat(2,1,1,1),teacher.repeat(2,1,1,1),labels.repeat(2,1,1),[0,1,1])]:
        other,_=orientation_terms(a,b,y,m);assert torch.allclose(value,other['orientation_loss'],atol=1e-12,rtol=1e-12)
    independent,_=orientation_terms(rotate(teacher.detach()),teacher,labels,[0,1,1]);assert independent['orientation_loss']>.1
    four=torch.tensor([0,1,2,3,0,1]).view(1,1,6).expand(2,4,6).clone()
    alternate,info=orientation_terms(student,teacher,four,[0,0,1,1]);assert info['orientation_leaf_nodes']==4 and info['orientation_parent_nodes']==2
    eps=1e-6;hi=student.detach().clone();lo=hi.clone();hi[0,0,0,0]+=eps;lo[0,0,0,0]-=eps
    fd=(orientation_terms(hi,teacher,labels,[0,1,1])[0]['orientation_loss']-orientation_terms(lo,teacher,labels,[0,1,1])[0]['orientation_loss'])/(2*eps)
    again,_=orientation_terms(student,teacher,labels,[0,1,1]);fdgrad,=torch.autograd.grad(again['orientation_loss'],student)
    assert torch.allclose(fdgrad[0,0,0,0],fd,atol=1e-8,rtol=1e-5)
    for a,b,y in [(student,teacher,torch.full_like(labels,255)),(torch.zeros_like(student,requires_grad=True),teacher,labels),
                  (student,torch.zeros_like(teacher),labels)]:
        v,_=orientation_terms(a,b,y,[0,1,1]);g,=torch.autograd.grad(v['orientation_loss'],a)
        assert v['orientation_loss']==0 and torch.count_nonzero(g)==0 and torch.isfinite(g).all()
    one,info=orientation_terms(student,teacher,torch.zeros_like(labels),[0,1,1]);assert info['orientation_leaf_nodes']==1 and info['orientation_parent_nodes']==0 and one['orientation_loss']>0
    ignored=labels.clone();ignored[:,:,:2]=255
    v,_=orientation_terms(student,teacher,ignored,[0,1,1]);g,=torch.autograd.grad(v['orientation_loss'],student)
    assert torch.count_nonzero(g[:,:,:,:2])==0
    changed=student.detach().clone();changed[:,:,:,:2]+=3
    vv,_=orientation_terms(changed,teacher,ignored,[0,1,1]);assert torch.equal(v['orientation_loss'],vv['orientation_loss'])
    invalid=0
    for a,b,y,m in [(student,teacher,labels,[]),(student,teacher,labels,[0,2,2]),(student,teacher,labels,[0,True,1]),
                     (student,teacher,labels.float(),[0,1,1]),(student,teacher,labels+3,[0,1,1]),
                     (student[:,:,:,:3],teacher,labels,[0,1,1]),(student,teacher*float('nan'),labels,[0,1,1]),
                     (torch.full_like(student,1e308),teacher,labels,[0,1,1]),(student.half(),teacher.half(),labels,[0,1,1]),
                     (student,teacher[:,:3],labels,[0,1,1])]:
        try:orientation_terms(a,b,y,m)
        except (ValueError,FloatingPointError):invalid+=1
        else:raise AssertionError('invalid orientation input accepted')
    assert invalid==10
    fit=torch.nn.Parameter(student.detach().clone());opt=torch.optim.Adam([fit],lr=.05)
    start=float(orientation_terms(fit,teacher,labels,[0,1,1])[0]['orientation_loss'].detach())
    for i in range(32):
        opt.zero_grad();loss=orientation_terms(fit,teacher,labels,[0,1,1])[0]['orientation_loss'];loss.backward()
        assert torch.isfinite(fit.grad).all();physical.call(i+1);opt.step()
    end=float(orientation_terms(fit,teacher,labels,[0,1,1])[0]['orientation_loss'].detach());assert end<start*.8
    return dict(status='PASS',cpu_optimizer_updates=32,extra_cpu_gradient_vjps=6,finite_difference_error=float(abs(fdgrad[0,0,0,0]-fd)),fit_initial=start,fit_final=end,invalid_rejections=invalid,
                checks=['teacher detach; identity zero; common rotation invariant but independent teacher/student rotation detected; positive scaling; leaf rename/4leaf/2D3D/batch replication',
                        'live normalized prototype finite difference; zero/all-ignore unsupported gradients zero; singleton parent excluded but leaf retained',
                        'ignored pixels gradient zero; strict invalid/nonfinite/channel/precision rejection; bounded two-case orientation fit'])
