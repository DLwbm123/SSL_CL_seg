"""Bounded dense checks, one registered two-case fit; no model or patient data."""
import torch
from .dense import dense_terms


def check(physical):
    torch.manual_seed(171)
    labels=torch.tensor([0,1,2,0,1,2]).view(1,1,6).expand(2,4,6).clone()
    vectors=torch.tensor([[-1.,0.,0.,0.],[.2,.8,.1,0.],[.2,-.8,.1,0.]],dtype=torch.double)
    teacher=(vectors[labels].permute(0,3,1,2)+torch.randn(2,4,4,6,dtype=torch.double)*.03).requires_grad_(True)
    student=(teacher.detach()+torch.randn_like(teacher)*.3).requires_grad_(True)
    terms,stats=dense_terms(student,teacher,labels,[0,1,1]);value=terms['dense_loss']
    assert stats['dense_leaf_regions']==3 and stats['dense_parent_regions']==1 and stats['dense_valid_pixels']==48 and value>0
    grad,tgrad=torch.autograd.grad(value,(student,teacher),allow_unused=True);assert tgrad is None and torch.isfinite(grad).all() and grad.abs().sum()>0
    same,_=dense_terms(teacher.detach(),teacher,labels,[0,1,1]);assert abs(same['dense_loss'])<1e-14
    rot=torch.tensor([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.,0.]],dtype=torch.double)
    rotate=lambda x:torch.einsum('ij,bjhw->bihw',rot,x)
    renamed=torch.tensor([2,0,1])[labels]
    for a,b,y,m in [(rotate(student),rotate(teacher),labels,[0,1,1]),(student*7,teacher*3,labels,[0,1,1]),
                    (student,teacher,renamed,[1,1,0]),(student.unsqueeze(2),teacher.unsqueeze(2),labels.unsqueeze(1),[0,1,1]),
                    (student.repeat(2,1,1,1),teacher.repeat(2,1,1,1),labels.repeat(2,1,1),[0,1,1])]:
        other,_=dense_terms(a,b,y,m);assert torch.allclose(value,other['dense_loss'],atol=1e-12,rtol=1e-12)
    independent,_=dense_terms(rotate(teacher.detach()),teacher,labels,[0,1,1]);assert independent['dense_loss']>.1
    four=torch.tensor([0,1,2,3,0,1]).view(1,1,6).expand(2,4,6).clone()
    alternate,info=dense_terms(student,teacher,four,[0,0,1,1]);assert info['dense_leaf_regions']==4 and info['dense_parent_regions']==2
    # Increasing only background area cannot dominate balanced leaf/eligible-parent groups.
    bg=labels==0;si=[];ti=[];yi=[]
    for b in range(2):
        si.append(torch.cat((student[b].flatten(1),student[b].flatten(1)[:,bg[b].flatten()].repeat(1,3)),1))
        ti.append(torch.cat((teacher[b].flatten(1),teacher[b].flatten(1)[:,bg[b].flatten()].repeat(1,3)),1))
        yi.append(torch.cat((labels[b].flatten(),labels[b].flatten()[bg[b].flatten()].repeat(3)),0))
    balanced,_=dense_terms(torch.stack(si).unsqueeze(2),torch.stack(ti).unsqueeze(2),torch.stack(yi).unsqueeze(1),[0,1,1])
    assert torch.allclose(value,balanced['dense_loss'],atol=1e-12,rtol=1e-12)
    # Fixed V24 within-class permutation: pooling misses it, corresponding-pixel loss detects it.
    yy=torch.tensor([0,0,1,1,2,2]).reshape(1,1,6)
    tt=torch.tensor([[-1.,0.],[-.8,.6],[1.,0.],[.8,.6],[0.,1.],[.6,.8]],dtype=torch.double).T.reshape(1,2,1,6)
    permuted,_=dense_terms(tt[:,:,:,[1,0,3,2,5,4]],tt,yy,[0,1,1]);assert abs(permuted['dense_loss']-.2)<1e-12
    eps=1e-6;hi=student.detach().clone();lo=hi.clone();hi[0,0,0,0]+=eps;lo[0,0,0,0]-=eps
    fd=(dense_terms(hi,teacher,labels,[0,1,1])[0]['dense_loss']-dense_terms(lo,teacher,labels,[0,1,1])[0]['dense_loss'])/(2*eps)
    again,_=dense_terms(student,teacher,labels,[0,1,1]);fdgrad,=torch.autograd.grad(again['dense_loss'],student)
    assert torch.allclose(fdgrad[0,0,0,0],fd,atol=1e-8,rtol=1e-5)
    for a,b,y in [(student,teacher,torch.full_like(labels,255)),(torch.zeros_like(student,requires_grad=True),teacher,labels),
                  (student,torch.zeros_like(teacher),labels)]:
        v,_=dense_terms(a,b,y,[0,1,1]);g,=torch.autograd.grad(v['dense_loss'],a)
        assert v['dense_loss']==0 and torch.count_nonzero(g)==0 and torch.isfinite(g).all()
    one,info=dense_terms(student,teacher,torch.zeros_like(labels),[0,1,1]);assert info['dense_leaf_regions']==1 and info['dense_parent_regions']==0 and one['dense_loss']>0
    ignored=labels.clone();ignored[:,:,:2]=255
    v,_=dense_terms(student,teacher,ignored,[0,1,1]);g,=torch.autograd.grad(v['dense_loss'],student)
    assert torch.count_nonzero(g[:,:,:,:2])==0
    changed=student.detach().clone();changed[:,:,:,:2]+=3
    vv,_=dense_terms(changed,teacher,ignored,[0,1,1]);assert torch.equal(v['dense_loss'],vv['dense_loss'])
    invalid=0
    for a,b,y,m in [(student,teacher,labels,[]),(student,teacher,labels,[0,2,2]),(student,teacher,labels,[0,True,1]),
                     (student,teacher,labels.float(),[0,1,1]),(student,teacher,labels+3,[0,1,1]),
                     (student[:,:,:,:3],teacher,labels,[0,1,1]),(student,teacher*float('nan'),labels,[0,1,1]),
                     (torch.full_like(student,1e308),teacher,labels,[0,1,1]),(student.half(),teacher.half(),labels,[0,1,1]),
                     (student,teacher[:,:3],labels,[0,1,1])]:
        try:dense_terms(a,b,y,m)
        except (ValueError,FloatingPointError):invalid+=1
        else:raise AssertionError('invalid dense input accepted')
    assert invalid==10
    fit=torch.nn.Parameter(student.detach().clone());opt=torch.optim.Adam([fit],lr=.05)
    start=float(dense_terms(fit,teacher,labels,[0,1,1])[0]['dense_loss'].detach())
    for i in range(32):
        opt.zero_grad();loss=dense_terms(fit,teacher,labels,[0,1,1])[0]['dense_loss'];loss.backward()
        assert torch.isfinite(fit.grad).all();physical.call(i+1);opt.step()
    end=float(dense_terms(fit,teacher,labels,[0,1,1])[0]['dense_loss'].detach());assert end<start*.8
    return dict(status='PASS',cpu_optimizer_updates=32,extra_cpu_gradient_vjps=6,finite_difference_error=float(abs(fdgrad[0,0,0,0]-fd)),fit_initial=start,fit_final=end,invalid_rejections=invalid,
                checks=['teacher detach; identity zero; common rotation invariant but independent teacher/student rotation detected; positive scaling; leaf rename/4leaf/2D3D/batch replication',
                        'live per-position normalization finite difference; zero/all-ignore unsupported gradients zero; singleton parent excluded but leaf retained',
                        'ignored pixels gradient zero; strict invalid/nonfinite/channel/precision rejection; background area replication and within-class spatial permutation; bounded two-case dense fit'])
