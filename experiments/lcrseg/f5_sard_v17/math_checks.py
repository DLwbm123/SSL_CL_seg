import torch
import torch.nn.functional as F
from .model import structure_score,structured_reference
from ..f5_class_reference_v13.model import class_reference


def check(cpu):
    rng=torch.Generator().manual_seed(17017)
    f=torch.randn(2,3,5,6,generator=rng,dtype=torch.double,requires_grad=True)
    score=structure_score(f);assert not score.requires_grad
    k=f.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])
    small=F.unfold(F.pad(f.detach(),(1,1,1,1),mode='replicate'),3).reshape(2,3,9,5,6)
    dx=(small*k.flatten()[None,None,:,None,None]).sum(2)
    dy=(small*k.T.flatten()[None,None,:,None,None]).sum(2)
    v=torch.stack((dx,dy),dim=-1)
    eigen=torch.linalg.eigvalsh(torch.einsum('bchwi,bchwj->bhwij',v,v))
    large=F.unfold(F.pad(f.detach(),(7,7,7,7),mode='replicate'),15).reshape(2,3,225,5,6)
    dispersion=torch.linalg.vector_norm(large-large.mean(2,keepdim=True),dim=1).mean(1)
    exact=2*(eigen[...,1]-eigen[...,0])+eigen.prod(-1).clamp_min(0).sqrt()+dispersion
    assert torch.allclose(score,exact,atol=1e-10,rtol=1e-10)
    assert torch.count_nonzero(structure_score(torch.ones(1,2,5,5,dtype=torch.double)))==0
    impulse=torch.zeros(1,2,33,33,dtype=torch.double);impulse[:,:,16,16]=1
    localized=structure_score(impulse)
    assert localized[:,14:19,14:19].sum()>0 and localized[:,:8,:8].sum()==0
    z=torch.randn(2,3,5,6,generator=rng,dtype=torch.double,requires_grad=True)
    q=torch.randn(2,3,5,6,generator=rng,dtype=torch.double).softmax(1).requires_grad_(True)
    valid=torch.ones(2,5,6,dtype=torch.bool)
    base,counts=class_reference(z,q,valid)
    for s in (torch.ones_like(score),torch.zeros_like(score)):
        value,stats=structured_reference(z,q,valid,s)
        assert stats['class_pixels']==counts and torch.allclose(base,value,atol=1e-12,rtol=1e-12)
    s=score.detach().requires_grad_(True);value,stats=structured_reference(z,q,valid,s)
    gz,gq,gs=torch.autograd.grad(value,(z,q,s),allow_unused=True,retain_graph=True)
    assert gq is None and gs is None and torch.isfinite(gz).all()
    expected=z.softmax(1).detach()-q.detach();labels=q.detach().argmax(1)
    for c in range(3):
        mask=labels==c;expected*=torch.where(mask,score/score[mask].sum()/3,torch.ones_like(score))[:,None]
    assert torch.allclose(gz,expected,atol=1e-12,rtol=1e-12)
    empty,_=structured_reference(z,q,torch.zeros_like(valid),score)
    assert empty==0 and torch.count_nonzero(torch.autograd.grad(empty,z)[0])==0
    for bad in (score[:,:,:2],score-1e9,score*float('nan')):
        try:structured_reference(z,q,valid,bad)
        except ValueError:pass
        else:raise AssertionError('invalid score accepted')
    for bad in (f[:,:0],f*float('nan')):
        try:structure_score(bad)
        except (ValueError,FloatingPointError):pass
        else:raise AssertionError('invalid features accepted')
    # Two generated cases fit the fixed structured reference with actual optimizer calls.
    fit=torch.nn.Parameter(torch.zeros(2,3,5,6,dtype=torch.double))
    opt=torch.optim.Adam([fit],lr=.15)
    start=float(structured_reference(fit,q,valid,score)[0].detach())
    for i in range(32):
        opt.zero_grad();loss=structured_reference(fit,q,valid,score)[0];loss.backward();cpu.call(i+1);opt.step()
    end=float(structured_reference(fit,q,valid,score)[0].detach())
    assert end<start*.1,(start,end)
    return dict(exact_structure_max_error=float((score-exact).abs().max()),two_case_initial=start,two_case_final=end,
                cases='unfold/eigensolver formula; flat/impulse localization; uniform V13 and zero fallback; analytic gradient and teacher/weight detach; empty/invalid; 32-step two-case fit')
