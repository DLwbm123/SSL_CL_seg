import torch
from .model import care_reference


def check(cpu):
    # Each class has a forward and a reverse example; classes have unequal pixel counts.
    p=torch.tensor([[.8,.1,.1],[.8,.1,.1],[.1,.8,.1],[.1,.8,.1],[.1,.1,.8],[.1,.1,.8],[.8,.1,.1]],dtype=torch.double)
    q=torch.tensor([[.6,.2,.2],[.9,.05,.05],[.2,.6,.2],[.05,.9,.05],[.2,.2,.6],[.05,.05,.9],[.6,.2,.2]],dtype=torch.double)
    teacher=p.log().T[None,:,None,:].clone().requires_grad_(True)
    z=q.log().T[None,:,None,:].clone().requires_grad_(True);valid=torch.ones(1,1,7,dtype=torch.bool)
    value,stats=care_reference(z,teacher,valid)
    gz,gt=torch.autograd.grad(value,(z,teacher),allow_unused=True);assert gt is None
    entropy=lambda x:-(x*x.log()).sum(1)
    forward=entropy(p)<=entropy(q);logratio=q.log()-p.log();reverse=(q*logratio).sum(1)
    per=torch.where(forward,(p*(p.log()-q.log())).sum(1),reverse)
    expected=torch.where(forward[:,None],q-p,q*(logratio-reverse[:,None]))
    terms=[]
    for c in range(3):
        selected=p.argmax(1)==c;expected[selected]/=int(selected.sum())*3;terms.append(per[selected].mean())
    assert torch.allclose(value,torch.stack(terms).mean(),atol=1e-12,rtol=1e-12)
    error=float((gz[0,:,0].T-expected).abs().max());assert error<1e-12
    assert stats['forward_pixels']==4 and stats['reverse_pixels']==3 and stats['class_pixels']==[3,2,2]
    same,ss=care_reference(teacher,teacher,valid);assert abs(float(same))<1e-14 and ss['reverse_pixels']==0
    empty,_=care_reference(z,teacher,torch.zeros_like(valid));assert empty==0 and torch.count_nonzero(torch.autograd.grad(empty,z)[0])==0
    extreme=torch.tensor([1000.,-1000.,-1000.],dtype=torch.double).view(1,3,1,1)
    ev,es=care_reference(extreme,torch.tensor([1.,0.,0.],dtype=torch.double).view(1,3,1,1),valid[:,:,:1]);assert torch.isfinite(ev) and es['reverse_pixels']==1
    floored,fs=care_reference(torch.zeros_like(extreme),extreme,valid[:,:,:1]);assert torch.isfinite(floored) and fs['teacher_log_floor_values']==2
    for bad_z,bad_t,bad_m in [(z[:,:2],teacher,valid),(z,teacher*float('nan'),valid),(z,teacher,valid.float()),(z,teacher,valid[:,:,:2])]:
        try:care_reference(bad_z,bad_t,bad_m)
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid input accepted')
    # Two generated distributions fit while the hard branch can change; no native/patient probe.
    fit=torch.nn.Parameter(torch.zeros(2,3,1,1,dtype=torch.double));target=torch.tensor([[.8,.1,.1],[.1,.8,.1]],dtype=torch.double).log()[:,:,None,None]
    mask=torch.ones(2,1,1,dtype=torch.bool);opt=torch.optim.Adam([fit],lr=.15)
    start=float(care_reference(fit,target,mask)[0].detach())
    for i in range(32):
        opt.zero_grad();loss=care_reference(fit,target,mask)[0];loss.backward();cpu.call(i+1);opt.step()
    end=float(care_reference(fit,target,mask)[0].detach());assert end<start*.1,(start,end)
    return dict(analytic_gradient_max_error=error,branch_stats=stats,two_case_initial=start,two_case_final=end,cases='forward/reverse analytic gradients; unequal class supports; detached teacher/gate; equality forward; empty; extreme/floor; four invalid inputs; two-case fit')
