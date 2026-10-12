import torch
from .model import block_halfspace


def check(cpu):
    blocks=dict(A=[0],B=[1],R=[2]);dtype=torch.double
    cases=[([-1.,2.,0.],[1.,1.,0.],[0.,2.,0.]),([-2.,1.,0.],[1.,1.,0.],[0.,1.,0.]),([1.,2.,3.],[1.,1.,1.],[1.,2.,3.]),([-1.,2.,3.],[0.,0.,0.],[-1.,2.,3.])]
    for g,h,wanted in cases:
        tensors=lambda values:[torch.tensor([v],dtype=dtype) for v in values]
        out,stats=block_halfspace(tensors(g),tensors(h),blocks)
        assert torch.allclose(torch.cat(out),torch.tensor(wanted,dtype=dtype),atol=1e-12,rtol=1e-12)
        assert all(x['dot_after']>=-1e-12 for x in stats['parameter_blocks'].values())
    out,hidden=block_halfspace([torch.tensor([-1.,1.],dtype=dtype),torch.tensor([2.,1.],dtype=dtype),torch.zeros(2,dtype=dtype)],[torch.tensor([1.,0.],dtype=dtype)]*3,blocks)
    assert hidden['dot_before']>0 and hidden['v13_block_violations']==1
    rng=torch.Generator().manual_seed(20020);max_error=0.
    for _ in range(64):
        g=[torch.randn(5,generator=rng,dtype=dtype) for _ in range(3)];h=[torch.randn(5,generator=rng,dtype=dtype) for _ in range(3)]
        out,stats=block_halfspace(g,h,blocks)
        for a,b,c in zip(g,h,out):
            dot=a@b;expected=a-torch.minimum(dot,torch.zeros_like(dot))/(b@b)*b
            max_error=max(max_error,float((expected-c).abs().max()));assert torch.allclose(c,expected,atol=1e-12,rtol=1e-12)
            assert c@b>=-1e-12
        scaled,_=block_halfspace(g,[h[0]*.1,h[1]*10,h[2]*3],blocks)
        assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(out,scaled))
    for bad in [dict(A=[0],B=[0],R=[2]),dict(A=[0],B=[1],R=[]),dict(A=[0],B=[1],R=[3])]:
        try:block_halfspace(g,h,bad)
        except ValueError:pass
        else:raise AssertionError('invalid partition accepted')
    for bad_g,bad_h in [([torch.full((5,),float('nan'))]*3,h),(g,[torch.zeros(4)]*3)]:
        try:block_halfspace(bad_g,bad_h,blocks)
        except (ValueError,FloatingPointError):pass
        else:raise AssertionError('invalid gradient accepted')
    fit=torch.nn.Parameter(torch.tensor([[.3,0.],[-.3,0.]],dtype=dtype));opt=torch.optim.SGD([fit],lr=.2);target=torch.tensor([[1.,1.],[-1.,1.]],dtype=dtype)
    initial=fit[:,0].detach().clone();start=float((fit-target).square().sum().detach())
    for i in range(32):
        opt.zero_grad();loss=.5*(fit-target).square().sum();grad,=torch.autograd.grad(loss,fit);h=torch.zeros_like(fit);h[:,0]=fit[:,0].detach()
        out,_=block_halfspace([grad[0],grad[1],torch.zeros(1,dtype=dtype)],[h[0],h[1],torch.zeros(1,dtype=dtype)],blocks)
        fit.grad=torch.stack(out[:2]);cpu.call(i+1);opt.step()
    end=float((fit-target).square().sum().detach());assert end<start*.6 and torch.allclose(fit[:,0],initial,atol=1e-12,rtol=1e-12)
    return dict(analytic_cases=4,random_cases=64,invalid_cases=5,max_closed_form_error=max_error,hidden_conflict_case=hidden,two_case_initial=start,two_case_final=end,protected_coordinate_drift=float((fit[:,0]-initial).abs().max()))
