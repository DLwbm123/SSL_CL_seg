"""CPU-only mathematical checks; zero optimizer calls and no patient IO."""
import copy,json
import torch
from experiments.lcrseg.f5_cayley_v8.module import CayleyAdapter
from experiments.lcrseg.five_frameworks_v1.kernels import StageSubspaceAdapter

def checks():
    rng=torch.Generator().manual_seed(163);q=torch.linalg.qr(torch.randn(16,8,dtype=torch.float64,generator=rng)).Q
    a=CayleyAdapter(q);x=torch.randn(2,16,3,3,dtype=torch.float64,generator=rng);assert torch.allclose(a(x),x,atol=1e-12,rtol=1e-12)
    with torch.no_grad():a.r.copy_(torch.randn(8,8,dtype=torch.float64,generator=rng)*.2)
    f=a.effective();eye=torch.eye(16,dtype=torch.float64);assert torch.allclose(f.T@f,eye,atol=1e-11,rtol=1e-11)
    out,before=a(x,return_pre_previous=True);assert torch.allclose(out,torch.einsum('ij,bjhw->bihw',f,x),atol=1e-11,rtol=1e-11)
    assert torch.allclose(a(x).square().sum(1),x.square().sum(1),atol=1e-11,rtol=1e-11)
    b=CayleyAdapter(q,a.seal());b.r.data.copy_(a.r.detach()/2);out2,before2=b(x,return_pre_previous=True);assert torch.allclose(out2,torch.einsum('ij,bjhw->bihw',a.seal(),before2),atol=1e-11,rtol=1e-11);combined=b.seal();assert torch.allclose(combined.T@combined,eye,atol=1e-11,rtol=1e-11)
    t=copy.deepcopy(a).requires_grad_(False);t.r.zero_();old=t.r.clone();a.update_teacher(t,decay=.5);assert torch.equal(t.r,.5*old+.5*a.r.detach());assert torch.allclose(t.effective().T@t.effective(),eye,atol=1e-11,rtol=1e-11)
    state=copy.deepcopy(b.state_dict());c=CayleyAdapter(q,a.seal());c.load_state_dict(state);assert torch.equal(c(x),b(x))
    linear=StageSubspaceAdapter(q);disabled=CayleyAdapter(q,cayley_enabled=False);linear.r.data.copy_(a.r.detach());disabled.load_state_dict(linear.state_dict());assert torch.equal(disabled(x),linear(x));teacher=copy.deepcopy(disabled).requires_grad_(False);disabled.update_teacher(teacher);assert torch.allclose(teacher.r,disabled.r.detach())
    r=torch.randn(3,3,dtype=torch.float64,generator=rng).requires_grad_()
    def rotation(z):s=(z-z.T)/2;i=torch.eye(3,dtype=z.dtype);return torch.linalg.solve(i-s,i+s)
    assert torch.autograd.gradcheck(rotation,(r,),eps=1e-6,atol=1e-5,rtol=1e-4)
    objective=a(x).square()[0,0].sum();objective.backward();assert a.r.grad is not None and a.r.grad.norm()>0 and torch.allclose(a.r.grad+a.r.grad.T,torch.zeros_like(a.r),atol=1e-11,rtol=1e-11)
    assert not a.q.requires_grad and not a.previous.requires_grad and a.previous.grad is None and all(v.grad is None for v in t.parameters())
    try:CayleyAdapter(q,torch.eye(16,dtype=torch.float64)*2)
    except ValueError:pass
    else:raise AssertionError('nonorthogonal old prefix accepted')
    a.r.data.fill_(float('nan'))
    try:a.current_matrix()
    except FloatingPointError:pass
    else:raise AssertionError('nonfinite Cayley parameter accepted')
    return dict(status='MATH_PASS_ONLY',cpu_optimizer_updates=0,synthetic_cuda_updates=0,real_patient_reads=0,native_training_qualified=False,checks=['identity initialization','orthogonality and feature norm','own two-stage seal composition','raw-generator teacher EMA','state restoration and linear-disabled parity','finite-difference gradient and skew tangent','frozen Q/previous/teacher','reject nonorthogonal prefix and nonfinite parameter'])
if __name__=='__main__':print(json.dumps(checks()))
