"""Zero-update check for the observed CUDA pooling-backward failure."""
from pathlib import Path
import torch
from .module import DeterministicPool
from ..main_style_ssl_v1.runtime import write

def main(cfg):
    torch.manual_seed(198)
    x=torch.randn(2,128,48,48,device='cuda',requires_grad=True)
    y=DeterministicPool.apply(x);reference=torch.nn.functional.adaptive_avg_pool2d(x,(8,8))
    assert torch.equal(y,reference)
    grad=torch.randn_like(y)
    actual=torch.autograd.grad(y,x,grad,create_graph=True)[0]
    expected=grad.repeat_interleave(6,-2).repeat_interleave(6,-1)/36
    assert torch.equal(actual,expected)
    first=torch.autograd.grad(y.square().sum(),x,create_graph=True)[0]
    second=torch.autograd.grad(first.square().sum(),x)[0]
    assert torch.isfinite(second).all() and second.norm()>0
    small=torch.randn(1,1,8,16,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(DeterministicPool.apply,(small,),fast_mode=True)
    assert torch.autograd.gradgradcheck(DeterministicPool.apply,(small,),fast_mode=True)
    write(Path(cfg['root'])/'POOL_REPAIR_CHECK.json',dict(status='PASS',optimizer_calls=0,functional_native_updates=0,patient_access=False,CUDA_forward_exact=True,CUDA_adjoint_exact=True,second_derivative_finite=True,CPU_gradcheck=True,CPU_gradgradcheck=True,determinism_enabled=torch.are_deterministic_algorithms_enabled()))
