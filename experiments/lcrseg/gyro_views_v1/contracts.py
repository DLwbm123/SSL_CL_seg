from pathlib import Path
import torch
from . import core
from ..gyro_native_v1.module import frame,spectral,self_check
from ..main_style_ssl_v1.runtime import write
from ..gyro_autonomy_12h.space_guard import SpaceGuard


def main(cfg):
    assert cfg['views']==4 and self_check()['status']=='PASS'
    fs=torch.stack([frame(90163),frame(90164)]);ids=torch.tensor([1]*16+[0]*8)
    z=torch.randn(24,4,128,generator=torch.Generator().manual_seed(20),requires_grad=True)
    for beta in [0.,1/3]:
        assert torch.equal(core.spectral_views(z[:,:2],fs,ids,beta),spectral(z[:,:2],fs,ids,beta))
        a=core.spectral_views(z,fs,ids,beta);b=core.spectral_views(z[:,[3,2,1,0]],fs,ids,beta)
        assert torch.allclose(a,b,atol=1e-6)
        assert torch.isfinite(torch.autograd.grad(a,z)[0]).all()
    assert abs(.45*(1-1/3)-.3)<1e-12
    SpaceGuard(cfg['storage_authorization']).check(500_000_000)
    write(Path(cfg['root'])/'SYNTHETIC_CHECKS.json',dict(status='PASS',optimizer_calls=0,V2_equation_exact=True,V4_permutation_invariant=True,V4_finite_gradient=True))
