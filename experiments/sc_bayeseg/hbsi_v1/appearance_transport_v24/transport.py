"""Fixed low-frequency amplitude transport; no learned teacher or U labels."""
import math
import torch
from torch.nn import functional as F

BETA = .05


def low_band(x):
    h,w=x.shape[-2:];radius=math.floor(min(h,w)*BETA)
    y=torch.arange(h,device=x.device)-h//2
    z=torch.arange(w,device=x.device)-w//2
    return (y[:,None].abs()<=radius)&(z[None,:].abs()<=radius)


@torch.no_grad()
def transport(x, donor):
    assert x.shape==donor.shape and x.ndim==4
    fx=torch.fft.fft2(x);fd=torch.fft.fft2(donor)
    amplitude=torch.fft.fftshift(fx.abs(),dim=(-2,-1))
    da=torch.fft.fftshift(fd.abs(),dim=(-2,-1))
    amplitude=torch.where(low_band(x),da,amplitude)
    # Keep source phase, including phase zero at zero-amplitude frequencies.
    spectrum=torch.fft.ifftshift(amplitude,dim=(-2,-1))*torch.polar(torch.ones_like(x),fx.angle())
    result=torch.fft.ifft2(spectrum)
    assert torch.isfinite(result).all()
    assert result.imag.abs().max()<1e-8*(1+result.real.abs().max())
    return result.real


class DonorData:
    """L donor uses image-only reads. U donor retains the existing image-only gate."""
    def __init__(self,base,use_u):
        self.base=base;self.use_u=use_u
        self.evidence=base.evidence;self.counts=base.counts

    def batch(self,role,size,generator):
        if role=='L':return self.base.batch(role,size,generator)
        assert role=='U'
        if self.use_u:return self.base.batch('U',size,generator)
        records=self.base.records['L'];indices=torch.randint(len(records),(size,),generator=generator).tolist()
        xs=[]
        for i in indices:
            x=self.base.read(records[i],'image').to(getattr(torch,self.base.config.dtype))[None]
            xs.append(F.interpolate(x[None],(self.base.config.height,self.base.config.width),mode='bilinear',align_corners=False)[0])
        return dict(x=torch.stack(xs),role='appearance_donor_L',stage=self.base.stage)
