"""Analytic, endpoint-preserving acquisition-like transforms; no spatial warps."""
import torch
from torch import nn
from torch.nn import functional as F
from ..five_frameworks_v1.recipes import generator

def knots(v):
    widths=.125+F.softmax(v,1)
    yy=torch.cat((torch.full_like(v[:,:1],-1),-1+widths.cumsum(1)),1)
    # Endpoints are part of the definition, not a learned extrapolation.
    return torch.cat((yy[:,:1],yy[:,1:-1],torch.ones_like(yy[:,:1])),1)
def raw_spline(x,v,inverse=False):
    y=knots(v);flat=x.flatten(1)
    if inverse:
        i=(torch.searchsorted(y.contiguous(),flat.contiguous(),right=True)-1).clamp(0,7)
    else:i=torch.floor((flat+1)*4).long().clamp(0,7)
    lo=y.gather(1,i);hi=y.gather(1,i+1);xi=-1+.25*i.to(x)
    out=xi+.25*(flat-lo)/(hi-lo) if inverse else lo+(flat-xi)*(hi-lo)/.25
    return out.reshape_as(x)
def spline(x,v,inverse=False):return x+(raw_spline(x,v,inverse)-raw_spline(x,torch.zeros_like(v),inverse))
def pool4(x):
    h,w=x.shape[-2:];return torch.stack([torch.stack([x[:,:,i*h//4:((i+1)*h+3)//4,j*w//4:((j+1)*w+3)//4].mean((-2,-1)) for j in range(4)],-1) for i in range(4)],-2)
def field(a,size):
    t=F.interpolate(a,size=size,mode='bilinear',align_corners=False).tanh()
    return .05*(t-t.mean((2,3),keepdim=True))
def bias_inverse(x,b):return 2*(x-b)/(1+torch.sqrt(1-4*b*(x-b)))

def transform(x,step,stage,diagnostic=False,calls=None):
    g=generator(20271702 if diagnostic else 20271701,1,0 if stage=='source' else 1,step,'ACQUISITION_VIEW')
    v=(torch.rand(len(x),8,generator=g)*2-1).to(x);a=(torch.rand(len(x),1,4,4,generator=g)*2-1).to(x)
    if calls:
        calls.extra('transform_samples',len(x));calls.extra('spline_samples',2*len(x));calls.extra('bias_samples',len(x))
    m=spline(x,v);b=field(a,x.shape[-2:]);out=m+b*(1-m.square())
    assert torch.isfinite(out).all() and float(out.detach().abs().max())<=1+1e-5
    return out

class Canonicalizer(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Sequential(nn.Conv2d(1,8,3,2,1),nn.ReLU(),nn.Conv2d(8,16,3,2,1),nn.ReLU(),nn.Conv2d(16,16,3,2,1),nn.ReLU());self.map=nn.Linear(16,8);self.bias=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.map.weight);nn.init.zeros_(self.map.bias);nn.init.zeros_(self.bias.weight);nn.init.zeros_(self.bias.bias)
    def forward(self,x,calls=None):
        if calls:
            calls.extra('appearance_samples',len(x));calls.extra('spline_samples',2*len(x));calls.extra('bias_samples',len(x))
        h=self.stem(x);v=self.map(h.mean((2,3)));a=pool4(self.bias(h));b=field(a,x.shape[-2:]);out=spline(bias_inverse(x,b),v,inverse=True)
        assert torch.isfinite(out).all() and float(out.detach().abs().max())<=1+1e-5
        return out

def check(device):
    torch.manual_seed(37);dtype=torch.double;x=torch.linspace(-1,1,257,device=device,dtype=dtype).reshape(1,1,1,257).expand(2,1,17,257).contiguous();v=torch.randn(2,8,device=device,dtype=dtype);a=torch.randn(2,1,4,4,device=device,dtype=dtype);b=field(a,x.shape[-2:]);m=spline(x,v);y=m+b*(1-m.square());back=spline(bias_inverse(y,b),v,inverse=True);err=float((back-x).abs().max());assert err<=1e-10
    k=knots(v);slopes=(k[:,1:]-k[:,:-1])/.25;assert float(slopes.min())>=.5-1e-12 and float(slopes.max())<=4.5+1e-12 and float((1-2*m*b).min())>=.8-1e-12
    model=Canonicalizer().to(device=device,dtype=dtype);xx=x.clone().requires_grad_();out=model(xx);assert torch.equal(out,xx)
    target=(xx*.8).detach();loss=(model(y.detach())-target).square().mean();loss.backward();assert model.map.weight.grad is not None and float(model.map.weight.grad.norm())>0 and torch.isfinite(model.map.weight.grad).all()
    return dict(status='PASS',device=device,inverse_max_error=err,initial_identity_exact=True,gradient_nonzero=True,synthetic_reverse=1)
