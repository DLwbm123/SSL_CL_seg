"""Finite implementation checks only; no scientific diagnostic run."""
import torch
from .transport import transport,low_band
from ..semantic_evidence_v22.method import sample_semantics
from ..paired_semantic_v19.method import balanced_risk


def check_paths(s,L):
    x=L['x'];g=torch.Generator().manual_seed(24001)
    y=torch.randn(x.shape,generator=g,dtype=x.dtype).to(x.device)
    assert torch.allclose(transport(x,x),x,rtol=1e-10,atol=1e-10)
    z=transport(x,y);fz=torch.fft.fftshift(torch.fft.fft2(z),dim=(-2,-1));fx=torch.fft.fftshift(torch.fft.fft2(x),dim=(-2,-1));fy=torch.fft.fftshift(torch.fft.fft2(y),dim=(-2,-1));mask=low_band(x)
    assert torch.allclose(fz[...,~mask],fx[...,~mask],rtol=1e-8,atol=1e-8)
    assert torch.allclose(fz.abs()[...,mask],fy.abs()[...,mask],rtol=1e-8,atol=1e-8)
    assert not z.requires_grad and torch.equal(z,transport(x,y))
    # One backwards pass confirms real-label risk reaches the deployed predictor.
    theta=s.q.mean;a=s.current.rsample(theta,g)
    prob,_=sample_semantics(s,z,theta,a,g)
    loss=balanced_risk(prob,L['observations']['y_obs'].to(prob.dtype)).mean()
    loss.backward()
    assert s.q.mean.grad is not None and s.q.mean.grad.norm()>0
    for head in (s.phi.enc,s.phi.c_head,s.phi.r_head):
        assert head.weight.grad is not None and head.weight.grad.norm()>0 and torch.isfinite(head.weight.grad).all()
    s.optimizer.zero_grad(set_to_none=True)
    return dict(identity=True,high_frequency_unchanged=True,low_amplitude_matches_donor=True,deterministic=True,detached_transform=True,real_label_risk_reaches_q_phi=True,reverse_calls=1,optimizer_updates=0)
