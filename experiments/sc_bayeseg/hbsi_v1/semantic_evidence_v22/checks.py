"""Necessary density/gradient checks, not scientific performance gates."""
import torch
from ..config import Config
from ..stream_state import StreamState
from .method import ImageEvidence


def check_density():
    c=Config(height=4,width=4,dct_order=2,channels=1,inner_steps=0)
    s=StreamState(c);theta=s.q.mean.detach().clone().requires_grad_();a=torch.tensor([.1,-.2,-.4],dtype=torch.float64,requires_grad=True)
    score=ImageEvidence(s.model,a);x=torch.randn(2,1,4,4,dtype=torch.float64);prob=torch.randn(2,2,4,4,dtype=torch.float64).softmax(1)
    basis=torch.eye(16,dtype=torch.float64).reshape(16,4,4)
    cov0=torch.fft.ifft2(torch.fft.fft2(basis)*score.spectrum[0]).real.reshape(16,16).T
    W=score.W.reshape(-1,16).T;cov=cov0+W@W.T
    mean=a[0].exp()*s.model.structure_mean(prob,theta)+a[1]
    expected=torch.distributions.MultivariateNormal(mean.reshape(2,16),covariance_matrix=cov).log_prob(x.reshape(2,16))
    actual=score.score(x,prob,theta);torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)
    actual.sum().backward();assert torch.isfinite(a.grad).all() and torch.isfinite(theta.grad).all()
    def f(v):return ImageEvidence(s.model,v).score(x,prob,theta.detach())
    assert torch.autograd.gradcheck(f,(a.detach().requires_grad_(),),eps=1e-6,atol=1e-4)
    return dict(SAR_FFT_Woodbury_matches_dense_gaussian=True,domain_gradient_finite_difference_PASS=True,generative_gradients_finite=True)
