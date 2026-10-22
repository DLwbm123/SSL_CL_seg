"""Independent numerical checks for the reference kernel; no scientific claims."""
import inspect
import json
import math
from pathlib import Path
import sys
import torch
from .posterior_reference import Reference, elliptical, normal, solve, diagnostics, probability_checks
from .config import Config
from .stream_state import StreamState
from .fixed_inference import true_theta
from .budgets import ResourceLedger


def checks():
    torch.set_num_threads(1);torch.manual_seed(22031);g=torch.Generator().manual_seed(22031)
    c=Config(height=4,width=6,dct_order=2);s=StreamState(c);theta=true_theta(s)
    a=torch.tensor([.15,.4,-2.],dtype=torch.float64);ref=Reference(c,s.model,theta,a)
    n,k=ref.n,ref.k;omega=torch.linspace(.3,2.,n,dtype=torch.float64)
    nv=(2*a[2]).exp()/omega;chol=torch.linalg.cholesky(ref.C0+torch.diag(nv))
    mean=normal((n+k,),g)*.1;x=normal((n,),g)
    precision=torch.linalg.inv(ref.V)+ref.H.T@torch.diag(1/nv)@ref.H
    covariance=torch.linalg.inv(precision)
    truth=covariance@(torch.linalg.solve(ref.V,mean)+ref.H.T@(x/nv))
    computed=mean+ref.VHt@solve(chol,x-ref.H@mean)
    conditional_cov=ref.V-ref.VHt@torch.cholesky_solve(ref.VHt.T,chol)
    err=max(float((computed-truth).abs().max()),float((conditional_cov-covariance).abs().max()))
    assert err<1e-10
    # Independent dense MVN identity tests normalization, not only posterior means.
    h=normal((n+k,),g)
    prior=torch.distributions.MultivariateNormal(mean,covariance_matrix=ref.V)
    obs=torch.distributions.Normal(ref.H@h,nv.sqrt()).log_prob(x).sum()
    evidence=torch.distributions.MultivariateNormal(ref.H@mean,covariance_matrix=ref.C0+torch.diag(nv)).log_prob(x)
    post=torch.distributions.MultivariateNormal(truth,covariance_matrix=covariance).log_prob(h)
    densityerr=float((prior.log_prob(h)+obs-evidence-post).abs());assert densityerr<1e-8
    samples=torch.stack([ref.conditional_draw(mean,chol,nv,x,g) for _ in range(12000)])
    standardized=float(((samples.mean(0)-truth)/(covariance.diag()/len(samples)).sqrt()).abs().max())
    cov_error=float((torch.cov(samples.T)-covariance).abs().max());assert standardized<6 and cov_error<.005
    # Integrate Gamma precision analytically, including every normalizing constant.
    nu=4.;scale=.3;r=torch.linspace(-3,3,100,dtype=torch.float64)
    alpha=torch.tensor(nu/2,dtype=torch.float64);rate=alpha
    logmix=alpha*rate.log()-torch.lgamma(alpha)-math.log(scale)-.5*math.log(2*math.pi)+torch.lgamma(alpha+.5)-(alpha+.5)*torch.log(rate+r*r/(2*scale*scale))
    stud=torch.distributions.StudentT(nu,0.,scale).log_prob(r)
    # Distribution constants must use float64 too.
    stud=torch.distributions.StudentT(r.new_tensor(nu),r.new_tensor(0.),r.new_tensor(scale)).log_prob(r)
    mixerr=float((logmix-stud).abs().max());assert mixerr<1e-12
    # Elliptical kernel has an independent conjugate target N(.8,.2).
    u=torch.zeros(1,dtype=torch.float64);retained=[]
    for i in range(14000):
        u,_=elliptical(u,lambda v:-.5*((v-1)/.5).square().sum(),g)
        if i>=2000:retained.append(u)
    draws=torch.stack(retained).flatten()
    merr=abs(float(draws.mean())-.8);verr=abs(float(draws.var())-.2)
    assert merr<.035 and verr<.025
    iid=normal((4,1000,8),g);good=diagnostics(iid)
    shifted=iid.clone();shifted[0]+=2;bad=diagnostics(shifted)
    stuck=diagnostics(torch.ones_like(iid));assert good['pass_gate'] and not bad['pass_gate'] and not stuck['pass_gate']
    prob=normal((4,1000,8),g).sigmoid();assert probability_checks(prob)['pass_gate']
    assert 'y' not in inspect.signature(Reference.chain).parameters
    return dict(status='PASS',conditional_error=err,density_identity_error=densityerr,
        conditional_mean_max_standard_errors=standardized,conditional_covariance_max_error=cov_error,
        student_mixture_error=mixerr,elliptical_mean_error=merr,elliptical_variance_error=verr,
        iid_diagnostics=good,shifted_diagnostics=bad,constant_chains_rejected=True,label_argument_absent=True,
        seed=22031)


if __name__=='__main__':
    ledger=ResourceLedger(Path(sys.argv[1]),max_updates=24)
    with ledger.attempt('reference_checks01',seconds=300,updates=0) as out:
        result=checks();(out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
