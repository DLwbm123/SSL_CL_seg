"""Independent reduction, kernel and NumPy diagnostic checks."""
import math
from statistics import NormalDist
import numpy as np
import torch
from .kernel_comparison import root_matrix,reduce_u,restore,Reduced,block_step
from .posterior_reference import normal,diagnostics,gaussian_log,rank_normal,split,rhat,effective
from .config import Config
from .stream_state import StreamState
from .fixed_inference import true_theta

def independent_diagnostics(raw):
    x=np.concatenate([raw[:,:raw.shape[1]//2],raw[:,-raw.shape[1]//2:]],0)
    def ranks(v):
        flat=v.reshape(-1,v.shape[-1]);z=np.empty_like(flat)
        for j in range(flat.shape[1]):
            order=np.argsort(flat[:,j],kind='stable');values=flat[order,j]
            start=0
            while start<len(values):
                end=start+1
                while end<len(values) and values[end]==values[start]:end+=1
                rank=(start+1+end)/2
                z[order[start:end],j]=NormalDist().inv_cdf((rank-.375)/(len(values)+.25));start=end
        return z.reshape(v.shape)
    def rh(v):
        n=v.shape[1];W=np.var(v,axis=1,ddof=1).mean(0);between=np.var(v.mean(1),axis=0,ddof=1)
        return np.sqrt(((n-1)/n*W+between)/W)
    def ess(v):
        m,n,d=v.shape;W=np.var(v,axis=1,ddof=1).mean(0);V=(n-1)/n*W+np.var(v.mean(1),axis=0,ddof=1)
        centered=v-v.mean(1,keepdims=True);ac=np.empty((n,d))
        # Direct lag products, independent of the FFT implementation.
        for lag in range(n):ac[lag]=(centered[:,:n-lag]*centered[:,lag:]).sum(1).mean(0)/n
        rho=1-(W-ac)/V;rho[0]=1;answer=[]
        for j in range(d):
            total=0.;prev=float('inf')
            for lag in range(0,n-1,2):
                pair=rho[lag,j]+rho[lag+1,j]
                if pair<=0:break
                prev=min(prev,pair);total+=prev
            answer.append(m*n/max(1.,-1+2*total))
        return np.array(answer)
    z=ranks(x);median=np.sort(x.reshape(-1,x.shape[-1]),axis=0)[(x.shape[0]*x.shape[1]-1)//2]
    rr=np.maximum(rh(z),rh(ranks(abs(x-median))))
    flat=x.reshape(-1,x.shape[-1]);low,hi=np.quantile(flat,[.05,.95],axis=0)
    return dict(rhat_max=float(rr.max()),bulk_ess_min=float(ess(z).min()),tail_ess_min=float(np.minimum(ess((x<=low).astype(float)),ess((x>=hi).astype(float))).min()))

@torch.no_grad()
def checks():
    g=torch.Generator().manual_seed(20264100);s=StreamState(Config());theta=true_theta(s)
    a=torch.tensor([.15,.4,-2.],dtype=torch.float64);ref=Reduced(Config(),s.model,theta,a);L=root_matrix()
    K=torch.eye(16,dtype=torch.float64)+.01*torch.ones(16,16,dtype=torch.float64)
    kerr=float((L@L.T-K).abs().max());assert kerr<1e-12
    chol=torch.linalg.cholesky(ref.C0+torch.eye(256,dtype=torch.float64)*.03);x=normal((256,),g)
    errors=[]
    for _ in range(64):
        u=normal((34,),g);v=reduce_u(u);m1,p1=ref.mean_prob(u);m2,p2=ref.mean_v(v)
        errors.extend([float((p1-p2).abs().max()),float((m1-m2).abs().max()),float(abs(gaussian_log(x,ref.H@m1,chol)-gaussian_log(x,ref.H@m2,chol)))])
    assert max(errors)<1e-10
    # Projected directions give the same entire ellipse, hence same slice decisions.
    u=normal((34,),g);direction=normal((34,),g);trajectory=0.
    for angle in torch.linspace(-3,3,33):
        left=reduce_u(u*angle.cos()+direction*angle.sin())
        right=reduce_u(u)*angle.cos()+reduce_u(direction)*angle.sin()
        trajectory=max(trajectory,float((left-right).abs().max()))
    assert trajectory<1e-12
    v=normal((16,),g);samples=torch.stack([restore(v,g) for _ in range(16000)])
    mean=.1*v.sum()/1.16;residual=samples[:,0]-mean
    meanerr=float(abs(residual.mean()));varerr=float(abs(residual.var()-1/1.16))
    assert meanerr<.025 and varerr<.035 and float((reduce_u(samples)-v).abs().max())<1e-12
    z=ref.unpack(samples[0]);z['S']=normal((1,1,16,16),g);z['b']=normal((1,1,15),g)*.1
    assert torch.isfinite(s.model.log_joint(x.reshape(1,1,16,16),z,theta,a).total).all()
    # A full correlated Gaussian target from an independent linear observation.
    H=normal((8,16),g)/4;y=normal((8,),g);noise=.8
    covariance=torch.linalg.inv(torch.eye(16,dtype=torch.float64)+H.T@H/noise**2)
    truth=covariance@H.T@y/noise**2;kernel_results={}
    for arm in ('A','B'):
        w=torch.zeros(16,dtype=torch.float64);rows=[]
        for i in range(5000):
            w,_=block_step(w,lambda z:-.5*((H@z-y)/noise).square().sum(),g,arm)
            if i>=1000:rows.append(w)
        draws=torch.stack(rows);me=float((draws.mean(0)-truth).abs().max());ce=float((torch.cov(draws.T)-covariance).abs().max())
        assert me<.12 and ce<.15,(arm,me,ce)
        kernel_results[arm]=dict(mean_error=me,covariance_error=ce)
    raw=normal((4,400,4),g);raw[0,:,0]+=.2
    td=diagnostics(raw);nd=independent_diagnostics(raw.numpy())
    cross={k:abs(td[k]-nd[k]) for k in nd};assert max(cross.values())<1e-8
    priorerr=0.
    for _ in range(16):
        v=normal((16,),g);_,p=ref.mean_v(v);_,q=ref.mean_v(-v);priorerr=max(priorerr,float((p+q-1).abs().max()))
    assert priorerr<1e-12
    return dict(status='PASS',seed=20264100,covariance_error=kerr,equivalence_max_error=max(errors),ellipse_projection_error=trajectory,
                recovery_mean_error=meanerr,recovery_variance_error=varerr,original_joint_finite=True,gaussian_kernel=kernel_results,
                independent_numpy=nd,torch_diagnostic={k:td[k] for k in nd},diagnostic_errors=cross,analytic_prior_symmetry_error=priorerr)
