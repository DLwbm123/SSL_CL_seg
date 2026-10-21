"""Read-only synthetic diagnostics of completed neural-head qualification; no optimizer."""
import json
from pathlib import Path
import torch
import numpy as np
from validate_neural_kappa import initial,data,features,head,kl,COST

def density_checks():
    rows=[]
    for k in [-2.,0.,.01,1.,2.,209.]:
        noise=np.array([.3,-.7]);sc,sr=.4,.3
        delta=np.array([sc*noise[0],-k*sc*noise[0]+sr*noise[1]])
        cov=np.array([[sc*sc,-k*sc*sc],[-k*sc*sc,k*k*sc*sc+sr*sr]])
        dense=-np.log(2*np.pi)-.5*np.linalg.slogdet(cov)[1]-.5*delta@np.linalg.solve(cov,delta)
        triangular=-np.log(2*np.pi)-np.log(sc*sr)-.5*noise@noise
        error=abs(dense-triangular);assert error<1e-7
        rows.append(dict(kappa=k,density_error=float(error)))
    return rows


def main():
    _,im=data(torch.float64)
    w=initial('linear',torch.float64)+torch.arange(85,dtype=torch.float64).reshape(5,17)*1e-5
    wf=w.float().requires_grad_(True);img=im.float().requires_grad_(True)
    _,_,k=head(wf,img,'linear')
    # Independent affine head derivative: d k_n/d W_4j = feature_nj.
    a=torch.linspace(-.7,1.1,12,dtype=torch.float32)
    COST['autograd_requests']+=1;g=torch.autograd.grad((k*a).sum(),wf)[0]
    analytic=(features(img.detach()).double()*a.double()[:,None]).sum(0)
    err=float(((g[4].double()-analytic).abs()/(1+analytic.abs())).max())
    # Exactly rounded FP32 inputs/weights promoted to FP64, no initial-distribution change.
    vals=[]
    for dtype in [torch.float32,torch.float64]:
        ww=wf.detach().to(dtype).requires_grad_(True);ii=img.detach().to(dtype).requires_grad_(True)
        v=kl(ww,ii,'linear').mean();COST['autograd_requests']+=1;gg,gi=torch.autograd.grad(v,(ww,ii))
        vals.append((gg.double(),gi.double()))
    error=(vals[0][0]-vals[1][0]).abs()/(1+vals[1][0].abs())
    source=json.loads((Path(__file__).parent/'results'/'BAYESEG_G_NEURAL_KAPPA_SYNTHETIC.json').read_text())
    decomposition=[]
    for r in source['runs']:
        rows=[]
        for row in r['cases']:
            t,x=row['tau'],row['observation'];d=1+1/t**2;b=1/t**2;k=row['kappa'];vc,vr=np.array(row['sd'])**2
            delta=np.array(row['means'])-x/(2+t*t)
            mean_term=.5*(np.sum(delta**2)+np.sum(delta)**2/t**2)
            c=1+k*k+(1-k)**2/t**2
            # Profiled variance KL at this k gives a nonnegative coupling-only gap.
            coupling=.5*np.log(c*d/(1+2/t**2))
            scale=.5*((c*vc-1-np.log(c*vc))+(d*vr-1-np.log(d*vr)))
            residual=abs(mean_term+coupling+scale-row['KL']);assert residual<1e-7
            rows.append(dict(tau=t,observation=x,mean_KL=float(mean_term),coupling_profile_KL=float(coupling),scale_KL=float(scale),sum_error=float(residual)))
        decomposition.append(dict(kind=r['kind'],solver=r['solver'],cases=rows))
    result=dict(extra_density_evaluations=6,density_checks=density_checks(),affine_kappa_FP32_chain_error=err,affine_chain_pass=err<2e-6,
                full_KL_same_rounded_inputs_block_errors=error.max(1).values.tolist(),
                full_KL_same_rounded_inputs_image_error=float(((vals[0][1]-vals[1][1]).abs()/(1+vals[1][1].abs())).max()),
                decomposition=decomposition,cost=COST,network_updates=0,
                limitation='Post-hoc location only, does not replace original full-KL precision failure or endpoint gate')
    assert err<2e-6
    path=Path(__file__).parent/'results'/'BAYESEG_G_NEURAL_KAPPA_DIAGNOSTICS.json'
    assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='decomposition'}))
if __name__=='__main__':main()
