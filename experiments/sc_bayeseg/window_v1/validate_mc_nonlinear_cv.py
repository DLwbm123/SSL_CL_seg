"""Fixed nonlinear decoder with an exact quadratic control variate."""
import json,signal,time
import torch
from validate_mc_sampling import ROOT,fixture,samples,summarize,energy

def residual(f,p,kind,eps=None):
    _,_,_,sd,_,ell,_=f.values(p,kind)
    delta=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None]
    dc,dr=delta.T;vc=sd[:,0].square();cov=sd[:,0]*ell;a=.1
    if eps is None:
        cubic=(dc+dr)*(dc.square()+vc)+2*dc*(vc+cov)
        fourth=dc**4+6*dc.square()*vc+3*vc.square()
        return ((a*cubic+.5*a*a*fourth)/f.t.square()).mean()
    c=dc+sd[:,0]*eps[:,:,0];r=dr+ell*eps[:,:,0]+sd[:,1]*eps[:,:,1]
    return ((a*(c+r)*c.square()+.5*a*a*c**4)/f.t.square()).mean()

def objective(f,p,kind):return f.values(p,kind)[0].mean()+residual(f,p,kind)

def main():
    torch.set_num_threads(1);signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_MC_NONLINEAR_CV.json';assert not path.exists()
    out=dict(protocol='MC_NONLINEAR_CV_PREREG_ZH.md',formal_training=False,optimizer_updates=0,checks=[],rows=[])
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    beg=time.monotonic()
    try:
        f,pre=fixture();noise=[samples(r) for r in range(128)]
        for run in pre['runs']:
            kind=run['kind']
            for point in ['initial','endpoint']:
                p=(torch.zeros(3,33) if point=='initial' else torch.tensor(run['weights'],dtype=torch.float32)).requires_grad_()
                exact=torch.autograd.grad(objective(f,p,kind),p)[0].double().flatten()
                q=p.detach().double().requires_grad_();ref=torch.autograd.grad(objective(f,q,kind),q)[0]
                d=torch.cos(torch.arange(99,dtype=torch.float64)).reshape(3,33);h=1e-5
                with torch.no_grad():num=(objective(f,q+h*d,kind)-objective(f,q-h*d,kind))/(2*h)
                mixed=float(((exact.reshape(3,33)-ref).abs()/(1+ref.abs())).max());fd=float(abs(num-(ref*d).sum())/(1+abs(num)))
                check=dict(kind=kind,point=point,mixed_error=mixed,FD_error=fd,passed=mixed<2e-6 and fd<2e-6);out['checks'].append(check);save();assert check['passed']
                for method in ['IID','antithetic','Sobol']:
                    baseline=None
                    for estimator in ['full','CV']:
                        gs=[]
                        for n in noise:
                            eps=n[method]
                            loss=(energy(f,p,kind,eps) if estimator=='full' else f.values(p,kind)[0].mean())+residual(f,p,kind,eps)
                            gs.append(torch.autograd.grad(loss,p)[0].double().flatten())
                        row=dict(kind=kind,point=point,method=method,estimator=estimator,draws=128,exact_norm=float(exact.norm()),**summarize(gs,exact))
                        if baseline is None:baseline=row['variance_trace']
                        row['variance_ratio_to_full_same_sampler']=row['variance_trace']/baseline
                        out['rows'].append(row);save();print(row,flush=True)
        out.update(status='COMPLETED',MC_gradient_requests=3072,reference_gradient_requests=8,FD_objective_requests=8,latent_pairs=4718592,seconds=time.monotonic()-beg);save()
    except BaseException as e:out.update(status='FAILED',failure=repr(e));save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
