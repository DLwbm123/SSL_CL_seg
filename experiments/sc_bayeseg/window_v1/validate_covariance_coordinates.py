"""Anchored shared covariance heads: kappa versus direct Cholesky off-diagonal."""
import json,signal,time
from pathlib import Path
import torch
import torch.nn.functional as F
import validate_nonlinear_stochastic as v
from validate_block_mechanism import features
from bounded_lbfgs import make_optimizer,accepted_step
ROOT=Path(__file__).parent

class Fixture:
    def __init__(self,w,im):
        self.im=im.double();self.h=features(w,self.im).detach()
        self.mu,self.sd0,self.k0=[a.detach() for a in v.outputs(w,self.im,torch.float64)]
        self.raw0=v.B0+4*(self.h@torch.cat([w[1152:1312].reshape(5,32),w[1312:,None]],1).T)[:,2:4]
        self.ell0=-self.k0*self.sd0[:,0]
        self.t=10**self.im[:,0,0,0];x=self.im[:,1,0,0]
        delta=self.mu-x[:,None]/(2+self.t.square())[:,None]
        self.mean=.5*(delta.square().sum(1)+delta.sum(1).square()/self.t.square())
    def values(self,p,kind):
        z=self.h@p.double().T
        sd=1e-6+F.softplus(self.raw0+4*z[:,:2])
        ell=-(self.k0+z[:,2])*sd[:,0] if kind=='K' else self.ell0+z[:,2]
        k=-ell/sd[:,0];t=self.t;d=1+1/t.square()
        # Original precision quadratic in stable positive Cholesky form.
        trace=sd[:,0].square()+ell.square()+(sd[:,0]+ell).square()/t.square()+d*sd[:,1].square()
        kl=self.mean+.5*(trace-2-torch.log1p(2/t.square())-2*sd.log().sum(1))
        c=1+k*k+(1-k).square()/t.square()
        coupling=.5*torch.log(c*d/(1+2/t.square()))
        scaled=torch.stack([c*sd[:,0].square(),d*sd[:,1].square()],1)
        scale=.5*(scaled-1-scaled.log()).sum(1)
        cov=torch.stack([sd[:,0].square(),sd[:,0]*ell,ell.square()+sd[:,1].square()],1)
        return kl,coupling,scale,sd,k,ell,cov
    def evaluate(self,p,kind):
        with torch.no_grad():kl,c,s,sd,k,ell,cov=self.values(p,kind)
        assert float((kl-self.mean-c-s).abs().max())<1e-7
        rows=[dict(tau=float(self.t[i]),observation=float(self.im[i,1,0,0]),KL=float(kl[i]),mean_KL=float(self.mean[i]),coupling_KL=float(c[i]),scale_KL=float(s[i]),sd=sd[i].tolist(),kappa=float(k[i]),ell21=float(ell[i]),covariance_entries=cov[i].tolist()) for i in range(12)]
        return dict(passed=int((kl.abs()<=1e-6).sum()),max_KL=float(kl.max()),mean_KL=float(kl.mean()),cases=rows)
    def gradient_check(self,p,kind,finite_difference=False):
        p=p.detach().float().requires_grad_(True)
        g=torch.autograd.grad(self.values(p,kind)[0].mean(),p)[0]
        q=p.detach().double().requires_grad_(True);ref=torch.autograd.grad(self.values(q,kind)[0].mean(),q)[0]
        error=float(((g.double()-ref).abs()/(1+ref.abs())).max());out=dict(mixed_error=error,mixed_pass=error<2e-6)
        if finite_difference:
            d=torch.cos(torch.arange(99,dtype=torch.float64)).reshape(3,33);h=1e-5
            with torch.no_grad():num=(self.values(q+h*d,kind)[0].mean()-self.values(q-h*d,kind)[0].mean())/(2*h)
            err=float(abs(num-(ref*d).sum())/(1+abs(num)));out.update(directional_error=err,directional_pass=err<2e-6)
        return out

def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600 second budget')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_COVARIANCE_COORDINATES.json';assert not path.exists()
    source=json.loads((ROOT/'results/BAYESEG_G_BLOCK_MECHANISM.json').read_text())
    w=torch.tensor(source['mean_diagnostic']['weights'],dtype=torch.float64);_,im=v.base.data(torch.float32);f=Fixture(w,im)
    zero=torch.zeros(3,33,dtype=torch.float32);left=f.values(zero,'K');right=f.values(zero,'L')
    covariance_error=float((left[-1]-right[-1]).abs().max());objective_error=float((left[0]-v.terms(w,im,torch.float64)[0]).abs().max())
    assert covariance_error<1e-12 and objective_error<1e-12
    out=dict(protocol='COVARIANCE_COORDINATE_PREREG_ZH.md',formal_training=False,real_data_reads=0,initial_covariance_error=covariance_error,initial_objective_error=objective_error,initial=f.evaluate(zero,'K'),runs=[])
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for kind in ['K','L']:
            check=f.gradient_check(torch.sin(torch.arange(99)).reshape(3,33)*1e-4,kind,True)
            p=zero.clone().requires_grad_(True);opt=make_optimizer(p);counter={'calls':0};trace=[];stop='iteration_limit';beg=time.monotonic()
            for i in range(200):
                try:move=accepted_step(opt,p,lambda q:f.values(q,kind)[0].mean(),counter,400)
                except RuntimeError as e:stop=str(e);break
                if move==0:stop='no parameter displacement';break
                ev=f.evaluate(p,kind);trace.append(dict(iteration=i+1,calls=counter['calls'],step_norm=move,max_KL=ev['max_KL'],mean_KL=ev['mean_KL'],passed=ev['passed']))
            r=dict(kind=kind,calls=counter['calls'],sample_gradient_evaluations=12*counter['calls'],iterations=len(trace),seconds=time.monotonic()-beg,stop=stop,initial_gradient_check=check,endpoint_gradient_check=f.gradient_check(p,kind),trace=trace,endpoint=f.evaluate(p,kind),weights=p.detach().tolist(),hidden_updates=0,mean_updates=0)
            r['at_call_budget']={str(b):next((q for q in reversed(trace) if q['calls']<=b),None) for b in [100,200,300,400]}
            out['runs'].append(r);save();print({k:r[k] for k in ['kind','calls','iterations','stop','initial_gradient_check']},r['endpoint']['passed'],r['endpoint']['max_KL'],flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:
        out['status']='FAILED';out['failure']=repr(e);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
