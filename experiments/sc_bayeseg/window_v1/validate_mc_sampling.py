"""Equal-draw IID, antithetic and randomized Sobol pathwise gradients."""
import json, signal, time
from pathlib import Path
import torch
from validate_covariance_coordinates import Fixture, v
ROOT=Path(__file__).parent

def fixture():
    source=json.loads((ROOT/'results/BAYESEG_G_BLOCK_MECHANISM.json').read_text())
    pre=json.loads((ROOT/'results/BAYESEG_G_COVARIANCE_PRECONDITION.json').read_text())
    _,im=v.base.data(torch.float32)
    f=Fixture(torch.tensor(source['mean_diagnostic']['weights'],dtype=torch.float64),im)
    f.h=f.h@torch.tensor(pre['transform'],dtype=torch.float64)
    return f,pre

def samples(rep):
    gen=torch.Generator().manual_seed(20261024+rep)
    iid=torch.randn(128,12,2,generator=gen,dtype=torch.float64)
    anti=torch.stack([iid[:64],-iid[:64]],1).reshape(128,12,2)
    u=torch.quasirandom.SobolEngine(24,scramble=True,seed=20261024+rep).draw(128,dtype=torch.float64)
    sobol=(2**.5)*torch.erfinv(2*u.clamp(2**-53,1-2**-53)-1)
    assert torch.equal(anti[::2],-anti[1::2]) and torch.isfinite(sobol).all()
    return {'IID':iid,'antithetic':anti,'Sobol':sobol.reshape(128,12,2)}

def energy(f,p,kind,eps):
    _,_,_,sd,_,ell,_=f.values(p,kind)
    delta=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None]
    c=delta[:,0]+sd[:,0]*eps[:,:,0]
    r=delta[:,1]+ell*eps[:,:,0]+sd[:,1]*eps[:,:,1]
    return (.5*(c.square()+r.square()+(c+r).square()/f.t.square())).mean()-sd.log().sum(1).mean()

def summarize(gs,exact):
    gs=torch.stack(gs);avg=gs.mean(0);se=gs.std(0,unbiased=True)/len(gs)**.5;err=avg-exact
    return dict(normalized_RMSE=float((gs-exact).square().sum(1).mean().sqrt()/(1+exact.norm())),variance_trace=float(gs.var(0,unbiased=True).sum()),normalized_mean_error=float(err.norm()/(1+exact.norm())),outside_5SE=int((err.abs()>5*se.clamp_min(1e-12)).sum()),max_Z=float((err.abs()/se.clamp_min(1e-12)).max()))

def main():
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_MC_SAMPLING.json';assert not path.exists()
    out=dict(protocol='MC_SAMPLING_PREREG_ZH.md',formal_training=False,optimizer_updates=0,rows=[])
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    beg=time.monotonic()
    try:
        f,pre=fixture();start=time.monotonic();noise=[samples(r) for r in range(128)];out['sampling_seconds']=time.monotonic()-start
        for run in pre['runs']:
            for point in ['initial','endpoint']:
                p=(torch.zeros(3,33) if point=='initial' else torch.tensor(run['weights'],dtype=torch.float32)).requires_grad_()
                exact=torch.autograd.grad(f.values(p,run['kind'])[0].mean(),p)[0].double().flatten()
                for draws in [8,32,128]:
                    baseline=None
                    for method in ['IID','antithetic','Sobol']:
                        gs=[torch.autograd.grad(energy(f,p,run['kind'],n[method][:draws]),p)[0].double().flatten() for n in noise]
                        row=dict(kind=run['kind'],point=point,draws=draws,method=method,exact_norm=float(exact.norm()),**summarize(gs,exact))
                        if baseline is None:baseline=row['variance_trace']
                        row['variance_ratio_to_IID']=row['variance_trace']/baseline
                        out['rows'].append(row);save();print(row,flush=True)
        out.update(status='COMPLETED',MC_gradient_requests=4608,exact_gradient_requests=4,latent_pairs=3096576,seconds=time.monotonic()-beg);save()
    except BaseException as e:out.update(status='FAILED',failure=repr(e));save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
