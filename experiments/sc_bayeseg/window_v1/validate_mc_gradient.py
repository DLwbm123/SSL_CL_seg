"""Zero-update MC pathwise covariance gradient study with analytic entropy."""
import json,signal,time
from pathlib import Path
import torch
from validate_covariance_coordinates import Fixture,v
ROOT=Path(__file__).parent

def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_MC_GRADIENT.json';assert not path.exists()
    source=json.loads((ROOT/'results/BAYESEG_G_BLOCK_MECHANISM.json').read_text());pre=json.loads((ROOT/'results/BAYESEG_G_COVARIANCE_PRECONDITION.json').read_text())
    _,im=v.base.data(torch.float32);f=Fixture(torch.tensor(source['mean_diagnostic']['weights'],dtype=torch.float64),im);f.h=f.h@torch.tensor(pre['transform'],dtype=torch.float64)
    gen=torch.Generator().manual_seed(20261023);noise=torch.randn(128,12,128,2,generator=gen,dtype=torch.float64)
    delta=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None]
    out=dict(protocol='MC_GRADIENT_PREREG_ZH.md',optimizer_updates=0,formal_training=False,rows=[]);beg=time.monotonic()
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        for run in pre['runs']:
            for point in ['initial','endpoint']:
                p=(torch.zeros(3,33) if point=='initial' else torch.tensor(run['weights'],dtype=torch.float32)).requires_grad_(True)
                exact=torch.autograd.grad(f.values(p,run['kind'])[0].mean(),p)[0].double().flatten();norm=float(exact.norm())
                for draws in [8,32,128]:
                    gs=[]
                    for rep in range(128):
                        _,_,_,sd,_,ell,_=f.values(p,run['kind']);eps=noise[rep,:,:draws]
                        c=delta[:,0,None]+sd[:,0,None]*eps[:,:,0]
                        r=delta[:,1,None]+ell[:,None]*eps[:,:,0]+sd[:,1,None]*eps[:,:,1]
                        energy=.5*(c.square()+r.square()+(c+r).square()/f.t[:,None].square())
                        loss=(energy.mean(1)-sd.log().sum(1)).mean()
                        gs.append(torch.autograd.grad(loss,p)[0].double().flatten())
                    gs=torch.stack(gs);avg=gs.mean(0);se=gs.std(0,unbiased=True)/(128**.5);error=avg-exact
                    row=dict(kind=run['kind'],point=point,draws=draws,repeats=128,exact_gradient_norm=norm,normalized_mean_error=float(error.norm()/(1+norm)),normalized_single_drawset_RMSE=float(((gs-exact).square().sum(1).mean()).sqrt()/(1+norm)),coordinates_outside_5SE=int((error.abs()>5*se.clamp_min(1e-12)).sum()),max_standardized_error=float((error.abs()/se.clamp_min(1e-12)).max()),mean_cosine=float(F_cos(gs,exact).mean()) if norm>1e-8 else None)
                    out['rows'].append(row);save();print(row,flush=True)
        out.update(status='COMPLETED_ZERO_UPDATE',MC_gradient_requests=1536,exact_gradient_requests=4,sampled_latent_pairs=1032192,seconds=time.monotonic()-beg);save()
    except BaseException as e:out.update(status='FAILED',failure=repr(e));save();raise
    finally:signal.alarm(0)

def F_cos(gs,g):return (gs@g)/(gs.norm(dim=1)*g.norm()).clamp_min(1e-30)
if __name__=='__main__':main()
