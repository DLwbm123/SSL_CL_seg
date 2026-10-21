"""Fixed synthetic-only precision x scale-coordinate repair; no real data."""
import json, signal, time
from pathlib import Path
import torch
import validate_neural_kappa as base

ROOT=Path(__file__).parent
B0=float(base.initial('linear',torch.float64)[2,-1])


def mapped(w,gain,precision):
    v=w.to(precision)
    offset=torch.zeros_like(v[2:4]);offset[:,-1]=B0
    return torch.cat([v[:2],gain*v[2:4]+offset,v[4:]],0)


def loss(w,im,gain,precision):
    return base.kl(mapped(w,gain,precision),im.to(precision),'linear')


def initial():
    w=base.initial('linear',torch.float32)
    w[2:4]=0
    return w


def run(gain,precision):
    start=time.monotonic();_,im=base.data(torch.float32)
    # Check a nonzero shared head, with identical rounded coordinates and input.
    w=(initial()+torch.arange(85).reshape(5,17)*1e-5).requires_grad_(True)
    im.requires_grad_(True)
    gs=torch.autograd.grad(loss(w,im,gain,precision).mean(),(w,im))
    wd=w.detach().double().requires_grad_(True);imd=im.detach().double().requires_grad_(True)
    ref=torch.autograd.grad(loss(wd,imd,gain,torch.float64).mean(),(wd,imd))
    err=max(float(((a.double()-b).abs()/(1+b.abs())).max()) for a,b in zip(gs,ref))
    w=initial().requires_grad_(True);im=im.detach()
    opt=torch.optim.Adam([w],lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    trace=[]
    for i in range(5000):
        opt.zero_grad();v=loss(w,im,gain,precision).mean();v.backward()
        assert torch.isfinite(v) and torch.isfinite(w.grad).all()
        torch.nn.utils.clip_grad_norm_([w],1.);opt.step()
        if i in [999,1999,2999,3999,4999]:
            with torch.no_grad():q=loss(w,im,gain,torch.float64)
            trace.append({'step':i+1,'max_KL':float(q.max())})
    result=base.evaluate(mapped(w.detach(),gain,torch.float64),'linear')
    return dict(gain=gain,compute=str(precision),storage='torch.float32',gradient_error=err,gradient_pass=err<2e-6,trace=trace,steps=5000,seconds=time.monotonic()-start,**result)


def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600 second budget')))
    signal.alarm(600)
    p=ROOT/'results/BAYESEG_G_NEURAL_KAPPA_REPAIR.json'
    assert not p.exists(),'create-only'
    out=dict(protocol='NEURAL_KAPPA_REPAIR_PREREG_ZH.md',formal_training=False,real_data_reads=0,server_bytes_added=0,runs=[])
    def save():p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        for gain in [1,4]:
            for precision in [torch.float32,torch.float64]:
                r=run(gain,precision);out['runs'].append(r);save()
                print({k:r[k] for k in ['gain','compute','gradient_error','gradient_pass','passed','max_KL','seconds']},flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:
        out['status']='FAILED';out['failure']=repr(e);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
