"""Fixed nonlinear and minibatch synthetic validation; no medical dependencies."""
import json, signal, time
from pathlib import Path
import torch
import torch.nn.functional as F
import validate_neural_kappa as base
from validate_neural_kappa_repair import B0

ROOT=Path(__file__).parent
SIZES=[64,32,1024,32,160,5]

def initial():
    gen=torch.Generator().manual_seed(20261021)
    chunks=[]
    for n,fan in [(64,2),(32,0),(1024,32),(32,0),(160,0),(5,0)]:
        chunks.append((torch.rand(n,generator=gen)*2-1)/fan**.5 if fan else torch.zeros(n))
    return torch.cat(chunks)

def outputs(w,im,dtype):
    a,b,c,d,e,f=w.to(dtype).split(SIZES)
    z=im.to(dtype).mean((-1,-2));z=torch.stack([(z[:,0]+.5)/1.5,z[:,1]/2],1)
    z=torch.tanh(F.linear(z,a.reshape(32,2),b))
    z=torch.tanh(F.linear(z,c.reshape(32,32),d))
    z=F.linear(z,e.reshape(5,32),f)
    return z[:,:2],1e-6+F.softplus(B0+4*z[:,2:4]),.01+z[:,4]

def terms(w,im,dtype):
    mu,sd,k=outputs(w,im,dtype);im=im.to(dtype)
    t=10**im[:,0].mean((-1,-2));x=im[:,1].mean((-1,-2));d=1+1/t**2
    delta=mu-x[:,None]/(2+t*t)[:,None];c=1+k*k+(1-k)**2/t**2
    mean=.5*(delta.square().sum(1)+delta.sum(1).square()/t**2)
    coupling=.5*torch.log(c*d/(1+2/t**2))
    v=torch.stack([c*sd[:,0]**2,d*sd[:,1]**2],1)
    scale=.5*(v-1-v.log()).sum(1)
    # Original positive Gaussian KL, with decomposition evaluated separately.
    kl=mean+.5*(v.sum(1)-2-torch.log1p(2/t**2)-2*sd.log().sum(1))
    return kl,mean,coupling,scale

def evaluate(w,im):
    with torch.no_grad():
        vals=terms(w,im,torch.float64)
        assert float((vals[0]-sum(vals[1:])).abs().max())<1e-7
        mu,sd,k=outputs(w,im,torch.float64)
    rows=[]
    for i in range(len(im)):
        rows.append(dict(tau=float(10**im[i,0,0,0]),observation=float(im[i,1,0,0]),KL=float(vals[0][i]),mean_KL=float(vals[1][i]),coupling_KL=float(vals[2][i]),scale_KL=float(vals[3][i]),kappa=float(k[i]),means=mu[i].tolist(),sd=sd[i].tolist()))
    return dict(passed=int((vals[0].abs()<=1e-6).sum()),max_KL=float(vals[0].max()),cases=rows)

def checks(w,im):
    w=(w+torch.sin(torch.arange(w.numel()))*1e-4).requires_grad_(True);im=im.clone().requires_grad_(True)
    grads=[]
    for dtype in [torch.float32,torch.float64]:
        grads.append(torch.autograd.grad(terms(w,im,dtype)[0].mean(),(w,im)))
    wd=w.detach().double().requires_grad_(True);ii=im.detach().double().requires_grad_(True)
    ref=torch.autograd.grad(terms(wd,ii,torch.float64)[0].mean(),(wd,ii))
    err=[max(float(((a.double()-b).abs()/(1+b.abs())).max()) for a,b in zip(g,ref)) for g in grads]
    direction=torch.cos(torch.arange(w.numel(),dtype=torch.float64));di=torch.sin(torch.arange(ii.numel(),dtype=torch.float64)).reshape_as(ii);h=1e-5
    with torch.no_grad():num=(terms(wd+h*direction,ii+h*di,torch.float64)[0].mean()-terms(wd-h*direction,ii-h*di,torch.float64)[0].mean())/(2*h)
    analytic=(ref[0]*direction).sum()+(ref[1]*di).sum();fd=float(abs(num-analytic)/(1+abs(num)))
    full=torch.autograd.grad(terms(wd,ii,torch.float64)[0].mean(),wd)[0]
    batch=[torch.autograd.grad(terms(wd,ii[j:j+4],torch.float64)[0].mean(),wd)[0] for j in [0,4,8]]
    averaging=float(((sum(batch)/3-full).abs()/(1+full.abs())).max())
    assert averaging<2e-6
    return dict(FP32_error=err[0],mixed_error=err[1],FP32_pass=err[0]<2e-6,mixed_pass=err[1]<2e-6,directional_error=fd,directional_pass=fd<2e-6,partition_gradient_error=averaging,batch_relative_deviation=[float((g-full).norm()/(1+full.norm())) for g in batch])

def polish(start,im,dtype,batches=None):
    from bounded_lbfgs import make_optimizer,accepted_step
    w=start.clone().requires_grad_(True);calls=0;iterations=0;failure=None
    for indices in ([torch.arange(12)] if batches is None else batches):
        opt=make_optimizer(w);counter={'calls':0}
        try:
            for _ in range(200 if batches is None else 10):
                move=accepted_step(opt,w,lambda q:terms(q,im[indices],dtype)[0].mean(),counter,400 if batches is None else 20)
                if move==0:break
                iterations+=1
        except RuntimeError as e:failure=str(e)
        calls+=counter['calls']
        if failure:break
    return w.detach(),dict(calls=calls,iterations=iterations,failure=failure,closure_batch_fixed=True,budget_restores_last_accepted=True)

def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600 second hard limit')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_NONLINEAR_STOCHASTIC.json';assert not path.exists()
    _,im=base.data(torch.float32);gen=torch.Generator().manual_seed(20261022)
    batches=[torch.randperm(12,generator=gen)[:4] for _ in range(5020)]
    unseen=torch.tensor([[torch.log10(torch.tensor(t,dtype=torch.float64)),x] for t in [.03,.3,3.,30.] for x in [-1.,0.,1.]],dtype=torch.float64)[:,:,None,None].expand(-1,-1,8,8).clone()
    init=initial();mu,sd,k=outputs(init,im,torch.float64)
    assert float(mu.abs().max())==0 and torch.allclose(sd,torch.full_like(sd,torch.exp(torch.tensor(-4.)).item()),atol=1e-8)
    out=dict(protocol='NONLINEAR_STOCHASTIC_PREREG_ZH.md',parameter_count=init.numel(),formal_training=False,real_data_reads=0,checks=checks(init,im),runs=[])
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for dtype in [torch.float32,torch.float64]:
            for stochastic in [False,True]:
                beg=time.monotonic();w=init.clone().requires_grad_(True)
                opt=torch.optim.Adam([w],lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
                for step in range(5000):
                    opt.zero_grad();v=terms(w,im[batches[step]] if stochastic else im,dtype)[0].mean();v.backward()
                    assert torch.isfinite(v) and torch.isfinite(w.grad).all()
                    torch.nn.utils.clip_grad_norm_([w],1.);opt.step()
                r=dict(compute=str(dtype),stochastic=stochastic,adam_steps=5000,adam=evaluate(w,im),adam_unseen=evaluate(w,unseen))
                wp,meta=polish(w.detach(),im,dtype)
                r.update(polish=meta,endpoint=evaluate(wp,im),unseen=evaluate(wp,unseen),weights=wp.tolist())
                if dtype==torch.float64 and stochastic:
                    wb,mb=polish(w.detach(),im,dtype,batches[5000:])
                    r.update(batch_polish=mb,batch_endpoint=evaluate(wb,im),batch_unseen=evaluate(wb,unseen),batch_weights=wb.tolist())
                r['seconds']=time.monotonic()-beg;out['runs'].append(r);save()
                print(dict(compute=r['compute'],stochastic=stochastic,Adam=r['adam']['passed'],LBFGS=r['endpoint']['passed'],max_KL=r['endpoint']['max_KL'],unseen=r['unseen']['passed'],failure=meta['failure']),flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:
        out['status']='FAILED';out['failure']=repr(e);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
