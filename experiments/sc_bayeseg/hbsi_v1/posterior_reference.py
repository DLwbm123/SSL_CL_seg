"""Bounded, image-only collapsed Gibbs reference for the frozen grayscale toy.

No recognition network, label likelihood, optimizer, or real-data reader is used.
"""
import argparse
import inspect
import json
import math
from pathlib import Path
import time
import torch
from .config import Config
from .stream_state import StreamState
from .fixed_inference import true_theta
from .synthetic import scores
from .spatial_gaussian import sar, sar_solve
from .budgets import ResourceLedger


def normal(shape, g):
    return torch.randn(shape, dtype=torch.float64, generator=g)


def solve(chol, b):
    return torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)


def gaussian_log(x, mean, chol):
    r = x - mean
    return -.5 * (r @ solve(chol, r) + 2 * chol.diag().log().sum() + len(x)*math.log(2*math.pi))


def elliptical(current, loglike, g):
    direction = normal(current.shape, g)
    threshold = loglike(current) + torch.rand((), generator=g, dtype=torch.float64).log()
    angle = float(torch.rand((), generator=g, dtype=torch.float64)) * 2*math.pi
    lo, hi = angle-2*math.pi, angle
    for count in range(1000):
        proposal = current*math.cos(angle) + direction*math.sin(angle)
        if loglike(proposal) >= threshold:
            return proposal, count+1
        if angle < 0: lo = angle
        else: hi = angle
        angle = lo + float(torch.rand((), generator=g, dtype=torch.float64))*(hi-lo)
    raise RuntimeError('elliptical bracket exhausted; no silent rejected step')


class Reference:
    def __init__(self, config, model, theta, a):
        if config.channels != 1 or config.classes != 2:
            raise ValueError('this reference is only qualified for the frozen grayscale binary toy')
        self.c, self.model, self.theta = config, model, theta
        self.a = a
        self.n = config.height*config.width
        self.k = config.b_dim
        eye = torch.eye(self.n, dtype=torch.float64)
        A = sar(eye.reshape(self.n,1,config.height,config.width),config.alpha).reshape(self.n,self.n).T
        self.vs = config.sigma_s**2 * torch.linalg.inv(A.T@A)
        self.ls = torch.linalg.cholesky(self.vs)
        self.B = model.basis.reshape(self.k,self.n).T
        self.H = torch.cat([a[0].exp()*eye,self.B],1)
        self.V = torch.block_diag(self.vs, config.tau_b**2*torch.eye(self.k,dtype=torch.float64))
        self.VHt = self.V@self.H.T
        self.C0 = self.H@self.VHt
        self.calls = 0

    def unpack(self,u):
        return {'c':u[:self.c.c_dim][None],
                'R':u[self.c.c_dim:].reshape(1,self.c.r_channels,self.c.r_height,self.c.r_width)}

    def mean_prob(self,u):
        self.calls += 1
        p = self.model.class_probabilities(self.unpack(u),self.theta)
        s = self.model.structure_mean(p,self.theta).flatten()
        return torch.cat([s, s.new_zeros(self.k)]), p[0,1].flatten()

    def conditional_draw(self,mean,chol,noise_var,x,g):
        h0 = mean + torch.cat([self.ls@normal((self.n,),g),self.c.tau_b*normal((self.k,),g)])
        error = noise_var.sqrt()*normal((self.n,),g)
        return h0 + self.VHt@solve(chol,x-self.H@h0-error)

    @torch.no_grad()
    def chain(self,x,seed,scale=1.,warmup=500,draws=1000,progress=None):
        g=torch.Generator().manual_seed(seed)
        u=normal((self.c.c_dim+self.c.r_channels*self.c.r_height*self.c.r_width,),g)*scale
        omega=torch.ones(self.n,dtype=torch.float64)
        centered=x.flatten()-self.a[1]
        probs=[]; traces=[]; shrinks=[]
        for step in range(warmup+draws):
            nv=self.a[2].mul(2).exp()/omega
            chol=torch.linalg.cholesky(self.C0+torch.diag(nv))
            def ll(v):
                mean,_=self.mean_prob(v)
                return gaussian_log(centered,self.H@mean,chol)
            u,count=elliptical(u,ll,g)
            mean,p=self.mean_prob(u)
            h=self.conditional_draw(mean,chol,nv,centered,g)
            residual=centered-self.H@h
            rate=(self.c.nu+residual.square()*self.a[2].mul(-2).exp())/2
            omega=torch._standard_gamma(torch.full_like(rate,(self.c.nu+1)/2),generator=g)/rate
            if not torch.isfinite(omega).all() or not (omega>0).all():
                raise FloatingPointError('invalid conditional precision')
            if step>=warmup:
                z=self.unpack(u);z['S']=h[:self.n].reshape(1,1,self.c.height,self.c.width);z['b']=h[self.n:].reshape(1,1,self.k)
                logjoint=self.model.log_joint(x[None],z,self.theta,self.a).total[0]
                probs.append(p);traces.append(torch.cat([u,h,omega,logjoint[None]]));shrinks.append(count)
            if progress and step%100==0:progress()
        return torch.stack(probs),torch.stack(traces),sum(shrinks)/len(shrinks)


def rank_normal(x):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    sorted_,order=flat.sort(0)
    ranks=torch.empty_like(flat)
    # Tied indicator/constant draws require average ranks.
    for j in range(flat.shape[1]):
        _,counts=torch.unique_consecutive(sorted_[:,j],return_counts=True)
        ends=counts.cumsum(0).double(); means=ends-(counts.double()-1)/2
        ranks[order[:,j],j]=torch.repeat_interleave(means,counts)
    q=(ranks-.375)/(len(flat)+.25)
    return (math.sqrt(2)*torch.erfinv(2*q-1)).reshape(shape)


def split(x):
    n=x.shape[1]//2
    return torch.cat([x[:,:n],x[:,-n:]],0)


def rhat(x):
    m,n,_=x.shape
    W=x.var(1).mean(0)
    B=n*x.mean(1).var(0)
    var=(n-1)/n*W+B/n
    # Constant chains are never accepted as evidence of mixing.
    return torch.where(W>1e-24,(var/W).sqrt(),torch.full_like(W,float('inf')))


def effective(x):
    m,n,d=x.shape
    W=x.var(1).mean(0);var=(n-1)/n*W+x.mean(1).var(0)
    centered=x-x.mean(1,keepdim=True)
    f=torch.fft.rfft(centered,n=2*n,dim=1)
    ac=torch.fft.irfft(f.conj()*f,n=2*n,dim=1)[:,:n].real/n
    rho=1-(W[None]-ac.mean(0))/var.clamp_min(1e-30)[None]
    rho[0]=1
    pairs=rho[:2*(n//2)].reshape(n//2,2,d).sum(1)
    alive=(pairs>0).cumprod(0)
    monotone=pairs.cummin(0).values.clamp_min(0)*alive
    tau=(-1+2*monotone.sum(0)).clamp_min(1.)
    return torch.where(W>1e-24,m*n/tau,torch.zeros_like(W))


def diagnostics(x):
    s=split(x);r=rank_normal(s)
    folded=rank_normal((s-s.reshape(-1,s.shape[-1]).median(0).values).abs())
    rh=torch.maximum(rhat(r),rhat(folded))
    low=torch.quantile(s.reshape(-1,s.shape[-1]),.05,dim=0)
    high=torch.quantile(s.reshape(-1,s.shape[-1]),.95,dim=0)
    bulk=effective(r)
    tail=torch.minimum(effective((s<=low).double()),effective((s>=high).double()))
    return dict(rhat_max=float(rh.max()),bulk_ess_min=float(bulk.min()),tail_ess_min=float(tail.min()),
                worst_rhat_coordinate=int(rh.argmax()),worst_bulk_coordinate=int(bulk.argmin()),
                pass_gate=bool((rh<=1.01).all() and (bulk>=100).all() and (tail>=100).all()))


def probability_checks(p):
    batches=p.reshape(p.shape[0],-1,50,p.shape[-1]).mean(2)
    se=(batches.var(1).mean(0)/(p.shape[0]*batches.shape[1])).sqrt()
    spread=p.mean(1).max(0).values-p.mean(1).min(0).values
    return dict(mcse_max=float(se.max()),chain_mean_range_max=float(spread.max()),
                pass_gate=bool(se.max()<=.03 and spread.max()<=.10))


@torch.no_grad()
def sealed_data(state):
    c=state.config;theta=true_theta(state);items=[]
    for d,seed in enumerate((20263101,20263102)):
        g=torch.Generator().manual_seed(seed);n=4
        z={'c':normal((n,c.c_dim),g),'R':normal((n,c.r_channels,c.r_height,c.r_width),g),
           'b':normal((n,c.channels,c.b_dim),g)*c.tau_b}
        p=state.model.class_probabilities(z,theta)
        mean=state.model.structure_mean(p,theta)
        z['S']=mean+c.sigma_s*sar_solve(normal(mean.shape,g),c.alpha)
        e=normal(mean.shape,g);chi=normal((4,*mean.shape),g).square().sum(0)
        a=mean.new_tensor([0.,-.2,-2.] if d==0 else [.15,.4,-2.])
        x=a[0].exp()*z['S']+a[1]+torch.einsum('bck,khw->bchw',z['b'],state.model.basis)+a[2].exp()*e/(chi/4).sqrt()
        y=torch.multinomial(p.permute(0,2,3,1).reshape(-1,2),1,generator=g).reshape(n,c.height,c.width)
        for i in range(n):items.append(dict(domain=d,index=i,x=x[i],y=y[i],true_probability=p[i],a=a))
    return items


def as_prob(p,c):
    p=p.reshape(-1,c.height,c.width)
    return torch.stack([1-p,p],1)


@torch.no_grad()
def run(out,ledger):
    cfg=Config();state=StreamState(cfg);theta=true_theta(state)
    items=sealed_data(state)
    torch.save(items,out/'SEALED_SYNTHETIC.pt')
    prior=[];g=torch.Generator().manual_seed(920001)
    for _ in range(4):
        accum=[]
        for _ in range(16):
            z={'c':normal((256,cfg.c_dim),g),'R':normal((256,cfg.r_channels,cfg.r_height,cfg.r_width),g)}
            accum.append(state.model.class_probabilities(z,theta).mean(0))
        prior.append(torch.stack(accum).mean(0))
    prior=torch.stack(prior);torch.save(prior,out/'PRIOR_PREDICTIONS.pt')
    rows=[];posterior=[];start=time.monotonic()
    for idx,item in enumerate(items):
        ref=Reference(cfg,state.model,theta,item['a']);ps=[];ts=[];counts=[]
        for chain,scale in enumerate((1.,2.,.5,3.)):
            p,t,count=ref.chain(item['x'],900001+idx*10000+chain*101,scale,progress=ledger.check)
            ps.append(p);ts.append(t);counts.append(count)
            torch.save({'probability':p,'trace':t},out/f'image{idx}_chain{chain}.pt')
            print(json.dumps({'image':idx,'chain':chain,'seconds':time.monotonic()-start}),flush=True)
        p=torch.stack(ps);t=torch.stack(ts)
        pd=diagnostics(p);td=diagnostics(t);pc=probability_checks(p)
        row=dict(image=idx,domain=item['domain'],probability_diagnostics=pd,latent_diagnostics=td,
                 probability_mc=pc,reference_pass=pd['pass_gate'] and td['pass_gate'] and pc['pass_gate'],
                 average_ellipse_evaluations=counts,generator_evaluations=ref.calls)
        rows.append(row);posterior.append(as_prob(p.mean((0,1)),cfg)[0])
        (out/'PROGRESS.json').write_text(json.dumps(rows,indent=2))
    # Labels become visible to scoring only after every chain and prediction is sealed.
    torch.save(torch.stack(posterior),out/'SEALED_POSTERIOR.pt')
    for i,(item,p) in enumerate(zip(items,posterior)):
        y=item['y'][None];constant=torch.full_like(p,.5);fg=torch.zeros_like(p);fg[1]=1
        rows[i]['scores']={name:scores(v[None],y) for name,v in
            [('posterior',p),('prior',prior.mean(0)),('true_latent_oracle',item['true_probability']),
             ('constant_half',constant),('all_foreground',fg),('all_background',1-fg)]}
    domains=[]
    for d in (0,1):
        rr=[r for r in rows if r['domain']==d]
        avg={name:{key:sum(r['scores'][name][key] for r in rr)/len(rr) for key in
             ('NLL','Brier','mean_Dice','predicted_foreground_fraction')} for name in rr[0]['scores']}
        po,pr=avg['posterior'],avg['prior']
        gate=po['NLL']<=min(pr['NLL'],math.log(2))-.02 and po['Brier']<=pr['Brier'] and .02<=po['predicted_foreground_fraction']<=.98
        domains.append(dict(domain=d,averages=avg,information_gate=gate))
    better=sum(r['scores']['posterior']['NLL']<r['scores']['prior']['NLL'] for r in rows)
    reliable=all(r['reference_pass'] for r in rows)
    useful=all(d['information_gate'] for d in domains) and better>=7
    decision='RECOMMEND_SEPARATE_FAST_INFERENCE_TRAINING' if reliable and useful else ('REVIEW_GENERATIVE_INFORMATION' if reliable else 'REFERENCE_UNQUALIFIED_NO_FAST_TRAINING')
    result=dict(reference_status='PASS' if reliable else 'UNQUALIFIED',
        information_status=('PASS' if useful else 'FAIL') if reliable else 'UNDETERMINED',decision=decision,
        images=rows,domains=domains,images_better_than_prior=better,prior_batch_probability_range=float((prior.max(0).values-prior.min(0).values).abs().max()),
        global_updates=0,domain_updates=0,phi_updates=0,real_data_access=0,retained_draws=32000,
        total_gibbs_steps=48000,seconds=time.monotonic()-start)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--unit-receipt',type=Path,required=True);parser.add_argument('--revision-receipt',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from .qualification import require_receipt
    for kind,path in [('UNIT',args.unit_receipt),('REVISION',args.revision_receipt)]:
        receipt=json.loads(path.read_text());require_receipt(path,kind,Config(),receipt['evidence_fingerprint'])
    ledger=ResourceLedger(args.output,max_updates=24)
    with ledger.attempt('independent_reference01',seconds=2400,updates=0) as out:
        result=run(out,ledger)
    return 0 if result['reference_status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
