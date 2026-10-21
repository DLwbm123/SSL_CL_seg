"""Independent-pilot gradient CV, polynomial sanity and sine generalization."""
import json,math,signal,time
import torch
import torch.nn.functional as F
from validate_mc_sampling import ROOT,fixture,summarize
from validate_mc_nonlinear_cv import objective as poly_objective

def exact(f,p,kind,decoder):
    if decoder=='poly':return poly_objective(f,p,kind)
    _,_,_,sd,_,ell,_=f.values(p,kind);d=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None]
    dc,dr=d.T;vc=sd[:,0].square();cov=sd[:,0]*ell
    cross=torch.exp(-vc/2)*((dc+dr)*dc.sin()+(vc+cov)*dc.cos())
    sin2=(1-torch.exp(-2*vc)*(2*dc).cos())/2
    return f.values(p,kind)[0].mean()+((.1*cross+.005*sin2)/f.t.square()).mean()

def local(f,p,kind,decoder,eps):
    z=(f.h@p.detach().double().T).expand(len(eps),-1,-1).clone().requires_grad_()
    sd=1e-6+F.softplus(f.raw0+4*z[:,:,:2]);ell=-(f.k0+z[:,:,2])*sd[:,:,0] if kind=='K' else f.ell0+z[:,:,2]
    d=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None]
    c=d[:,0]+sd[:,:,0]*eps[:,:,0];r=d[:,1]+ell*eps[:,:,0]+sd[:,:,1]*eps[:,:,1]
    g=c+.1*(c.square() if decoder=='poly' else c.sin())
    values=.5*(c.square()+r.square()+(g+r).square()/f.t.square())-sd.log().sum(2)
    return torch.autograd.grad(values.sum(),z)[0].detach(),values.mean()

def basis(e):
    def h(x):return [torch.ones_like(x),x,x*x-1,x**3-3*x,x**4-6*x*x+3]
    a,b=h(e[:,:,0]),h(e[:,:,1])
    return torch.stack([a[i]*b[j]/math.sqrt(math.factorial(i)*math.factorial(j)) for degree in range(5) for i in range(degree+1) for j in [degree-i]],-1)

def mapped(f,g):return (g.mean(0).T@f.h/12).flatten()

def main():
    torch.set_num_threads(1);signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600s')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_MC_HERMITE.json';assert not path.exists()
    out=dict(protocol='MC_HERMITE_PREREG_ZH.md',formal_training=False,updates=0,checks=[],rows=[]);beg=time.monotonic()
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        f,pre=fixture();noise=[]
        for rep in range(128):
            gen=torch.Generator().manual_seed(20261025+rep);iid=torch.randn(128,12,2,generator=gen,dtype=torch.float64)
            u=torch.quasirandom.SobolEngine(24,scramble=True,seed=20261025+rep).draw(128,dtype=torch.float64)
            sobol=(2**.5*torch.erfinv(2*u.clamp(2**-53,1-2**-53)-1)).reshape(128,12,2);noise.append((iid,sobol))
        for decoder in ['poly','sin']:
            for run in pre['runs']:
                kind=run['kind']
                for point in ['initial','endpoint']:
                    p=(torch.zeros(3,33) if point=='initial' else torch.tensor(run['weights'],dtype=torch.float32)).double().requires_grad_()
                    ref=torch.autograd.grad(exact(f,p,kind,decoder),p)[0].flatten().detach()
                    d=torch.cos(torch.arange(99,dtype=torch.float64)).reshape(3,33);h=1e-5
                    num=(exact(f,p+h*d,kind,decoder)-exact(f,p-h*d,kind,decoder))/(2*h)
                    fd=float(abs(num-(ref*d.flatten()).sum())/(1+abs(num)))
                    loc,_=local(f,p,kind,decoder,noise[0][0]);z=f.h@p.T;sd=1e-6+F.softplus(f.raw0+4*z[:,:2]);ell=-(f.k0+z[:,2])*sd[:,0] if kind=='K' else f.ell0+z[:,2]
                    delta=f.mu-f.im[:,1,0,0,None]/(2+f.t.square())[:,None];eps=noise[0][0];c=delta[:,0]+sd[:,0]*eps[:,:,0];r=delta[:,1]+ell*eps[:,:,0]+sd[:,1]*eps[:,:,1];g=c+.1*(c.square() if decoder=='poly' else c.sin())
                    loss=(.5*(c.square()+r.square()+(g+r).square()/f.t.square())).mean()-sd.log().sum(1).mean()
                    chain=torch.autograd.grad(loss,p)[0].flatten();ce=float(((mapped(f,loc)-chain).abs()/(1+chain.abs())).max())
                    check=dict(decoder=decoder,kind=kind,point=point,FD_error=fd,chain_error=ce,passed=fd<2e-6 and ce<2e-6);out['checks'].append(check);save();assert check['passed']
                    gs={k:[] for k in ['IID','Sobol','Hermite64+64']};maxcond=0
                    for iid,sobol in noise:
                        gi,_=local(f,p,kind,decoder,iid);gq,_=local(f,p,kind,decoder,sobol);b=basis(iid)
                        # Pilot fit is independent of all evaluation draws; intercept is not subtracted.
                        design=b[:64].permute(1,0,2);coef=torch.linalg.lstsq(design,gi[:64].permute(1,0,2),driver='gelsd').solution
                        correction=torch.einsum('ndk,dkc->ndc',b[64:,:,1:],coef[:,1:])
                        gs['IID'].append(mapped(f,gi));gs['Sobol'].append(mapped(f,gq));gs['Hermite64+64'].append(mapped(f,gi[64:]-correction))
                        maxcond=max(maxcond,float(torch.linalg.cond(design).max()))
                    baseline=None
                    for method,values in gs.items():
                        row=dict(decoder=decoder,kind=kind,point=point,method=method,**summarize(values,ref))
                        if baseline is None:baseline=row['variance_trace']
                        errors=((torch.stack(values)-ref).abs()/(1+ref.abs())).amax(1)
                        row.update(variance_ratio_to_IID=row['variance_trace']/baseline,max_coordinate_error=float(errors.max()),strict_pass_repeats=int((errors<2e-6).sum()),pilot_max_condition=maxcond if method=='Hermite64+64' else None)
                        out['rows'].append(row);save();print(row,flush=True)
        out.update(status='COMPLETED',seconds=time.monotonic()-beg,local_gradient_requests=2056,reference_gradient_requests=8,chain_gradient_requests=8,FD_objective_requests=16,least_squares_systems=12288,arm_latent_pairs=4718592,actual_latent_pairs=3158016);save()
    except BaseException as e:out.update(status='FAILED',failure=repr(e));save();raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
