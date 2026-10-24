"""Eight coupled constrained Newton directions with a Fenchel dual certificate."""
import torch
from torch.nn import functional as F
from ..apos_v2.math import area,lift,pixel,project
from ..apos_v3.math import mixture,penalty,precision,latent,energy

def cells(x):return x.permute(0,2,3,1).reshape(x.shape[0],-1,3)
def image(x,g):return x.reshape(x.shape[0],g,g,3).permute(0,3,1,2)
def face_project(c,H,calls=None):
    # All15 faces, including zero, for a full3x3 metric. Fixed first KKT tie.
    masks=[];masses=[]
    for mass in (False,True):
        for bits in range(1 if mass else 0,8):masks.append([(bits>>j)&1 for j in range(3)]);masses.append(mass)
    mask=torch.tensor(masks,device=c.device,dtype=c.dtype);mass=torch.tensor(masses,device=c.device,dtype=torch.bool)
    if calls:calls.cell(c.shape[0]*c.shape[1]*15)
    HH=H[...,None,:,:]*mask[None,None,:,:,None]*mask[None,None,:,None,:]+torch.diag_embed(1-mask)[None,None]
    rhs=(H@c[...,None])[...,0][...,None,:]*mask
    rr=torch.stack((rhs,mask.expand_as(rhs)),-1);sol=torch.linalg.solve(HH,rr);x=sol[...,0];j=sol[...,1]
    denom=torch.where(mass,j.sum(-1),torch.ones_like(j.sum(-1)));eta=torch.where(mass,(x.sum(-1)-1)/denom,torch.zeros_like(denom));x=x-eta[...,None]*j
    grad=(H[...,None,:,:]@(x-c[...,None,:])[...,None])[...,0]+eta[...,None]
    scale=1+(H@c[...,None])[...,0].abs().amax(-1)[...,None]+eta.abs()
    valid=((x*mask).amin(-1)>=-1e-12)&torch.where(mask.bool(),grad.abs()<=1e-10*scale[...,None],grad>=-1e-10*scale[...,None]).all(-1)&torch.where(mass,eta>=-1e-10*scale,x.sum(-1)<=1+1e-12)
    assert bool(valid.any(-1).all()),'full-metric face KKT'
    ix=valid.long().argmax(-1);out=x.gather(-2,ix[...,None,None].expand(*ix.shape,1,3))[...,0,:]
    # This is a full-metric projection; only signed zero is cleaned after the KKT solve.
    assert float(out.min())>=-1e-12 and float(out.sum(-1).max())<=1+1e-12
    out=out.clamp_min(0)
    return out,mask[ix],mass[ix]

def face_matrices(H,mask,mass,D,Tinv):
    eye=torch.eye(3,device=H.device,dtype=H.dtype);Hinv=torch.linalg.solve(H,eye.expand_as(H))
    base=mask.long().argmax(-1);ee=F.one_hot(base,3).to(H)
    Z=torch.diag_embed(mask)-ee[..., :,None]*mask[...,None,:]
    Z=torch.where(mass[...,None,None],Z,torch.diag_embed(mask));used=Z.square().sum(-2)>0
    B=Z@torch.linalg.solve(Z.transpose(-1,-2)@H@Z+torch.diag_embed((~used).to(H)),Z.transpose(-1,-2))
    A=torch.diag_embed(1-mask);row=ee[..., :,None].bool()
    A=torch.where((mass[...,None,None]&row),torch.ones_like(A),A);active=A.square().sum(-1)>0
    G=A@Hinv@A.transpose(-1,-2)+torch.diag_embed((~active).to(H))
    AHinvD=(A@Hinv)*D[...,None,:]
    K0=torch.linalg.solve(torch.diag_embed(1/D)+Tinv,eye.expand_as(H))
    K=K0+AHinvD.transpose(-1,-2)@torch.linalg.solve(G,AHinvD)
    return B,(K+K.transpose(-1,-2))*.5

def dual_certificate(lp,lq,u,a0,z,f,d,w,kind):
    M=u.shape[-1]**2;pp=lp.exp();r=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z)
    if kind=='Q':s=r/d;conjugate=.5*d*s.square();inward=0
    else:
        alpha=r/torch.sqrt(d+r.square());eps=torch.finfo(r.dtype).eps;inward=int((alpha.abs()>=1).sum());alpha=alpha.clamp(-1+eps,1-eps);s=alpha/d.sqrt();t=d*s.square();conjugate=t/(1+torch.sqrt((1-t).clamp_min(0)))
    v=w*(area(pp[:,1:],u.shape[-1]).double()-u)
    corr=torch.cat((torch.zeros_like(v[:,:1]),v),1)
    pixel_term=-torch.logsumexp(lq-lift(corr,lq.shape[-2:]),1).mean((1,2))
    support=torch.cat((torch.zeros_like(v[:,:1]),v-s.reshape_as(v)),1).amax(1).sum((1,2))
    dv=pixel_term-(.5*(v.square()/w).flatten(1).sum(1)+(a0.flatten(1)*s).sum(1)+conjugate.sum(1)+.5*torch.einsum('bnr,bn->br',f,s).square().sum(1)+support)/M
    ep=[]
    for b in range(len(lp)):ep.append(energy(lp[b:b+1],lq[b:b+1],u[b:b+1],a0[b:b+1],z[b:b+1],f[b:b+1],d[b:b+1],w[b:b+1],kind))
    primal=torch.stack(ep);gap=primal-dv;relative=gap/torch.maximum(torch.ones_like(gap),torch.maximum(primal.abs(),dv.abs()))
    return dict(primal=float(primal.max()),dual=float(dv.min()),duality_gap_max=float(gap.max()),duality_gap_min=float(gap.min()),relative_duality_gap_max=float(relative.max()),dual_inward_roundoff_coordinates=inward)

def solve(lq,lq0,w,pi,prior,kind,calls=None,failure=None):
    assert kind in ('Q','R');output_dtype=lq.dtype;lq=lq.double();lq=lq-torch.logsumexp(lq,1,keepdim=True);lq0=lq0.double();lq0=lq0-torch.logsumexp(lq0,1,keepdim=True)
    f,d=mixture(prior,pi);g=int((d.shape[1]//3)**.5);a0=area(lq0.exp()[:,1:],g);u=area(lq.exp()[:,1:],g);w=w.double();z=torch.zeros(len(lq),f.shape[-1],device=lq.device,dtype=torch.double)
    Fc=f.reshape(len(lq),3,g,g,-1).permute(0,2,3,1,4).reshape(len(lq),g*g,3,-1);dc=cells(d.reshape_as(u));wc=cells(w);trace=[];maxpixel=0.
    try:
        lp,res=pixel(lq,u,w,calls);maxpixel=max(maxpixel,float(res.detach()));E=energy(lp,lq,u,a0,z,f,d,w,kind);initial=float(E.detach())
        for direction in range(8):
            if calls:calls.component(len(lq))
            pf=lp.exp()[:,1:];aa=area(pf,g);v=w*(aa-u);r=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z);rc=cells(r.reshape_as(u));ss=rc/dc if kind=='Q' else rc/(dc*torch.sqrt(1+rc.square()/dc));D=1/dc if kind=='Q' else (1/dc)*(1+rc.square()/dc).pow(-1.5)
            cov=torch.diag_embed(cells(aa));prod=(pf[:,:,None]*pf[:,None,:]).flatten(1,2);meanprod=area(prod,g).reshape(len(lp),3,3,g,g).permute(0,3,4,1,2).reshape_as(cov);cov=cov-meanprod
            Tinv=torch.diag_embed(1/wc)+cov;eye=torch.eye(3,device=lq.device,dtype=torch.double);T=torch.linalg.solve(Tinv,eye.expand_as(cov));H=T+torch.diag_embed(D)
            gu=-cells(v)+ss;gz=z-torch.einsum('bmcr,bmc->br',Fc,ss)
            c=cells(u)-torch.linalg.solve(H,gu[...,None])[...,0];prox,mask,mass=face_project(c,H,calls);part=prox-cells(u);B,K=face_matrices(H,mask,mass,D,Tinv)
            S=torch.eye(f.shape[-1],device=lq.device,dtype=torch.double)[None]+torch.einsum('bmcr,bmcd,bmds->brs',Fc,K,Fc)
            rhs=-gz+torch.einsum('bmcr,bmc->br',Fc,D*part);dz=torch.cholesky_solve(rhs[...,None],torch.linalg.cholesky(S))[...,0]
            du=part+torch.einsum('bmcd,bmdr,br->bmc',B,D[...,None]*Fc,dz);slope=(gu*du).sum((1,2))+(gz*dz).sum(1);slope=slope.mean()/(g*g)
            accepted=False
            for alpha in (1.,.5,.25,.125):
                if calls:calls.merit(len(lq))
                cand,_,_=face_project(cells(u)+alpha*du,H,calls);uc=image(cand,g);zc=z+alpha*dz;lpc,rp=pixel(lq,uc,w,calls);Ec=energy(lpc,lq,uc,a0,zc,f,d,w,kind)
                actual_slope=((gu*(cand-cells(u))).sum((1,2))+(gz*(alpha*dz)).sum(1)).mean()/(g*g)
                if float(actual_slope.detach())<=1e-12 and float(Ec.detach())<=float((E+1e-4*actual_slope).detach())+1e-12*max(1,abs(float(E.detach()))):accepted=True;break
            assert accepted,('fixed Armijo candidates exhausted',kind,direction)
            linear_res=(S@dz[...,None])[...,0]-rhs;lr=float((linear_res.abs().amax(1)/(1+rhs.abs().amax(1))).max())
            assert lr<=1e-8,('joint linear residual',lr)
            trace.append(dict(direction=direction+1,energy=float(Ec.detach()),step=alpha,linear_residual=lr,active_zero_coordinates=int((mask==0).sum()),active_mass_cells=int(mass.sum())))
            u,z,lp,E=uc,zc,lpc,Ec;maxpixel=max(maxpixel,float(rp.detach()))
        lp,res=pixel(lq,u,w,calls);maxpixel=max(maxpixel,float(res.detach()))
        with torch.no_grad():
            rr=(u-a0).flatten(1)-torch.einsum('bnr,br->bn',f,z);Di=precision(rr,d,kind);zn=latent(u.flatten(1),a0.flatten(1),f,Di);zr=float(((zn-z).abs()/(1+z.abs().amax(1,keepdim=True))).max());gu=w*(u-area(lp.exp()[:,1:],g))+(Di*rr).reshape_as(u);hh=w+Di.reshape_as(u);ur=float((project(u-gu/hh,hh)-u).abs().max());stationary=max(float(res),zr,ur)
            if calls:calls.dual(len(lq))
            cert=dual_certificate(lp,lq,u,a0,z,f,d,w,kind);final=float(energy(lp,lq,u,a0,z,f,d,w,kind));stats=dict(initial_energy=initial,final_energy=final,max_pixel_residual=maxpixel,stationarity_residual=stationary,latent_residual=zr,occupancy_residual=ur,**cert,joint_trace=trace,W_mean=float(w.mean()),W_max=float(w.max()),probability_change_max=float((lp.exp()-lq.exp()).abs().max()),probability_change_mean=float((lp.exp()-lq.exp()).abs().mean()),argmax_flip_fraction=float((lp.argmax(1)!=lq.argmax(1)).float().mean()),class_flip_counts=[int(((lp.argmax(1)!=lq.argmax(1))&(lq.argmax(1)==k)).sum()) for k in range(4)],standardized_residual_mean=float((rr.abs()/d.sqrt()).mean()),robust_to_quadratic_influence_ratio_mean=float((1/torch.sqrt(1+rr.square()/d)).mean()))
            assert final<=initial+1e-6*max(1,abs(initial)),('energy increase',kind,initial,final)
            assert stationary<=1e-4,('joint stationarity residual',kind,stationary,zr,ur)
            assert cert['duality_gap_min']>=-1e-10 and cert['relative_duality_gap_max']<=1e-6,('duality gap',kind,cert)
        return lp.to(output_dtype),stats
    except BaseException as error:
        if failure:failure(dict(error=str(error),kind=kind,trace=trace,final_stats=locals().get('stats')),dict(logq=lq.detach(),logq0=lq0.detach(),W=w.detach(),pi=pi.detach(),F=f.detach(),d=d.detach(),u=u.detach(),z=z.detach()))
        raise

def check(device,prior,calls=None):
    torch.manual_seed(41);prior={k:v.to(device) for k,v in prior.items()};dtype=torch.double;grid=32
    coords=torch.linspace(-1,1,grid,device=device,dtype=dtype);xx,yy=torch.meshgrid(coords,coords,indexing='ij');base=torch.stack((xx*.2,yy*.2,-xx*.2,-yy*.2))[None];q0=F.log_softmax(base,1)
    pi=prior['mode_probs'].double()[None];w=torch.full((1,3,grid,grid),.5,device=device,dtype=dtype);f,d=mixture(prior,pi);aligned=f[0,:,0].reshape(1,3,grid,grid);aligned=.1*aligned/aligned.abs().max().clamp_min(1e-12)
    low=torch.cat((torch.zeros_like(aligned[:,:1]),aligned),1);resid=torch.sin(torch.arange(3*grid*grid,device=device,dtype=dtype)).reshape(1,3,grid,grid)*1e-3;noise=torch.cat((torch.zeros_like(resid[:,:1]),resid),1)
    boundary=q0.clone();boundary[:,1]=boundary[:,1]+8;boundary=F.log_softmax(boundary,1)
    stress=dict(prior,sigma2=torch.full_like(prior['sigma2'],1e-6))
    families=[('identity',q0,q0,prior),('aligned',F.log_softmax(q0+low,1),q0,prior),('joint_floor',F.log_softmax(q0+noise,1),q0,stress),('boundary',F.log_softmax(boundary+low,1),boundary,prior)]
    notes=[];derivatives=[]
    for name,lq,ref,pp in families:
        for kind in ('Q','R'):
            q=lq.detach().clone().requires_grad_();lp,stats=solve(q,ref,w,pi,pp,kind,calls,calls.capture if calls else None);value=lp.square().mean();value.backward();assert torch.isfinite(q.grad).all()
            if name=='identity':assert float((lp.exp()-ref.exp()).abs().max())<=5e-7 and torch.equal(lp.argmax(1),ref.argmax(1))
            if name=='aligned':
                direction=torch.cos(torch.arange(q.numel(),device=device,dtype=dtype)).reshape_as(q);direction=direction/direction.norm();eps=1e-5
                plus,_=solve(q.detach()+eps*direction,ref,w,pi,pp,kind,calls,calls.capture if calls else None);minus,_=solve(q.detach()-eps*direction,ref,w,pi,pp,kind,calls,calls.capture if calls else None);fd=float((plus.square().mean()-minus.square().mean())/(2*eps));auto=float((q.grad*direction).sum());error=abs(fd-auto);assert error<=2e-5 and error/max(1e-5,abs(fd),abs(auto))<=.02,('directional derivative',kind,fd,auto,error);derivatives.append(dict(kind=kind,finite_difference=fd,autograd=auto,error=error))
            notes.append(dict(family=name,kind=kind,stationarity=stats['stationarity_residual'],gap=stats['relative_duality_gap_max']))
    # A genuinely non-diagonal projection reference, compared with independent face KKT.
    cc=torch.tensor([[[-.2,.7,.9],[.4,.4,.4],[.1,.2,.3]]],device=device,dtype=dtype);HH=torch.tensor([[2.,.3,.1],[.3,1.,.2],[.1,.2,1.5]],device=device,dtype=dtype)[None,None].expand(1,3,3,3);uu,_,_=face_project(cc,HH,calls);assert float(uu.min())>=0 and float(uu.sum(-1).max())<=1+1e-12
    # Dense independent joint quadratic solve versus the Schur algebra, interior case.
    ff=torch.tensor([[.1,.2],[.3,-.1],[.05,.15]],device=device,dtype=dtype);WW=torch.diag(torch.tensor([.5,.7,.4],device=device,dtype=dtype));DD=torch.diag(torch.tensor([1e6,300.,5000.],device=device,dtype=dtype));a=torch.tensor([.02,-.01,.01],device=device,dtype=dtype);I=torch.eye(2,device=device,dtype=dtype)
    if calls:
        from .runtime import synthetic_record
        synthetic_record(calls.cfg,'reference_solve',1)
    dense=torch.cat((torch.cat((WW+DD,-DD@ff),1),torch.cat((-ff.T@DD,I+ff.T@DD@ff),1)),0);rhs=torch.cat((WW@a,torch.zeros(2,device=device,dtype=dtype)));full=torch.linalg.solve(dense,rhs);K=torch.linalg.solve(torch.linalg.inv(DD)+torch.linalg.inv(WW),torch.eye(3,device=device,dtype=dtype));zz=torch.linalg.solve(I+ff.T@K@ff,ff.T@K@a);ut=torch.linalg.solve(WW+DD,WW@a+DD@ff@zz);dense_error=float((full-torch.cat((ut,zz))).abs().max());assert dense_error<=1e-9
    return dict(status='PASS',device=device,synthetic_reverse=8,checks=notes,gradient_checks=derivatives,dense_joint_reference_error=dense_error,real_nonzero_prior=True)
