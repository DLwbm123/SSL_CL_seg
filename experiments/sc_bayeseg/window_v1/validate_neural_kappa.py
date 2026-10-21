"""Shared amortized posterior head on synthetic images only. No medical data/modules."""
import ast, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

ROOT=Path(__file__).parent
COST={'head_forward':0,'autograd_requests':0,'adam_updates':0,'finite_difference_forward':0}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def features(images):
    z=images.mean((-1,-2))
    t=(z[:,0]+.5)/1.5; x=z[:,1]/2
    def basis(a):return torch.stack([torch.ones_like(a),a,2*a*a-1,4*a**3-3*a],1)
    f=(basis(t)[:,:,None]*basis(x)[:,None,:]).reshape(-1,16)
    return torch.cat([f,torch.ones_like(f[:,:1])],1)


def head(w,images,kind):
    COST['head_forward']+=1
    z=features(images)@w.T
    sigma=1e-6+F.softplus(z[:,2:4])
    k=torch.sigmoid(z[:,4]) if kind=='sigmoid' else z[:,4]
    return z[:,:2],sigma,k


def kl(w,images,kind):
    means,sigma,k=head(w,images,kind)
    # Metadata used only to evaluate the known synthetic likelihood, not in head outputs.
    t=10**images[:,0].mean((-1,-2));x=images[:,1].mean((-1,-2))
    d=1+1/t**2;b=1/t**2;m=x/(2+t*t)
    delta=means-m[:,None];vc=sigma[:,0]**2;vr=sigma[:,1]**2
    # Positive form avoids cancellation d-2kb+k²d at small tau.
    c=1+k*k+(1-k)**2/t**2
    return .5*((delta[:,0]**2+delta[:,1]**2)+(delta.sum(1))**2/t**2+
               c*vc+d*vr-2-torch.log1p(2/t**2)-torch.log(vc)-torch.log(vr))


def initial(kind,dtype):
    w=torch.zeros(5,17,dtype=dtype)
    w[2:4,-1]=np.log(np.expm1(np.exp(-4)-1e-6))
    w[4,-1]=np.log(.01/.99) if kind=='sigmoid' else .01
    return w


def data(dtype):
    pairs=[(t,x) for t in [.01,.1,1.,10.] for x in [-2.,0.,2.]]
    im=torch.tensor([[np.log10(t),x] for t,x in pairs],dtype=dtype)[:,:,None,None].expand(-1,-1,8,8).clone()
    return pairs,im


def evaluate(w,kind):
    pairs,im=data(torch.float64)
    with torch.no_grad():
        vals=kl(w.double(),im,kind);mu,sd,k=head(w.double(),im,kind)
    return {'cases':[dict(tau=t,observation=x,KL=float(vals[i]),kappa=float(k[i]),means=mu[i].tolist(),sd=sd[i].tolist(),pass_original_gate=abs(float(vals[i]))<=1e-6) for i,(t,x) in enumerate(pairs)],
            'max_KL':float(vals.max()),'passed':int((vals.abs()<=1e-6).sum()),'weights':w.detach().double().tolist()}


def fit(kind,solver):
    start=dict(COST);beg=time.monotonic();trace=[]
    dtype=torch.float64 if solver=='L-BFGS-B' else torch.float32
    _,im=data(dtype);w=initial(kind,dtype)
    if solver=='L-BFGS-B':
        def obj(a):
            z=torch.tensor(a.reshape(5,17),dtype=dtype,requires_grad=True)
            loss=kl(z,im,kind).mean();COST['autograd_requests']+=1
            g=torch.autograd.grad(loss,z)[0]
            return float(loss.detach()),g.detach().numpy().ravel()
        def cb(a):
            with torch.no_grad():v=kl(torch.tensor(a.reshape(5,17)),im,kind)
            trace.append({'iteration':len(trace)+1,'mean_KL':float(v.mean()),'max_KL':float(v.max())})
        opt=minimize(obj,w.numpy().ravel(),jac=True,method='L-BFGS-B',callback=cb,options=dict(maxiter=2000,ftol=1e-13,gtol=1e-8,maxls=40))
        w=torch.tensor(opt.x.reshape(5,17));meta=dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),nfev=int(opt.nfev))
    else:
        w.requires_grad_(True)
        opt=torch.optim.Adam([w],lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        for i in range(5000):
            opt.zero_grad();loss=kl(w,im,kind).mean();COST['autograd_requests']+=1;loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(w.grad).all()
            torch.nn.utils.clip_grad_norm_([w],1.);opt.step();COST['adam_updates']+=1
            if i in [0,999,1999,2999,3999,4999]:
                with torch.no_grad():v=kl(w,im,kind)
                trace.append(dict(step=i+1,mean_KL=float(v.mean()),max_KL=float(v.max())))
        meta=dict(iterations=5000,success=None,message='fixed endpoint; no convergence flag')
    result=dict(kind=kind,solver=solver,dtype=str(dtype),trace=trace,optimizer=meta,**evaluate(w,kind))
    result['cost']={k:COST[k]-start[k] for k in COST};result['seconds']=time.monotonic()-beg
    return result


def checks():
    out={};_,im=data(torch.float64)
    # Feature interpolation proves only this finite fixture's expressibility, not generalization.
    f=features(im).numpy();t=10**im[:,0,0,0].numpy();x=im[:,1,0,0].numpy()
    sd=np.stack([np.sqrt((1+t*t)/(2+t*t)),np.sqrt(t*t/(1+t*t))],1)
    k=1/(1+t*t)
    for kind in ['sigmoid','linear']:
        targets=np.column_stack([x/(2+t*t),x/(2+t*t),np.log(np.expm1(sd-1e-6)),np.log(k/(1-k)) if kind=='sigmoid' else k])
        w=torch.tensor(np.linalg.lstsq(f,targets,rcond=None)[0].T)
        out[kind+'_exact_assignment_max_KL']=max(abs(c['KL']) for c in evaluate(w,kind)['cases'])
        assert out[kind+'_exact_assignment_max_KL']<1e-9
    # Joint directional finite differences through the actual image and shared neural head.
    w=initial('linear',torch.float64)+torch.arange(85,dtype=torch.float64).reshape(5,17)*1e-5
    w.requires_grad_(True);im.requires_grad_(True)
    v=kl(w,im,'linear').mean();COST['autograd_requests']+=1;gw,gi=torch.autograd.grad(v,(w,im))
    dw=torch.cos(torch.arange(85,dtype=torch.float64)).reshape(5,17);di=torch.sin(torch.arange(im.numel(),dtype=torch.float64)).reshape_as(im)
    h=1e-5
    for name,a,b in [('head',dw,torch.zeros_like(im)),('image',torch.zeros_like(w),di)]:
        with torch.no_grad():num=(kl(w+h*a,im+h*b,'linear').mean()-kl(w-h*a,im-h*b,'linear').mean())/(2*h)
        COST['finite_difference_forward']+=2
        analytic=(gw*a).sum()+(gi*b).sum();err=float(abs(num-analytic)/(1+abs(num)))
        out[name+'_directional_gradient_error']=err;assert err<2e-6
    wf=w.detach().float().requires_grad_(True);img=im.detach().float().requires_grad_(True)
    v=kl(wf,img,'linear').mean();COST['autograd_requests']+=1;gf,gif=torch.autograd.grad(v,(wf,img))
    out['FP32_gradient_normalized_error']=max(float(((gf.double()-gw).abs()/(1+gw.abs())).max()),float(((gif.double()-gi).abs()/(1+gi.abs())).max()))
    assert torch.isfinite(gf).all() and torch.isfinite(gif).all()
    out['FP32_gradient_comparison_pass']=out['FP32_gradient_normalized_error']<2e-6
    out['boundary_checks']=[]
    for value in [-2.,0.,.01,1.,2.,209.]:
        e=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True);kk=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        c=.2+.4*e[0];r=.1-kk*(c-.2)+.3*e[1]
        COST['autograd_requests']+=2
        jac=torch.stack([torch.autograd.grad(c,e,retain_graph=True)[0],torch.autograd.grad(r,e,retain_graph=True)[0]])
        err=float(abs(torch.det(jac)-.12));assert err<1e-9
        COST['autograd_requests']+=1;g=torch.autograd.grad((c+r)**2,kk)[0];assert torch.isfinite(g)
        out['boundary_checks'].append(dict(kappa=value,jacobian_error=err,kappa_gradient=float(g)))
    # Extract precisely the old Dice function without importing data loaders or checkpoints.
    tree=ast.parse((ROOT/'patient_loss_factor.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='foreground_dice_loss')
    ns={'F':F,'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen_dice>','exec'),ns)
    logits=torch.sin(torch.arange(4*3*4*4*4,dtype=torch.float64)).reshape(4,3,4,4,4).requires_grad_(True)
    y=torch.zeros(3,4,4,dtype=torch.long);y[1,:2]=1;y[2]=torch.arange(16).reshape(4,4)%4
    probs=logits.softmax(2);ce=torch.stack([F.cross_entropy(z,y) for z in logits]).mean();dice=ns['foreground_dice_loss'](probs.mean(0),y)
    ce2=torch.stack([-z.log_softmax(1).gather(1,y[:,None]).mean() for z in logits]).mean()
    pp=probs.mean(0);terms=[]
    for b in range(3):
        cc=[]
        for c in [1,2,3]:
            gt=(y[b]==c).to(pp.dtype)
            if gt.sum()>0:cc.append(1-(2*(pp[b,c]*gt).sum()+1e-6)/(pp[b,c].sum()+gt.sum()+1e-6))
        terms.append(torch.stack(cc).mean() if cc else pp[b].sum()*0)
    d2=torch.stack(terms).mean();COST['autograd_requests']+=2
    g1=torch.autograd.grad(ce+dice,logits,retain_graph=True)[0];g2=torch.autograd.grad(ce2+d2,logits)[0]
    out['CE_Dice_value_error']=float(abs(ce+dice-ce2-d2).detach());out['CE_Dice_gradient_error']=float(abs(g1-g2).max());assert max(out['CE_Dice_value_error'],out['CE_Dice_gradient_error'])<2e-6
    # Synthetic triangular reparameterization through fixed semantic logits and unchanged task risk.
    w=initial('linear',torch.float64).requires_grad_(True);means,sd,kk=head(w,im.detach()[:3],'linear')
    eps=torch.tensor([[-1.,.5],[.2,-.4],[1.1,.7],[-.3,-.8]],dtype=torch.float64)
    c=means[:,0][None]+sd[:,0][None]*eps[:,0,None];r=means[:,1][None]-kk[None]*(c-means[:,0][None])+sd[:,1][None]*eps[:,1,None]
    h=c+r;z=h[:,:,None,None,None]*torch.tensor([-.7,.1,.4,1.2])[None,None,:,None,None]+torch.cos(torch.arange(16,dtype=torch.float64)).reshape(1,1,1,4,4)
    risk=torch.stack([F.cross_entropy(v,y) for v in z]).mean()+ns['foreground_dice_loss'](z.softmax(2).mean(0),y)
    COST['autograd_requests']+=1;gg=torch.autograd.grad(risk,w)[0];assert torch.isfinite(gg).all()
    out['synthetic_task_gradient_norm']=float(gg.norm());out['synthetic_kappa_gradient_norm']=float(gg[4].norm())
    out['all_background_dice']=float(ns['foreground_dice_loss'](pp[:1].detach(),y[:1]));assert out['all_background_dice']==0
    return out


def main():
    path=ROOT/'results'/'BAYESEG_G_NEURAL_KAPPA_SYNTHETIC.json'
    assert not path.exists(),'create-only result; preserve earlier execution'
    out={'protocol':'NEURAL_KAPPA_SYNTHETIC_PREREG_ZH.md','runs':[],'cost':COST,'formal_training':False,'real_data_reads':0}
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        out['checks']=checks();save()
        for kind in ['sigmoid','linear']:
            for solver in ['L-BFGS-B','Adam']:
                out['runs'].append(fit(kind,solver));save()
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except Exception as e:
        out['failure']=repr(e);save();raise
    print(json.dumps({'checks':out['checks'],'runs':[{k:r[k] for k in ['kind','solver','passed','max_KL']} for r in out['runs']],'cost':COST}))

if __name__=='__main__':main()
