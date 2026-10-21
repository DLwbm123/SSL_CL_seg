"""Five-arm fixed-endpoint synthetic mechanism test; no medical training."""
import json,time,signal
from pathlib import Path
import torch
import torch.nn.functional as F
import validate_nonlinear_stochastic as v
from bounded_lbfgs import make_optimizer,accepted_step,self_check,BudgetExhausted
ROOT=Path(__file__).parent

def features(w,im):
    a,b,c,d,_,_=w.double().split(v.SIZES);z=im.double().mean((-1,-2));z=torch.stack([(z[:,0]+.5)/1.5,z[:,1]/2],1)
    z=torch.tanh(F.linear(z,a.reshape(32,2),b));z=torch.tanh(F.linear(z,c.reshape(32,32),d))
    return torch.cat([z,torch.ones(12,1,dtype=torch.float64)],1)

def reference(w,im):
    w=w.double();ii=im.double();f=features(w,ii);sv=torch.linalg.svdvals(f)
    t=10**ii[:,0,0,0];x=ii[:,1,0,0]
    sd=torch.stack([torch.sqrt((1+t*t)/(2+t*t)),torch.sqrt(t*t/(1+t*t))],1)
    target=torch.column_stack([x/(2+t*t),x/(2+t*t),(torch.log(torch.expm1(sd-1e-6))-v.B0)/4,1/(1+t*t)-.01])
    coef=torch.linalg.lstsq(f,target,driver='gelsd').solution
    assigned=torch.cat([w[:1152],coef[:-1].T.reshape(-1),coef[-1]])
    return dict(rank=int((sv>sv.max()*1e-12).sum()),singular_values=sv.tolist(),FP64=v.evaluate(assigned,ii),FP32_rounded=v.evaluate(assigned.float(),ii))

def gradient_blocks(w,im):
    w=w.detach().clone().requires_grad_(True);g=torch.autograd.grad(v.terms(w,im,torch.float64)[0].mean(),w)[0]
    head=torch.cat([g[1152:1312].reshape(5,32),g[1312:,None]],1)
    mu,_,_=v.outputs(w,im,torch.float64)
    return dict(hidden_norm=float(g[:1152].norm()),head_row_norms=head.norm(dim=1).tolist(),mean_output_difference=float((mu[:,0]-mu[:,1]).abs().max().detach()))

def run(name,start,im):
    beg=time.monotonic();w=start.double() if name=='J64' else start.clone();counter={'calls':0};trace=[];failure=None;optimizer=None;last_mask=None
    def mask_for(i):
        if name=='H' or (name=='A' and (i//20)%2==0):return torch.arange(1152,1317)
        if name=='A':return torch.arange(1152)
        return torch.arange(1317)
    for i in range(200):
        mask=mask_for(i)
        if optimizer is None or (name in ['JR','A'] and i%20==0):
            param=w[mask].clone().requires_grad_(True);fixed=w.clone();optimizer=make_optimizer(param);last_mask=mask
        def assemble(p):return fixed.scatter(0,last_mask,p)
        before=w.clone()
        try:move=accepted_step(optimizer,param,lambda p:v.terms(assemble(p),im,torch.float64)[0].mean(),counter,400)
        except RuntimeError as e:
            failure=str(e);w=assemble(param.detach()).detach();break
        w=assemble(param.detach()).detach()
        if move==0:failure='no parameter displacement';break
        ev=v.evaluate(w,im)
        trace.append(dict(iteration=i+1,calls=counter['calls'],passed=ev['passed'],max_KL=ev['max_KL'],mean_KL=sum(q['KL'] for q in ev['cases'])/12,step_norm=move,hidden_step_norm=float((w-before)[:1152].norm()),head_step_norm=float((w-before)[1152:].norm())))
    at_budget={}
    for b in [100,200,300,400]:
        rows=[r for r in trace if r['calls']<=b];at_budget[str(b)]=rows[-1] if rows else None
    return dict(name=name,calls=counter['calls'],sample_gradient_evaluations=12*counter['calls'],iterations=len(trace),extra_endpoint_batch_evaluations=len(trace)+1,seconds=time.monotonic()-beg,stop=failure or 'iteration_limit',trace=trace,at_call_budget=at_budget,endpoint=v.evaluate(w,im),gradient_blocks=gradient_blocks(w,im),hidden_total_displacement=float((w-start)[:1152].norm()),weights=w.tolist())

def mean_diagnostic(start,im):
    w=start.double();indices=torch.cat([torch.arange(1152,1216),torch.arange(1312,1314)])
    p=w[indices].clone().requires_grad_(True)
    def full(q):return w.scatter(0,indices,q)
    loss=v.terms(full(p),im,torch.float64)[0].mean();g=torch.autograd.grad(loss,p,create_graph=True)[0]
    r=-g.detach();direction=r.clone();delta=torch.zeros_like(r);rr=r@r;initial=float(rr.sqrt());calls=0;stop='hvp_limit'
    for _ in range(100):
        hv=torch.autograd.grad((g*direction).sum(),p,retain_graph=True)[0].detach();calls+=1;curvature=direction@hv
        if float(curvature)<=0:stop='nonpositive_curvature';break
        alpha=rr/curvature;delta+=alpha*direction;r-=alpha*hv;newrr=r@r
        if float(newrr.sqrt())<=1e-10*max(initial,1e-30):stop='relative_residual';rr=newrr;break
        direction=r+(newrr/rr)*direction;rr=newrr
    end=full(p.detach()+delta);ev=v.evaluate(end,im)
    return dict(hvp_calls=calls,gradient_requests=1,updates=1,stop=stop,initial_gradient_norm=initial,residual_norm=float(rr.sqrt()),endpoint=ev,mean_max=max(q['mean_KL'] for q in ev['cases']),mean_gate_pass=max(q['mean_KL'] for q in ev['cases'])<=1e-6,weights=end.tolist())

def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600 second hard limit')));signal.alarm(600)
    path=ROOT/'results/BAYESEG_G_BLOCK_MECHANISM.json';assert not path.exists()
    src=json.loads((ROOT/'results/BAYESEG_G_NONLINEAR_STOCHASTIC.json').read_text())
    chosen=next(r for r in src['runs'] if r['compute']=='torch.float64' and not r['stochastic'])
    start=torch.tensor(chosen['weights'],dtype=torch.float32);_,im=v.base.data(torch.float32)
    out=dict(protocol='BLOCK_MECHANISM_PREREG_ZH.md',formal_training=False,real_data_reads=0,restore_tests=self_check(),zero_update_checks=[dict(name='initial',**reference(v.initial(),im))]+[dict(name=r['compute']+str(r['stochastic']),**reference(torch.tensor(r['weights']),im)) for r in src['runs']],common_start=v.evaluate(start,im),common_gradient_blocks=gradient_blocks(start,im),runs=[])
    assert abs(out['common_start']['max_KL']-chosen['endpoint']['max_KL'])<1e-12
    def save():path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for name in ['J','JR','H','A','J64']:
            r=run(name,start,im);out['runs'].append(r);save();print(name,r['calls'],r['iterations'],r['endpoint']['passed'],r['endpoint']['max_KL'],r['stop'],flush=True)
        h=next(r for r in out['runs'] if r['name']=='H')
        if h['endpoint']['passed']<12:
            out['mean_diagnostic']=mean_diagnostic(torch.tensor(h['weights'],dtype=torch.float64),im);save()
            print('mean diagnostic',out['mean_diagnostic']['mean_max'],out['mean_diagnostic']['hvp_calls'],flush=True)
        out['status']='COMPLETED_SYNTHETIC_ONLY';save()
    except BaseException as e:
        out['status']='FAILED';out['failure']=repr(e);save();raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
