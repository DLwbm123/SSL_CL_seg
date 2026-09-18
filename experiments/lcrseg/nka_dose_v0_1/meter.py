"""Read-only FP32 Adam arithmetic; never invokes optimizer/functional Adam."""
import math,time,hashlib,importlib
import torch
from .protocol import ADAM_SOURCE,METER_VERSION,TOLERANCE
from ..lctx_weight_memory_v0_1.core import LAYERS


def validate_adam(optimizer):
    if type(optimizer) is not torch.optim.Adam or torch.__version__.split('+')[0]!='2.2.1':raise ValueError('unsupported Adam version/type')
    module=importlib.import_module('torch.optim.adam')
    if hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()!=ADAM_SOURCE:raise ValueError('Adam implementation changed')
    for g in optimizer.param_groups:
        if any(g.get(k,False) for k in ('amsgrad','capturable','differentiable','maximize','fused')):raise ValueError('unsupported Adam flags')
        if g['foreach'] not in (None,False,True) or type(g['lr']) not in (int,float) or not math.isfinite(g['lr']) or g['lr']<0:raise ValueError('unsupported lr/path')
        if g['betas']!=(.9,.999) or g['eps']!=1e-8 or g['weight_decay']!=4e-5:raise ValueError('unexpected Adam options')
        for p in g['params']:
            if p.dtype!=torch.float32 or p.is_complex():raise ValueError('FP32 real parameters required')


def sync(params):
    if params and params[0].is_cuda:torch.cuda.synchronize(params[0].device)


@torch.no_grad()
def candidate(optimizer,grads,names):
    """Clone-owned arithmetic matches v2.2.1 single/foreach noncapturable branches."""
    from torch.optim.optimizer import _default_to_fused_or_foreach
    validate_adam(optimizer)
    params=[p for g in optimizer.param_groups for p in g['params']];sync(params);started=time.perf_counter()
    values={names[id(p)]:p.detach().clone() for p in params};states={};paths=[];ops=0;elements=0
    for g in optimizer.param_groups:
        selected=[p for p in g['params'] if grads[id(p)] is not None]
        foreach=g['foreach']
        if foreach is None:_,foreach=_default_to_fused_or_foreach(selected,False,use_fused=False)
        paths.append(dict(group=g.get('name','unnamed'),foreach=foreach,lr=g['lr'],active_parameters=len(selected)))
        pp=[];gg=[];mm=[];vv=[];tt=[]
        for p in selected:
            grad=grads[id(p)]
            if grad.is_sparse or not torch.isfinite(grad).all():raise ValueError('invalid diagnostic gradient')
            st=optimizer.state.get(p,{})
            if st and set(st)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('unsupported optimizer state')
            step=st['step'].detach().clone() if st else torch.tensor(0.,device='cpu')
            if step.device.type!='cpu' or step.numel()!=1 or step.item()<0:raise ValueError('noncapturable CPU step required')
            pp.append(values[names[id(p)]]);gg.append(grad.detach().clone())
            mm.append(st['exp_avg'].detach().clone() if st else torch.zeros_like(p))
            vv.append(st['exp_avg_sq'].detach().clone() if st else torch.zeros_like(p));tt.append(step)
            elements+=p.numel()
        if not pp:continue
        b1,b2=g['betas'];lr=g['lr'];eps=g['eps'];wd=g['weight_decay']
        if foreach:
            torch._foreach_add_(tt,torch.tensor(1.,device='cpu'),alpha=1.)
            gg=torch._foreach_add(gg,pp,alpha=wd)
            torch._foreach_lerp_(mm,gg,1-b1);torch._foreach_mul_(vv,b2);torch._foreach_addcmul_(vv,gg,gg,1-b2)
            den=torch._foreach_sqrt(vv);torch._foreach_div_(den,[(1-b2**t.item())**.5 for t in tt]);torch._foreach_add_(den,eps)
            torch._foreach_addcdiv_(pp,mm,den,[-lr/(1-b1**t.item()) for t in tt]);ops+=9
        else:
            for p,grad,m,v,t in zip(pp,gg,mm,vv,tt):
                t+=1;grad=grad.add(p,alpha=wd);m.lerp_(grad,1-b1);v.mul_(b2).addcmul_(grad,grad,value=1-b2)
                den=(v.sqrt()/((1-b2**t.item())**.5)).add_(eps)
                p.addcdiv_(m,den,value=-lr/(1-b1**t.item()));ops+=9
        for p,m,v,t in zip(selected,mm,vv,tt):states[names[id(p)]]=dict(exp_avg=m,exp_avg_sq=v,step=t)
    sync(params);elapsed=time.perf_counter()-started
    from ..ssl_anchored_mix_v0_1 import telemetry
    if telemetry.ACTIVE is not None:
        telemetry.ACTIVE.counts.update(adam_read_only_candidates=1,adam_arithmetic_calls=ops,adam_candidate_seconds=elapsed)
    return values,states,dict(seconds=elapsed,arithmetic_calls=ops,active_elements=elements,paths=paths)


def norm(xs):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in xs))


def ratio(before,zero,active):
    denominator=norm([b-a for a,b in zip(before,zero)]);numerator=norm([c-b for b,c in zip(zero,active)])
    resolution=max(TOLERANCE['ratio_floor'],TOLERANCE['ratio_resolution_ulps']*torch.finfo(torch.float32).eps*norm(before))
    reason='zero_or_below_FP32_resolution' if denominator<=resolution else None
    return dict(rho=None if reason else numerator/denominator,reason=reason,numerator=numerator,denominator=denominator,
                active_increment=norm([c-a for a,c in zip(before,active)]),resolution=resolution)


def effective(base,a,b,v):return base+(b@(a-(a@v)@v.T)).reshape_as(base)


def scopes(names):
    layer=lambda n:n.removeprefix('parent.native.').rsplit('.',1)[0]+'.weight'
    groups={'all_AB':list(names),'upstream_AB':[n for n in names if layer(n) in LAYERS[:-1]]}
    for factor in ('A','B'):
        groups[factor]=[n for n in names if n.endswith('.'+factor)]
        groups['upstream_'+factor]=[n for n in groups['upstream_AB'] if n.endswith('.'+factor)]
    for l in LAYERS:
        ns=[n for n in names if layer(n)==l];groups[l]=ns
        for factor in ('A','B'):groups[l+'/'+factor]=[n for n in ns if n.endswith('.'+factor)]
    return groups


def measure(t,labeled,kl,align,unlabeled):
    named={n:p for n,p in t.model.named_parameters() if p.requires_grad};params=list(named.values());names={id(p):n for n,p in named.items()}
    allowed={id(p) for p in t.model.u_parameters()};vjp={};sync(params);started=time.perf_counter()
    for label,loss in [('supervised',labeled),('KL',kl),('alignment',align),('U_actual',unlabeled)]:
        t.alignment_cost['diagnostic_vjps']+=1
        values=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
        vjp[label]={id(p):None if g is None else g.detach() for p,g in zip(params,values)}
    def merge(which):
        result={}
        for p in params:
            a=vjp['supervised'][id(p)];b=vjp[which][id(p)] if id(p) in allowed else None
            result[id(p)]=None if a is None and b is None else (torch.zeros_like(p) if a is None else a)+(0 if b is None else b)
        return result
    before={n:p.detach().clone() for n,p in named.items()}
    zero,_,c0=candidate(t.optimizer,merge('KL'),names);active,state,ca=candidate(t.optimizer,merge('U_actual'),names)
    t.alignment_cost['adam_candidates']+=2
    t.alignment_cost['adam_arithmetic_calls']+=c0['arithmetic_calls']+ca['arithmetic_calls']
    group=scopes(named);theta={key:ratio([before[n] for n in ns],[zero[n] for n in ns],[active[n] for n in ns]) for key,ns in group.items()}
    sync(params);w_started=time.perf_counter();w_evaluations=0;weights={}
    # A/B scope substitutions hold other factors at entry; full AB includes the cross term.
    for key,ns in group.items():
        selected=set(ns);b0=[];bz=[];ba=[]
        for l in LAYERS:
            prefix='parent.native.'+l[:-7];an=prefix+'.A';bn=prefix+'.B'
            if an not in selected and bn not in selected:continue
            layer=t.model.parent.native.get_submodule(l[:-7]);base=layer.weight.detach();v=layer.right.detach()
            def w(values):return effective(base,values[an] if an in selected else before[an],values[bn] if bn in selected else before[bn],v)
            b0.append(w(before));bz.append(w(zero));ba.append(w(active));w_evaluations+=3
        weights[key]=ratio(b0,bz,ba)
    sync(params);w_seconds=time.perf_counter()-w_started
    t.alignment_cost['effective_W_evaluations']+=w_evaluations;t.alignment_cost['effective_W_seconds']+=w_seconds
    gradients={}
    for key,ns in group.items():
        vectors={label:torch.cat([(torch.zeros_like(named[n]) if values[id(named[n])] is None else values[id(named[n])]).flatten() for n in ns]) for label,values in vjp.items()}
        norms={k:float(v.double().norm()) for k,v in vectors.items()};angles={}
        for k in ('supervised','KL','U_actual'):
            cos=None if norms['alignment']==0 or norms[k]==0 else float(torch.dot(vectors['alignment'].double(),vectors[k].double())/(norms['alignment']*norms[k]))
            angles[k]=dict(cosine=cos,angle_degrees=None if cos is None else math.degrees(math.acos(max(-1.,min(1.,cos)))))
        gradients[key]=dict(norms=norms,alignment_vs=angles)
    sync(params);elapsed=time.perf_counter()-started;t.alignment_cost['meter_seconds']+=elapsed
    from ..ssl_anchored_mix_v0_1 import telemetry
    if telemetry.ACTIVE is not None:
        telemetry.ACTIVE.counts.update(meter_extra_vjps=4,effective_W_evaluations=w_evaluations,effective_W_seconds=w_seconds,meter_seconds=elapsed)
        telemetry.flush('read_only_meter_complete')
    record=dict(version=METER_VERSION,step=t.step+1,scope='local one-step counterfactual at current dose state; not a no-auxiliary trajectory',
                gradients=gradients,rho_theta=theta,rho_W=weights,candidates=[c0,ca],extra_vjps=4,read_only_candidates=2,
                meter_seconds=elapsed,effective_W_evaluations=w_evaluations,effective_W_seconds=w_seconds,losses={k:float(x.detach()) for k,x in [('supervised',labeled),('KL_weighted',kl),('alignment_weighted',align),('U_actual',unlabeled)]})
    return record,(before,active,state,named)


def verify_prediction(pending,optimizer):
    before,expected,states,named=pending;errors={};maximum=0.
    for name,p in named.items():
        actual=p.detach();pred=expected[name];error=float((actual-pred).abs().max())
        inc_actual=actual-before[name];inc_pred=pred-before[name];inc_error=float((inc_actual-inc_pred).abs().max());scale=float(inc_actual.abs().max())
        if error>TOLERANCE['parameter_atol'] or inc_error>TOLERANCE['increment_atol']+TOLERANCE['increment_rtol']*scale:raise FloatingPointError('Adam candidate/actual mismatch: '+name)
        if name in states:
            for key,x in states[name].items():
                y=optimizer.state[p][key]
                if not torch.allclose(x.to(y),y,rtol=2e-5,atol=1e-9):raise FloatingPointError('Adam moment mismatch: '+name+'/'+key)
        errors[name]=dict(parameter_max_abs=error,increment_max_abs=inc_error,actual_increment_max_abs=scale,
                         increment_relative=None if scale==0 else inc_error/scale)
        maximum=max(maximum,error)
    return dict(passed=True,parameter_max_abs=maximum,parameters=errors,tolerance=TOLERANCE)
