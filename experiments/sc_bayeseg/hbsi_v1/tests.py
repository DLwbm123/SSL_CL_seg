"""Deterministic algebra, predeclared MC checks and implementation negative controls."""
import copy
from dataclasses import replace
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch
import numpy as np
import torch
from torch.distributions import Normal, StudentT, MultivariateNormal
from .budgets import ResourceLedger, BudgetExceeded
from .config import Config
from .data_adapter import AccessGate, leaf_observations, confined
from .diagnostics import consistency, parameter_audit
from .distributions import normal_log_prob, student_log_prob, categorical_log_prob
from .local_inference import Refiner
from .objective import SequentialObjective, sum_estimate
from .predictive import Predictor, InferenceBudget, combine_domains, ess
from .spatial_gaussian import sar, sar_log_prob, dct_basis
from .stream_state import StreamState, digest

SEED = 20261022
ATOL = 1e-8
RTOL = 1e-6


def check_close(actual, expected, atol=ATOL, rtol=RTOL):
    a, b = torch.as_tensor(actual), torch.as_tensor(expected)
    if not torch.allclose(a, b.to(a), atol=atol, rtol=rtol):
        raise AssertionError(f'max error {float((a-b.to(a)).abs().max())}, atol={atol}, rtol={rtol}')
    return float((a-b.to(a)).abs().max())


def rejected(callback, kind=(ValueError, PermissionError, RuntimeError)):
    try:
        callback()
    except kind:
        return True
    raise AssertionError('negative control accepted')


def tiny(**kw):
    return StreamState(Config(height=4, width=4, r_height=2, r_width=2, dct_order=2, hidden=2, **kw))


def batches(s):
    c = s.config
    x = torch.linspace(-.8, .8, c.channels*c.height*c.width, dtype=torch.float64).reshape(1,c.channels,c.height,c.width)
    y = (x[:,0] > 0).long()
    return ({'x':x, 'observations':leaf_observations(y,c.classes), 'stage':s.active['id'], 'role':'trainL'},
            {'x':x.flip(-1), 'stage':s.active['id'], 'role':'trainU'})


def T01(out):
    g=torch.Generator().manual_seed(SEED); errors=[]
    for channels in (1,3):
        for h,w in ((4,4),(4,6),(8,8)):
            x=torch.randn(2,channels,h,w,generator=g,dtype=torch.float64)
            m=x*.17; sd=torch.linspace(.2,.7,channels,dtype=x.dtype)[None,:,None,None]
            errors.append(check_close(normal_log_prob(x,m,sd),Normal(m,sd).log_prob(x)))
            errors.append(check_close(student_log_prob(x,m,sd),StudentT(4.,m,sd).log_prob(x)))
            n=h*w; eye=torch.eye(n,dtype=x.dtype)
            A=sar(eye.reshape(n,1,h,w)).reshape(n,n).T
            Q=A.T@A/.2**2
            ref=sum(MultivariateNormal(m[:,k].reshape(2,n),precision_matrix=Q).log_prob(x[:,k].reshape(2,n)) for k in range(channels))
            errors.append(check_close(sar_log_prob(x,m),ref))
            B=dct_basis(h,w,min(4,h,w)).flatten(1)
            errors.append(check_close(B@B.T,torch.eye(len(B),dtype=x.dtype)))
            errors.append(check_close(B.sum(1),torch.zeros(len(B),dtype=x.dtype)))
    for k in (2,3,4):
        prob=torch.randn(2,k,4,6,generator=g,dtype=torch.float64).softmax(1)
        y=torch.randint(k,(2,4,6),generator=g)
        v=categorical_log_prob(prob,**{'allowed':leaf_observations(y,k)['y_obs']})
        ref=torch.distributions.Categorical(probs=prob.permute(0,2,3,1)).log_prob(y).sum((1,2))
        errors.append(check_close(v,ref))
        state=tiny(channels=3,classes=k)
        theta=state.q.rsample(state.generator);a=torch.zeros(9,dtype=torch.float64)
        image=torch.zeros(1,3,4,4,dtype=torch.float64)
        ctx=state.phi.context(image,theta,a)
        z,lq=state.phi.rsample_and_log_prob(ctx,state.phi.noise(1,image,state.generator),state.phi.initial_offsets(1,image))
        joint=state.model.log_joint(image,z,theta,a)
        assert joint.total.shape==(1,) and lq.shape==(1,) and torch.isfinite(joint.total).all()
    return {'max_error':max(errors),'threshold':{'atol':ATOL,'rtol':RTOL}}


def T02(out):
    s=tiny(classes=4); s.begin_task('one','0'*64); L,U=batches(s)
    x=U['x']; theta=s.q.rsample(s.generator); a=s.current.rsample(theta,s.generator)
    ctx=s.phi.context(x,theta,a); rho=s.phi.initial_offsets(1,x)
    z,lq=s.phi.rsample_and_log_prob(ctx,s.phi.noise(1,x,s.generator),rho)
    p=s.model.class_probabilities(z,theta)
    empty=torch.zeros_like(x[:,0],dtype=torch.bool)
    zero=categorical_log_prob(p,torch.ones_like(p,dtype=torch.bool),empty)
    assert zero.item()==0 and torch.equal(torch.autograd.grad(zero.sum(),p,retain_graph=True)[0],torch.zeros_like(p))
    leaves=torch.zeros_like(p,dtype=torch.bool); leaves[:,1:3]=True
    value=categorical_log_prob(p,leaves)
    error=check_close(value,(p[:,1]+p[:,2]).log().sum((1,2)))
    codes=torch.full((1,4,4),4,dtype=torch.long);codes[:,0,0]=-1
    mapped=leaf_observations(codes,4,((1,2),))
    mapped_value=categorical_log_prob(p,mapped['y_obs'],mapped['obs_mask'])
    ref=(p[:,1]+p[:,2]).log();ref=ref.clone();ref[:,0,0]=0
    error=max(error,check_close(mapped_value,ref.sum((1,2))))
    gate=AccessGate('one'); touched=[]
    def hidden():
        touched.append(True); raise AssertionError('hidden label read')
    rejected(lambda:gate.open({'stage':'one','role':'trainU'},'label',hidden),PermissionError)
    assert not touched
    return {'partial_label_error':error,'empty_label_value':0.,'empty_label_gradient':0.,'hidden_loader_calls':0,'configured_parent_and_missing_codes':True}


def T03(out):
    s=tiny(); g=s.generator; e=torch.randn(s.q.mean.shape,generator=g,dtype=torch.float64)
    theta=s.q.rsample(noise=e); other=s.q.rsample(noise=-e)
    z={'c':torch.ones(1,2,dtype=torch.float64),'R':torch.ones(1,2,2,2,dtype=torch.float64)}
    p=s.model.class_probabilities(z,theta); f=s.model.structure_mean(p,theta)
    assert torch.equal(theta,s.q.rsample(noise=e))
    delta=(p-s.model.class_probabilities(z,other)).abs().max()
    deltaf=(f-s.model.structure_mean(p,other)).abs().max()
    assert delta>1e-8 and deltaf>1e-8
    gradients=torch.autograd.grad(p[:,1].sum()+f.square().sum(),(s.q.mean,s.q.raw_scale))
    assert all(v.abs().sum()>0 for v in gradients)
    dummy=torch.zeros(3,requires_grad=True)
    unused=torch.autograd.grad(p.sum(),dummy,allow_unused=True,retain_graph=True)[0]
    assert unused is None
    return {'G_noise_effect':float(delta),'F_noise_effect':float(deltaf),
            'mean_gradient_norm':float(gradients[0].norm()),'scale_gradient_norm':float(gradients[1].norm()),'dummy_rejected':True}


def T04(out):
    s=tiny(); s.begin_task('one','0'*64)
    with torch.no_grad(): s.current.C.copy_(torch.arange(s.current.C.numel(),dtype=torch.float64).reshape_as(s.current.C)*.07)
    theta=s.q.rsample(s.generator); p=s.projection
    errors=[check_close(p(theta),p.dense()@theta)]
    m,cov=s.current.marginal(s.q); before=digest(s.current.state_dict())
    H=s.current.C@p.dense()
    errors.append(check_close(cov,torch.diag(s.current.sd.square())+(H*s.q.sd.square())@H.T))
    with torch.no_grad(): s.q.mean.add_(.2); s.q.raw_scale.add_(.1)
    m2,c2=s.current.marginal(s.q)
    assert (m-m2).norm()>1e-8 and (cov-c2).norm()>1e-8 and digest(s.current.state_dict())==before
    s.save(out/'state.pt'); re=StreamState.load(out/'state.pt')
    assert p.fingerprint()==re.projection.fingerprint()
    errors.append(check_close(s.current.mean(theta),re.current.mean(theta)))
    with torch.no_grad(): s.current.C.zero_()
    errors.append(check_close(s.current.mean(theta),s.current.mean(theta+3)))
    return {'max_error':max(errors),'marginal_mean_change':float((m2-m).norm()),'marginal_cov_change':float((c2-cov).norm())}


def T05(out):
    s=tiny(); s.begin_task('one','0'*64)
    with torch.no_grad(): s.current.C.fill_(.2); s.q.raw_scale.add_(1.)
    n=32768; theta=s.q.mean.detach()+s.q.sd.detach()*torch.randn(n,s.q.mean.numel(),generator=s.generator,dtype=torch.float64)
    mean=s.current.mean(theta).detach(); sd=s.current.sd.detach()
    vals=.5*(sd.square()+mean.square()-1-2*sd.log()).sum(-1)
    exact=s.current.expected_kl(s.q).detach(); se=vals.std()/math.sqrt(n)
    err=(vals.mean()-exact).abs(); omitted=.5*s.current.projection.variance(s.q.sd.square()).mul(s.current.C.square()).sum()
    assert err<=6*se+1e-6 and omitted>1e-4
    return {'samples':n,'absolute_error':float(err),'mc_standard_error':float(se),'threshold':'6*SE+1e-6','omitted_variance_term':float(omitted)}


def T06(out):
    g=torch.Generator().manual_seed(SEED)
    theta=torch.randn(2000,generator=g,dtype=torch.float64)*.7+.2
    old=theta*.6+torch.randn(2000,generator=g,dtype=torch.float64)*.4
    a=theta*.3+.1+torch.randn(2000,generator=g,dtype=torch.float64)*.5
    z=theta+a+torch.randn(2000,generator=g,dtype=torch.float64)*.8
    lold=Normal(theta*.6,.4).log_prob(old)
    lq=Normal(.2,.7).log_prob(theta); lp=Normal(-.1,1.2).log_prob(theta)
    lr=Normal(.3*theta+.1,.5).log_prob(a); la=Normal(0.,1.).log_prob(a)
    lz=Normal(theta+a,.8).log_prob(z); lj=Normal(0.,1.).log_prob(z)+Normal(theta+a+z,.3).log_prob(torch.ones_like(z))
    full=lq+lold+lr+lz-(lp+lold+la+lj)
    expanded=lq-lp+lr-la-(lj-lz)
    err=check_close(full,expanded)
    negative=float((full-(expanded+lold)).abs().mean()); assert negative>.1
    return {'pointwise_max_error':err,'extra_old_penalty_error':negative,'normalization_constant':'0: both expressions use unnormalized target'}


def gaussian_reference():
    # Independent NumPy precision-matrix reference: all observations jointly.
    obs=[np.array([.2,.5,.4]),np.array([1.3,1.1])]
    design=np.array([[1,1,0]]*3+[[1,0,1]]*2,dtype=float)
    precision=np.diag([1.,1/.49,1/.49])+design.T@design/.04
    cov=np.linalg.inv(precision); mean=cov@design.T@np.concatenate(obs)/.04
    # Scalar marginal updates plus immutable per-domain exact conditional kernels.
    v=1.; mu=0.; kernels=[]
    for data in obs:
        noise=.49+.04/len(data); nv=1/(1/v+1/noise)
        mu=nv*(mu/v+data.mean()/noise); v=nv
        vv=1/(1/.49+len(data)/.04)
        kernels.append((vv*data.sum()/.04,-vv*len(data)/.04,vv))
    outmean=np.array([mu]+[m+c*mu for m,c,_ in kernels])
    H=np.array([1.]+[c for _,c,_ in kernels])
    outcov=v*np.outer(H,H)+np.diag([0.]+[vv for _,_,vv in kernels])
    return obs,mean,cov,outmean,outcov


def T07(out):
    obs,mean,cov,sm,sc=gaussian_reference()
    err=max(np.max(abs(mean-sm)),np.max(abs(cov-sc))); assert err<=1e-8
    n0=.49+.04/len(obs[0]); v0=1/(1+1/n0); m0=v0*obs[0].mean()/n0
    fixed_old=.49/(.49+.04/3)*(obs[0].mean()-m0)
    fixed_error=abs(fixed_old-mean[1])
    repeat_v=1/(1+2/n0+1/(.49+.04/2))
    repeat_mean=repeat_v*(2*obs[0].mean()/n0+obs[1].mean()/(.49+.04/2))
    reset_mean=obs[1].mean()/(1+.49+.04/2)
    negative=[fixed_error,abs(repeat_mean-mean[0]),abs(reset_mean-mean[0])]
    assert min(negative)>1e-3
    return {'mean_max_error':float(np.max(abs(mean-sm))),'cov_max_error':float(np.max(abs(cov-sc))),
            'threshold':1e-8,'negative_control_errors':list(map(float,negative))}


def T08(out):
    L=torch.tensor([1.,3.,7.],dtype=torch.float64); U=torch.tensor([2.,8.],dtype=torch.float64)
    vals=[5.-sum_estimate(L[list(i)],3)-sum_estimate(U[list(j)],2) for i in itertools.product(range(3),repeat=2) for j in itertools.product(range(2),repeat=1)]
    err=check_close(torch.stack(vals).mean(),5.-L.sum()-U.sum())
    s=tiny(global_mc=1,local_mc=1,inner_steps=0); s.begin_task('one','0'*64); bL,bU=batches(s)
    ref=Refiner(s.model,s.phi); objective=SequentialObjective(s,ref)
    fixedtheta=s.q.rsample(s.generator); fixeda=s.current.rsample(fixedtheta,s.generator)
    def zero_noise(batch,like,generator): return {k:torch.zeros(shape,dtype=like.dtype) for k,shape in s.phi.shapes(batch).items()}
    with patch.object(s.q,'rsample',return_value=fixedtheta),patch.object(s.current,'rsample',return_value=fixeda),patch.object(s.phi,'noise',side_effect=zero_noise):
        one,_=objective.estimate(bL,bU,{'L':3,'U':2},s.generator)
        s.config=replace(s.config,global_mc=2,local_mc=2)
        two,_=objective.estimate(bL,bU,{'L':3,'U':2},s.generator)
    return {'enumeration_error':err,'double_MC_error':check_close(one,two)}


def T09(out):
    s=tiny(); s.begin_task('one','0'*64); bL,bU=batches(s); x=bL['x']
    ref=Refiner(s.model,s.phi)
    def value(detach=False):
        g=torch.Generator().manual_seed(8347)
        theta=s.q.rsample(g); a=s.current.rsample(theta,g)
        ctx,rho=ref.refine(x,theta,a,bL['observations'],K=3,training=True,generator=g)
        if detach: rho={k:v.detach() for k,v in rho.items()}
        e,*_=ref.draw(x,theta,a,ctx,rho,s.phi.noise(1,x,g),bL['observations'])
        return e.sum()
    params=[s.phi.c_head.bias,s.q.mean,s.q.raw_scale,s.current.m,s.current.C,s.current.raw_scale]
    grads=torch.autograd.grad(value(),params)
    rows=[]; epsilon=1e-5
    for name,p,grad in zip(('phi','global_mean','global_raw_scale','domain_mean','domain_C','domain_raw_scale'),params,grads):
        index=int(grad.abs().argmax()); original=p.detach().flatten()[index].item()
        with torch.no_grad(): p.flatten()[index]=original+epsilon
        plus=value().item()
        with torch.no_grad(): p.flatten()[index]=original-epsilon
        minus=value().item()
        with torch.no_grad(): p.flatten()[index]=original
        fd=(plus-minus)/(2*epsilon); ad=grad.flatten()[index].item()
        absolute=abs(fd-ad); relative=absolute/max(abs(fd),abs(ad),1e-8)
        assert absolute<=1e-6 or relative<=1e-4, (name,fd,ad,relative)
        rows.append(dict(parameter=name,finite_difference=fd,autograd=ad,relative_error=relative,absolute_error=absolute))
    broken=torch.autograd.grad(value(True),params)
    detach_effect=max(float((a-b).abs().max()) for a,b in zip(grads,broken)); assert detach_effect>1e-7
    g=torch.Generator().manual_seed(2); theta=s.q.rsample(g); a=s.current.rsample(theta,g)
    ctx,rho=ref.refine(x,theta,a,K=0,generator=g)
    noise=s.phi.noise(1,x,g)
    z,lq=s.phi.rsample_and_log_prob(ctx,noise,rho)
    _,lq0=s.phi.rsample_and_log_prob(s.phi.context(x,theta,a),noise,s.phi.initial_offsets(1,x))
    check_close(lq,lq0,atol=0,rtol=0)
    check_close(lq,s.phi.log_prob(z,ctx,rho))
    # Recompute conditional factors on changed parents, never cache old outputs.
    z2=dict(z,c=z['c']+1)
    assert (s.phi.parameters_given('R',ctx,z)-s.phi.parameters_given('R',ctx,z2)).abs().max()>1e-6
    return {'gradients':rows,'detached_negative_control_gradient_error':detach_effect,'K0_exact':True,'threshold':'relative<=1e-4 or abs<=1e-6'}


def _update(s):
    L,U=batches(s); obj=SequentialObjective(s,Refiner(s.model,s.phi))
    loss,_=obj.estimate(L,U,{'L':1,'U':1},s.generator)
    s.optimizer.zero_grad();loss.backward();s.optimizer.step();s.steps+=1
    from . import budgets
    if budgets.ACTIVE_COUNTER is not None:
        for k in ('global_updates','domain_updates','phi_updates','outer_backward_calls'):
            budgets.ACTIVE_COUNTER.add(k)


def T10(out):
    s=tiny(inner_steps=1,global_mc=1,local_mc=1);s.begin_task('one','0'*64)
    assert s.prior.mean.data_ptr()!=s.q.mean.data_ptr()
    prior=digest(s.prior.state_dict());_update(s);s.save(out/'resume.pt')
    restored=StreamState.load(out/'resume.pt');_update(s);_update(restored)
    assert digest(s.payload())==digest(restored.payload()) and digest(s.prior.state_dict())==prior
    s.finalize_task(); rejected(lambda:s.finalize_task());rejected(lambda:s.begin_task('one','0'*64))
    s.begin_task('two','1'*64)
    assert digest(s.prior.state_dict())==digest(s.q.state_dict())
    return {'resume_bitwise_equal':True,'prior_fixed':True,'duplicate_rejected':True}


def T11(out):
    # y=1[theta>0], theta~N(0,1), x|theta,d~N(theta+offset_d,.5²).
    x=.7; sigma=.5; offsets=[-1.,1.]; n=65536; repeats=4
    analytic=[]; evidence=[]
    for offset in offsets:
        var=1/(1+1/sigma**2);mu=var*(x-offset)/sigma**2
        analytic.append(Normal(0.,1.).cdf(torch.tensor(mu/math.sqrt(var),dtype=torch.float64)))
        evidence.append(Normal(offset,math.sqrt(1+sigma**2)).log_prob(torch.tensor(x,dtype=torch.float64)).exp())
    truth=sum(p*e for p,e in zip(analytic,evidence))/sum(evidence)
    estimates=[]; domain_known=[]
    for seed in range(repeats):
        g=torch.Generator().manual_seed(SEED+seed);ws=[];ps=[]
        for offset in offsets:
            theta=torch.randn(n,dtype=torch.float64,generator=g)
            ws.append(Normal(theta+offset,sigma).log_prob(torch.tensor(x,dtype=torch.float64))[:,None])
            yes=(theta>0).double();ps.append(torch.stack([1-yes,yes],1).reshape(n,1,1,2,1,1))
        p,_=combine_domains(ws,ps);estimates.append(p[0,1,0,0])
        known,_=combine_domains(ws[:1],ps[:1]);domain_known.append(known[0,1,0,0])
    err=float((torch.stack(estimates)-truth).abs().max())
    knownerr=float((torch.stack(domain_known)-analytic[0]).abs().max())
    assert err<.015 and knownerr<.015 and abs(truth-.5)>.03 and abs(truth-analytic[0])>.03
    return {'max_abs_error':err,'known_domain_error':knownerr,'threshold':.015,'samples_per_domain':n,'repeats':repeats,
            'analytic_probability':float(truth),'uniform_unconditional_error':float(abs(truth-.5)),'oracle_error':float(abs(truth-analytic[0]))}


def sealed():
    s=tiny();s.begin_task('one','0'*64);s.finalize_task();return s


def T12(out):
    s=sealed();x=torch.zeros(1,1,4,4,dtype=torch.float64)
    used=[];original=s.phi.noise
    def recording(batch,like,generator):
        result=original(batch,like,generator)
        used.append((generator.initial_seed(),digest(result)))
        return result
    with patch.object(s.phi,'noise',side_effect=recording):
        result=Predictor().predict_image_only(x,s,InferenceBudget(S=2,L=2,K=1,repeats=1))
    streams=result['repeats'][0]['streams'];assert len(set(streams.values()))==3
    assert [s for s,_ in used]==[streams['fit'],streams['importance'],streams['importance']]*2
    assert len(set(h for _,h in used))==len(used)
    return {'streams':streams,'fit_and_importance_independent':True,'actual_noise_calls_checked':len(used)}


def T13(out):
    s=sealed();x=torch.linspace(-1.,1.,32,dtype=torch.float64).reshape(2,1,4,4)
    b=InferenceBudget(S=2,L=2,K=1,repeats=1)
    one=Predictor().predict_image_only(x,s,b,slice_chunk=1)
    two=Predictor().predict_image_only(x,s,b,slice_chunk=2)
    err=check_close(one['probability'],two['probability'],atol=0,rtol=0)
    ratios=torch.tensor([[0.,2.,4.],[3.,1.,0.]],dtype=torch.float64)
    correct=ratios.sum(0).softmax(0)
    # A common wrong route averages already-conditioned slice predictions.
    wrong=ratios.softmax(1).mean(0)
    delta=float((correct-wrong).abs().max());assert delta>.05
    # Products of normalized weights ARE proportional if samples are aligned;
    # they lose evidence constants and cannot support domain mixing.
    normalized_product=ratios.log_softmax(1).sum(0)
    evidence_loss=float((torch.logsumexp(ratios.sum(0),0)-torch.logsumexp(normalized_product,0)).abs())
    assert evidence_loss>1
    return {'chunk_error':err,'slice_average_negative_error':delta,'normalized_product_lost_log_evidence':evidence_loss}


def T14(out):
    s=sealed();s.begin_task('two','1'*64)
    for p in s.phi.parameters(): p.grad=torch.ones_like(p)
    before=digest(s.payload());grads=digest([p.grad for p in s.phi.parameters()])
    tr=torch.random.get_rng_state().clone();nr=copy.deepcopy(np.random.get_state());rr=random.getstate()
    p=Predictor(); p.predict_image_only(torch.zeros(1,1,4,4,dtype=torch.float64),s,InferenceBudget(S=2,L=2,K=1,repeats=1))
    assert before==digest(s.payload()) and grads==digest([p.grad for p in s.phi.parameters()])
    assert torch.equal(tr,torch.random.get_rng_state()) and digest(nr)==digest(np.random.get_state()) and rr==random.getstate()
    assert not {'y','label','labels','domain','domain_id'} & set(inspect.signature(p.predict_image_only).parameters)
    return {'state_optimizer_buffers_grads_unchanged':True,'external_rng_unchanged':True,'no_label_or_domain_argument':True}


def T15(out):
    g=torch.Generator().manual_seed(SEED);e=torch.randn(2048,dtype=torch.float64,generator=g)
    x=.7;noise=.3;var=1/(1+1/noise**2);mu=var*x/noise**2
    z=mu+math.sqrt(var)*e
    normal=lambda m,s:Normal(torch.as_tensor(m,dtype=torch.float64),torch.as_tensor(s,dtype=torch.float64))
    joint=normal(0.,1.).log_prob(z)+normal(z,noise).log_prob(torch.tensor(x,dtype=z.dtype))
    exact=joint-normal(mu,math.sqrt(var)).log_prob(z)
    wrong=joint-normal(0.,1.).log_prob(z)
    a=consistency(exact);b=consistency(wrong)
    assert a['delta_variance']<1e-20 and b['delta_variance']>.1
    assert Normal(0.,math.sqrt(1+noise**2)).log_prob(torch.tensor(x))!=Normal(2.,math.sqrt(1+noise**2)).log_prob(torch.tensor(x))
    return {'exact_variance':float(a['delta_variance']),'wrong_variance':float(b['delta_variance']),
            'threshold':1e-20,'scope':'fixed theta/a only','diagnostic_gradient':False}


def T16(out):
    gate=AccessGate('current'); calls=[]
    def open_forbidden(): calls.append(1);raise AssertionError('access occurred')
    for stage,role,kind in [('old','trainL','image'),('current','val','image'),('current','test','image'),('current','trainU','label')]:
        rejected(lambda:gate.open({'stage':stage,'role':role},kind,open_forbidden),PermissionError)
    evaluator=AccessGate('current','evaluation')
    rejected(lambda:evaluator.open({'stage':'old','role':'val'},'label',open_forbidden),PermissionError)
    evaluator.open({'stage':'old','role':'val'},'label',lambda:True,prediction_sealed=True)
    assert calls==[]
    rejected(lambda:confined(out,'../private'),PermissionError)
    import h5py
    from .data_adapter import CurrentTrainingData
    root=out/'fixture';(root/'h5/v1').mkdir(parents=True);(root/'manifests/training').mkdir(parents=True)
    with h5py.File(root/'h5/v1/x.h5','w') as f:f['image']=np.zeros((2,4,4))
    with h5py.File(root/'h5/v1/u.h5','w') as f:f['image']=np.full((2,4,4),.2)
    with h5py.File(root/'h5/v1/y.h5','w') as f:f['label']=np.ones((2,4,4),dtype=np.int64)
    (root/'manifests/training/lcrseg_v1_seed0.csv').write_text('dataset,site_or_vendor,primary_20pct_split,image_h5_relpath,label_h5_relpath\nprostate,current,train_labeled,x.h5,y.h5\nprostate,current,train_unlabeled,u.h5,\nprostate,old,train_labeled,DO_NOT_OPEN,DO_NOT_OPEN\nprostate,current,val,DO_NOT_OPEN,DO_NOT_OPEN\n')
    data=CurrentTrainingData(root,'prostate','current','current',Config(height=4,width=4,dct_order=2))
    generator=torch.Generator().manual_seed(SEED)
    u=data.batch('U',1,generator);l=data.batch('L',1,generator)
    assert 'observations' not in u and l['observations']['y_obs'].shape==(1,2,4,4)
    assert not any(e['role']=='trainU' and e['kind']=='label' for e in data.gate.events)
    return {'forbidden_open_calls':0,'negative_cases':6,'sealed_evaluation_label_permitted':True,'HDF5_fixture_interface':'PASS','counts':data.counts}


def T17(out):
    ledger=ResourceLedger(out/'budget',max_seconds=1.,max_updates=1,max_bytes=100000)
    rejected(lambda:_fail_attempt(ledger),RuntimeError)
    assert ledger.data['attempts'][0]['status']=='FAILED' and ledger.data['seconds_charged']>0
    with ledger.attempt('done',seconds=.1,updates=1): pass
    rejected(lambda:_reserve(ledger,'excess',1),BudgetExceeded)
    rejected(lambda:_reserve(ledger,'done',0),FileExistsError)
    again=ResourceLedger(out/'budget',max_seconds=1.,max_updates=1,max_bytes=100000)
    assert again.data['updates_reserved']==1
    def timed():
        with ledger.attempt('timeout',seconds=.01):time.sleep(.05)
    rejected(timed,BudgetExceeded)
    small=ResourceLedger(out/'size',max_bytes=1024)
    (small.root/'overflow').write_bytes(b'0'*1024)
    rejected(small.check,BudgetExceeded)
    from .diagnostics import validate_public_summary
    for v in ({'patient_id':'fixture'},{'image':[[1,2]]},{'path':'/home/private/data'},torch.zeros(2)):
        rejected(lambda:validate_public_summary(v),PermissionError)
    validate_public_summary({'mean_Dice':.3,'test_access':0})
    return {'failed_attempt_charged':True,'update_hard_stop':True,'wall_timeout_hard_stop':True,'byte_hard_stop':True,
            'create_only':True,'resume_preserves_cost':True,'public_privacy_negative_controls':4}


def _fail_attempt(ledger):
    with ledger.attempt('failure',seconds=.1):
        time.sleep(.002);raise RuntimeError('intentional failure')


def _reserve(ledger,name,updates):
    with ledger.attempt(name,seconds=.1,updates=updates):pass


def T18(out):
    package=Path(__file__).parent
    snap=json.loads((package/'BASELINE_SNAPSHOT.json').read_text())
    root=package.parents[2]
    actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in snap['files']}
    assert actual==snap['files']
    return {'files_unchanged':len(actual),'hashes':actual,'baseline_HEAD':snap['HEAD']}


TESTS=[globals()[f'T{i:02d}'] for i in range(1,19)]


def run_tests(ledger, prefix, selected=None):
    results=[]
    for test in (t for t in TESTS if selected is None or t.__name__ in selected):
        start=time.monotonic()
        try:
            with ledger.attempt(f'{prefix}_{test.__name__}',seconds=300.,updates=3 if test.__name__=='T10' else 0) as out:
                detail=test(out)
            status='PASS';error=None
        except Exception as exc:
            status='FAIL';detail={};error=f'{type(exc).__name__}: {exc}'
        record=dict(id=test.__name__,status=status,seed=SEED,seconds=time.monotonic()-start,details=detail,error=error)
        results.append(record)
        print(json.dumps(record),flush=True)
        suffix='TEST_RESULTS' if selected is None else 'PARTIAL_TEST_RESULTS'
        (ledger.root/f'{prefix}_{suffix}.json').write_text(json.dumps(results,indent=2))
    return results
