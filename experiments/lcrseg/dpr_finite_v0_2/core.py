"""One native Adam proposal, detached realized-response probes, one finite candidate."""
from experiments.lcrseg.dpr_v0_1.core import *

ARMS=('DPR_FINITE_U','DPR_FINITE_SIGN_U')
MAIN=ARMS[0]
FAIL_HOOK=None
PHASE='pre'

def finite(*values):
    if not all(bool(torch.isfinite(v).all()) for v in values):raise FloatingPointError('nonfinite '+PHASE)

@torch.no_grad()
def correct(g,J,b,lam):
    g,J,b=g.double(),J.double(),b.double();finite(g,J,b)
    if g.ndim!=1 or J.shape!=(2,g.numel()) or b.shape!=(2,) or not 0<=lam<=1:raise ValueError('finite correction contract')
    if lam==0 or g.norm()<=1e-30 or J.norm()<=1e-30:return torch.zeros_like(g),1.
    e=g/g.norm();A=J-(J@e)[:,None]*e
    S=torch.eye(2,device=g.device,dtype=torch.float64)+lam*(A@A.T)
    eta=-lam*A.T@torch.linalg.solve(S,b);finite(eta)
    return eta,float(torch.linalg.cond(S))

@torch.no_grad()
def probes(delta,valid,key,arm):
    finite(delta);delta=delta.double().detach()*valid;den=delta.norm()
    v=delta/den if den>1e-30 else torch.zeros_like(delta)
    if arm==ARMS[1]:
        for j in range(2):
            gen=torch.Generator(device=v.device).manual_seed(stable_seed('DPR_FINITE_V0_2',*key,j,'sign_control'))
            v[:,j]*=torch.randint(0,2,v[:,j].shape,generator=gen,device=v.device)*2-1
    elif arm!=MAIN:raise PermissionError('unregistered arm')
    return v.detach(),float(den)

def query(model,batch,key,counts,stream,full=False):
    response_input(batch);logits,_=fwd(model,batch['image'],False,key,counts,stream);finite(logits)
    chi,valid=contrast(logits,batch['geometry']);finite(chi)
    native=None
    if full:
        C=logits.new_tensor([[-1/math.sqrt(2),1/math.sqrt(2),0],[-1/math.sqrt(6),-1/math.sqrt(6),2/math.sqrt(6)]])
        native=(torch.einsum('kc,bchw->bkhw',C,logits.detach()).double(),logits.detach().argmax(1))
    return chi,valid,native

def rows_at_raw(chi,v,ps,counts):
    if v.requires_grad:raise PermissionError('probe not detached')
    before=vector(ps,True).clone();rows=[]
    for j in range(2):
        phi=(v[:,j]*chi[:,j]).sum()
        grads=torch.autograd.grad(phi,ps,retain_graph=j==0,create_graph=False)
        rows.append(torch.cat([z.detach().reshape(-1) for z in grads]));counts['response_VJP']+=1
    if not torch.equal(before,vector(ps,True)):raise RuntimeError('response polluted supervised .grad')
    R=torch.stack(rows).double();finite(R);return R

@torch.no_grad()
def adopt(ps,raw,candidate,Eraw,Ecandidate):
    finite(raw,candidate,Eraw,Ecandidate)
    accepted=bool(Ecandidate<=Eraw)
    if not accepted:assign(ps,raw)
    return accepted

def ratio(a,b):return float(a/b) if b>1e-30 else None

@torch.no_grad()
def numerical_row(theta,raw,candidate,g,J,b,eta,lam,condition,Rnorm,accepted):
    theta,raw,candidate=theta.double(),raw.double(),candidate.double();d0=raw-theta
    applied=candidate if accepted else raw;target=raw+eta;eps=roundoff_budget(target)
    algebra=1e-12*max(float(g.norm()*eta.norm()),1e-30)
    allowance=1e-12*max(float(lam*b.square().sum()),float(eta.square().sum()),1e-30)
    linear=b+J@eta;objective=.5*eta.square().sum()+.5*lam*linear.square().sum();objective0=.5*lam*b.square().sum()
    progress=float(abs(g@eta));budget=float(g.abs()@eps)+algebra
    candidate_res=float(abs(g@(candidate-raw)));applied_res=float(abs(g@(applied-raw)))
    checks=dict(finite=all(bool(torch.isfinite(x).all()) for x in (candidate,eta,applied)),
        objective_nonincrease=float(objective-objective0)<=allowance,algebra_progress=progress<=algebra,
        candidate_progress=candidate_res<=budget,applied_progress=applied_res<=budget,
        candidate_cast_within_budget=bool(((candidate-target).abs()<=eps).all()),
        sampled_response_nonincrease=float(linear.norm())<=float(b.norm())+1e-12*max(float(b.norm()),1e-30),
        correction_bound=float(eta.norm())<=math.sqrt(lam)*float(b.norm())+1e-12*max(float(b.norm()),1e-30))
    return dict(checks=checks,lambda_response=lam,g_norm=float(g.norm()),d0_norm=float(d0.norm()),eta_norm=float(eta.norm()),
        correction_ratio=ratio(eta.norm(),d0.norm()),g_dot_d0=float(g@d0),g_dot_candidate=float(g@(candidate-theta)),g_dot_applied=float(g@(applied-theta)),
        b_norm=float(b.norm()),sampled_linear_candidate_norm=float(linear.norm()),sampled_linear_applied_norm=float((b+J@(applied-raw)).norm()),
        sampled_linear_candidate_ratio=ratio(linear.norm(),b.norm()),condition_number=condition,raw_response_norm=Rnorm,
        algebra_progress_residual=progress,algebra_progress_budget=algebra,candidate_progress_residual=candidate_res,
        applied_progress_residual=applied_res,applied_progress_budget=budget,objective_difference=float(objective-objective0),raw_non_descent=int(g@d0>=0))

@torch.no_grad()
def energies(before,raw,candidate,valid,v):
    d0=(raw.double()-before.double())*valid;dc=(candidate.double()-before.double())*valid
    denominator=valid.expand_as(before).sum()
    if denominator==0:raise RuntimeError('empty response geometry')
    er=d0.square().sum()/denominator;ec=dc.square().sum()/denominator
    qr=(v*d0).sum((0,2,3));qc=(v*dc).sum((0,2,3));finite(er,ec,qr,qc)
    return er,ec,dict(pooled_raw_RMS=float(er.sqrt()),pooled_candidate_RMS=float(ec.sqrt()),
        pooled_candidate_ratio=ratio(ec.sqrt(),er.sqrt()),same_probe_raw_norm=float(qr.norm()),same_probe_candidate_norm=float(qc.norm()),
        same_probe_candidate_ratio=ratio(qc.norm(),qr.norm()))

def step(model,ema,opt,l,probe,task,epoch,index,counts,patients,diagnostic=False):
    global PHASE
    PHASE='pre'
    if task['arm'] not in ARMS:raise PermissionError('unregistered arm')
    lam=strength(epoch);detail={};transient={};original=opt.step;key=(task['seed'],task['domain'],epoch,index)
    # Genuine warmup: no trial snapshot, response query, VJP, hook or second parameter write.
    if not lam:
        if probe is not None:raise PermissionError('warmup response access')
        row=parent.step(model,ema,opt,l,None,'T_LCTX',task['seed'],task['domain'],epoch,index,counts,patients)
        PHASE='final';row.update(checks={'native_warmup':True},lambda_response=0.,degeneracy='warmup',candidate_accepted=0,guard_rejected=0,response_images=0,response_VJP=0,response_forwards=0,pseudo_labels=0,EMA_updates=1)
        return row,{}
    ps=parameters(model);vjp_start=counts['response_VJP'];nf=0
    def proposal(*args,**kwargs):
        nonlocal nf
        global PHASE
        theta=vector(ps).clone();g=vector(ps,True).double();finite(theta,g)
        with torch.no_grad():before,valid,nb=query(model,probe,key,counts,'response_before',diagnostic);nf+=1
        original(*args,**kwargs);PHASE='raw';raw=vector(ps).clone();finite(raw)
        chi,valid_raw,nr=query(model,probe,key,counts,'response_raw',diagnostic);nf+=1
        if not torch.equal(valid,valid_raw):raise RuntimeError('geometry changed')
        delta=(chi.detach().double()-before.double())*valid;v,den=probes(delta,valid,key,task['arm'])
        R=rows_at_raw(chi,v,ps,counts) if g.norm()>1e-30 and den>1e-30 else torch.zeros(2,g.numel(),device=g.device,dtype=torch.float64)
        q=(v*delta).sum((0,2,3)).detach();norm=R.norm()
        J=R/norm if norm>1e-30 else torch.zeros_like(R);b=q/norm if norm>1e-30 else torch.zeros_like(q)
        reason='zero_g' if g.norm()<=1e-30 else 'zero_delta' if den<=1e-30 else 'zero_R' if norm<=1e-30 else None
        eta,condition=correct(g,J,b,lam)
        candidate=(raw.double()+eta).float()
        if reason:
            candidate=raw;cc=chi.detach();nc=nr
        else:
            assign(ps,candidate);PHASE='candidate'
            with torch.no_grad():cc,vc,nc=query(model,probe,key,counts,'response_candidate',diagnostic);nf+=1
            if not torch.equal(valid,vc):raise RuntimeError('candidate geometry changed')
        er,ec,metrics=energies(before,chi.detach(),cc,valid,v)
        accepted=adopt(ps,raw,candidate,er,ec) if reason is None else False
        PHASE='final'
        detail.update(numerical_row(theta,raw,candidate,g,J,b,eta,lam,condition,float(norm),accepted))
        detail.update(metrics,candidate_accepted=int(accepted),guard_rejected=int(reason is None and not accepted),degeneracy=reason,
            delta_norm=den,finite_raw_energy=float(er),finite_candidate_energy=float(ec),finite_applied_energy=float(ec if accepted else er),
            pooled_applied_RMS=metrics['pooled_candidate_RMS'] if accepted else metrics['pooled_raw_RMS'],
            same_probe_applied_norm=metrics['same_probe_candidate_norm'] if accepted else metrics['same_probe_raw_norm'])
        detail['checks']['finite_guard']=detail['finite_applied_energy']<=detail['finite_raw_energy']
        detail['checks']['raw_restored_exactly']=accepted or torch.equal(vector(ps),raw)
        if diagnostic:
            geom=probe['geometry'];denfull=2*geom.sum();full={}
            for name,native in (('raw',nr),('candidate',nc),('applied',nc if accepted else nr)):
                full['native_'+name+'_RMS']=float((((native[0]-nb[0]).square()*geom[:,None]).sum()/denfull).sqrt())
                full['native_'+name+'_prediction_change']=float(((native[1]!=nb[1])&geom).sum()/geom.sum())
            detail.update(full)
            transient.update(theta=theta.cpu(),raw=raw.cpu(),candidate=candidate.cpu())
    try:
        with patch.object(opt,'step',proposal):row=parent.step(model,ema,opt,l,None,'T_LCTX',task['seed'],task['domain'],epoch,index,counts,patients)
    except BaseException:
        if FAIL_HOOK:FAIL_HOOK(model,ema,opt,counts,PHASE)
        raise
    row.update(detail,pseudo_labels=0,response_images=len(probe['image']),response_VJP=counts['response_VJP']-vjp_start,response_forwards=nf,EMA_updates=1)
    return row,transient

@torch.no_grad()
def actual_diagnostic(model,l,probe,task,epoch,index,counts,transient,row):
    ps=parameters(model);applied=vector(ps).clone();loss={}
    try:
        for name,value in [('before',transient['theta']),('raw',transient['raw']),('candidate',transient['candidate']),('applied',applied)]:
            assign(ps,value.to(applied));loss[name]=supervised_value(model,l,task,epoch,index,counts)
        result={k:v for k,v in row.items() if k.startswith(('sampled_','same_probe_','pooled_','native_','finite_','g_dot_'))}
        result.update({name+'_loss':v for name,v in loss.items()});result.update(candidate_accepted=row['candidate_accepted'],guard_rejected=row['guard_rejected'],degeneracy=row['degeneracy'],
            scope='A sampled linear / B same detached probes finite / C full pooled guard / D native-resolution diagnostic / E same LCTX; guard is not independent retention evidence')
        return result
    finally:assign(ps,applied)
