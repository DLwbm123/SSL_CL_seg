"""Class decision-risk budgets on actual SGD displacement; frozen two-arm protocol."""
import copy,itertools,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import source_decision as base,neighbor_margin as parent
from . import core,target,runtime,spectrum_alignment as shared
from .source_decision import HEAD,Calls,Current,build,train,evaluate,geometry_check
from .reliable_margin import choose,take,hinge,margins,aggregate
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write,append
from ..five_frameworks_v1.recipes import generator
_source_setup=base.setup
_source_update_ema=base.update_ema
SEED=20271401
METHODS=('D','DP')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(400,256)),('detach',(640,320)),('engineering',(512,160)),('formal',(40000,20100)),('evaluation',(5784,0))] for b,n in zip(('native','reverse'),ns)}

def flatten(values,pars):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for g,p in zip(values,pars)]).double()

def dual_solve(gram,v):
    # Four constraints: enumerate all active sets, then verify the dual KKT system.
    h=gram.cpu().double()+torch.eye(len(v),dtype=torch.float64);v=v.cpu().double()
    for bits in itertools.product((False,True),repeat=len(v)):
        active=torch.tensor(bits,dtype=torch.bool);a=torch.zeros_like(v)
        if active.any():a[active]=torch.linalg.solve(h[active][:,active],v[active])
        residual=h@a-v;tol=1e-8*max(1.,float(v.norm()))
        if float(a.min())>=-tol and (not (~active).any() or float(residual[~active].min())>=-tol):
            if not active.any() or float(residual[active].abs().max())<=tol:
                assert float((a*residual).abs().max())<=tol*max(1.,float(a.norm()))
                return a.clamp_min(0),float(residual[active].abs().max()) if active.any() else 0.
    raise ArithmeticError('four-class QP failed KKT')

def correction(dopt,grads,risks,budgets,present):
    norms=grads.norm(dim=1);valid=present&(norms>1e-12)
    a=grads/(norms[:,None]+1e-12);a[~valid]=0
    b=(budgets-risks)/(norms+1e-12);b[~valid]=0
    v=a@dopt-b;dual,kkt=dual_solve(a@a.T,v)
    delta=-(a.T@dual.to(a.device));raw=float(delta.norm());dn=float(dopt.norm())
    alpha=min(1.,.25*dn/(raw+1e-12));delta=delta*alpha
    residual=a@(dopt+delta)-b
    return delta,dict(candidate_norm=dn,raw_correction_norm=raw,correction_norm=float(delta.norm()),correction_ratio=float(delta.norm())/(dn+1e-30),alpha=alpha,solver_kkt=kkt,predicted_violation_before=[float(x) for x in v],predicted_violation_after=[float(x) for x in residual],slack=[max(0.,float(x)) for x in residual])

def check_fixed(m,ema,opt):
    head=m.get_parameter(HEAD)
    assert not head.requires_grad and head.grad is None
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}
    assert all(id(head)!=id(p) for g in opt.param_groups for p in g['params'])
    assert torch.equal(head,m.reference.get_parameter(HEAD)) and torch.equal(ema.get_parameter(HEAD),head)
    assert all(not p.requires_grad and p.grad is None for p in m.reference.parameters())


def setup(cfg):
    # The legacy variant A exception must never determine this experiment's freeze policy.
    source,m,ema,opt=_source_setup(dict(cfg,variant='B'));m.cfg=cfg;m.step_context=None
    check_fixed(m,ema,opt)
    before=[]
    def pre(*_):
        nonlocal before
        check_fixed(m,ema,opt)
        before=[p.detach().clone() for g in opt.param_groups for p in g['params']] if m.step_context is not None else []
    @torch.no_grad()
    def post(*_):
        ctx=m.step_context;m.step_context=None
        if ctx is not None:
            pars=[p for g in opt.param_groups for p in g['params']]
            dopt=flatten([p-b for p,b in zip(pars,before)],pars)
            delta,stats=correction(dopt,ctx['grads'],ctx['risks'],ctx['budgets'],ctx['present'])
            apply=cfg['variant']=='DP' and not cfg.get('zero_new',False)
            if apply:
                offset=0
                for p in pars:
                    n=p.numel();p.add_(delta[offset:offset+n].reshape_as(p).to(p.dtype));offset+=n
                actual=flatten([p-b for p,b in zip(pars,before)],pars)
                stats['actual_correction_ratio']=float((actual-dopt).norm())/(float(dopt.norm())+1e-30)
                assert stats['actual_correction_ratio']<=.25001 and torch.isfinite(actual).all()
            stats.update(step=ctx['step'],applied=apply,risks=ctx['risks'].tolist(),budgets=ctx['budgets'].tolist(),risk_gradient_energy=ctx['grads'].square().sum(1).tolist(),candidate_momentum_preserved=True)
            if cfg['bucket']=='formal':append(Path(cfg['output'])/'step_control.jsonl',stats)
        check_fixed(m,ema,opt)
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
    return source,m,ema,opt

def update_ema(ema,m):
    _source_update_ema(ema,m)
    assert torch.equal(m.get_parameter(HEAD),m.reference.get_parameter(HEAD)) and torch.equal(ema.get_parameter(HEAD),m.get_parameter(HEAD))

def terms(m,data,step):
    l,aux=parent.original_terms(m,data,step);d=m.detail;x=d['x'];y=d['y']
    g=generator(data.cfg['target_seed'],1,1,step,'neighbor_random')
    noise=(torch.rand(x.shape,generator=g)>=.5).to(x.device,dtype=x.dtype)*2-1
    view=(x+.05*noise).clamp(-1,1).detach()
    with torch.no_grad():zt=m.reference(view);mask=d['support']&(zt.argmax(1)==y)
    z=m(view);ix=choose(mask,data.cfg['target_seed'],step);yy=take(y[...,None],ix).squeeze(-1);r=take(d['reliability'][...,None],ix).squeeze(-1)
    zz=take(z.permute(0,2,3,1),ix);z0=take(zt.permute(0,2,3,1),ix)
    ce=F.cross_entropy(zz,yy.long(),reduction='none')*r;ref=F.cross_entropy(z0,yy.long(),reduction='none')*r
    present=torch.tensor([bool((yy==c).any()) for c in range(4)],device=x.device)
    risks=torch.stack([ce[yy==c].mean() if present[c] else z.sum()*0 for c in range(4)])
    refs=torch.stack([ref[yy==c].mean() if present[c] else z0.sum()*0 for c in range(4)]).detach()
    budgets=refs+.05*torch.maximum(refs,refs.new_tensor(.1))
    scales=torch.as_tensor(m.cfg['scales'],device=x.device,dtype=x.dtype)
    wloss=aggregate(hinge(take(margins(z,y),ix),take(margins(zt,y),ix),yy,scales)*r,yy,z.sum())
    d.update(neighbor_mask=mask,neighbor_labels=yy,neighbor_values=ce,present=present,risks=risks,budgets=budgets,refs=refs,W=wloss)
    risk=risks[present].mean() if present.any() else z.sum()*0
    return l,aux['R'],risk

def losses(arm,m,data,step):
    l,r,d=terms(m,data,step);detail=m.detail
    pars=[p for g in m._budget_opt.param_groups for p in g['params']]
    gg=torch.stack([flatten(torch.autograd.grad(v,pars,retain_graph=True,allow_unused=True),pars) for v in detail['risks']])
    m.step_context=dict(step=step,grads=gg,risks=detail['risks'].detach().double(),budgets=detail['budgets'].double(),present=detail['present'])
    u=m.cfg['weights']['R']*r
    if m.cfg['variant']=='D' and not m.cfg.get('zero_new',False):u=u+m.cfg['weights']['D']*d
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        grad={k:flatten(torch.autograd.grad(v,pars,retain_graph=True,allow_unused=True),pars) for k,v in [('CE',l),('R',r),('D',d)]}
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared={k:float(v.square().sum()) for k,v in grad.items()}))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),R=float(r.detach()),D=float(d.detach()),accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

# Keep the optimizer outside module/checkpoint state, just like the frozen reference.
_fixed_setup=setup
def setup(cfg):
    source,m,ema,opt=_fixed_setup(cfg);object.__setattr__(m,'_budget_opt',opt);return source,m,ema,opt

def preflight(cfg,out,source,m,ema,opt,data):
    base.preflight(cfg,out,source,m,ema,opt,data);m.step_context=None
    check_fixed(m,ema,opt)
    write(out/'FIXED_HEAD_PREFLIGHT.json',dict(status='PASS',requires_grad_false=True,optimizer_excludes_head=True,student_EMA_source_exact=True,all_trainable_optimizer_ids_equal=True))

def qualify(cfg):
    # Reuse the existing golden/resume/two-case-overfit and actual last-step engine checks.
    with patch.object(base,'setup',setup),patch.object(base,'update_ema',update_ema):base.qualify(cfg)
    out=Path(cfg['output']);source,m,ema,opt=setup(cfg)
    v=torch.load(out/'engineering_terminal.pt',map_location='cpu',weights_only=False)
    assert torch.equal(v['model'][HEAD],source.state_dict()[HEAD]) and torch.equal(v['ema'][HEAD],source.state_dict()[HEAD])
    write(out/'FIXED_HEAD_TERMINAL.json',dict(status='PASS',student_EMA_unchanged=True))

def detach(cfg):
    models=[]
    for arm in METHODS:
        c=dict(cfg,group='DETACH_'+arm,variant=arm,zero_new=True);s,m,e,o=setup(c);models.append((m,e,o,Current(c)))
    for step in range(32):
        states=[]
        for m,e,o,data in models:
            o.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,step);(l+u).backward();o.step();update_ema(e,m)
            states.append(copy.deepcopy(target.engine.snapshot(m,e,o,data,step+1)))
        assert runtime.exact(*states)
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,model_opt_EMA_RNG_exact=True))

def compatibility(cfg):
    source,m,ema,opt=setup(cfg);data=Current(cfg);state=copy.deepcopy(m.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    l,a=parent.terms(m,data,0);old=(float(l.detach()),float(a['R'].detach()),float(a['W'].detach()));old_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all());m.detail=None
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);data=Current(cfg)
    l,r,_=terms(m,data,0);new=(float(l.detach()),float(r.detach()),float(m.detail['W'].detach()))
    assert old==new and runtime.exact(old_rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all())) and runtime.exact(state,m.state_dict())
    m.detail=None;write(Path(cfg['root'])/'REUSE_COMPATIBILITY.json',dict(status='PASS',CE_R_W_exact=True,retained_branch_global_RNG_exact=True,patient_metrics_not_consulted=True,old_controls_reused=True))

def sgd_candidate(opt,pars,grad):
    norm=float(grad.float().norm());grad=grad*min(1.,1./(norm+1e-6));offset=0;result=[]
    for group in opt.param_groups:
        for p in group['params']:
            n=p.numel();g=grad[offset:offset+n].reshape_as(p).to(p.dtype)+group['weight_decay']*p.detach();offset+=n
            old=opt.state[p].get('momentum_buffer');buf=g if old is None else group['momentum']*old+g
            direction=g+group['momentum']*buf if group['nesterov'] else buf
            result.append(-group['lr']*direction)
    return flatten(result,pars)

def measure(cfg):
    root=Path(cfg['root']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271592))
    saved=torch.load(Path(cfg['neighbor_root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256
    m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);check_fixed(m,ema,opt)
    state=copy.deepcopy((m.state_dict(),source.state_dict(),opt.state_dict(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    pars=[p for g in opt.param_groups for p in g['params']];sq={k:0. for k in ('CE','R','D')};records=[];coverage=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];share=[0.]*4;groups={'encoder':0.,'decoder':0.}
    for step in range(16):
        l,r,risk=terms(m,data,step);detail=m.detail
        grad={k:flatten(torch.autograd.grad(v,pars,retain_graph=True,allow_unused=True),pars) for k,v in [('CE',l),('R',r),('D',risk)]}
        gs=torch.stack([flatten(torch.autograd.grad(v,pars,retain_graph=True,allow_unused=True),pars) for v in detail['risks']])
        for k,v in grad.items():assert torch.isfinite(v).all();sq[k]+=float(v.square().sum())
        names={id(p):n for n,p in m.named_parameters()};offset=0
        for p in pars:
            n=p.numel();groups['decoder' if 'native.decoder.' in names[id(p)] else 'encoder']+=float(grad['D'][offset:offset+n].square().sum());offset+=n
        for c in range(4):
            mask=detail['neighbor_mask']&(detail['y']==c);coverage[c]+=int(mask.sum());denom[c]+=int((detail['y']==c).sum());share[c]+=float(detail['risks'][c])
            for slot,pid in enumerate(detail['ids']):
                if mask[slot].any():patients[c].add(pid)
        records.append((grad,gs,detail['risks'].detach().double(),detail['budgets'].double(),detail['present']))
        m.detail=None
    weight=.25*math.sqrt(sq['CE']/sq['D']) if sq['D']>0 else 0.;distances=[];ratios=[];nonzero=0
    for g,gs,risks,budgets,present in records:
        d0=sgd_candidate(opt,pars,g['CE']+cfg['weights']['R']*g['R']);dd=sgd_candidate(opt,pars,g['CE']+cfg['weights']['R']*g['R']+weight*g['D']);delta,stats=correction(d0,gs,risks,budgets,present);dp=d0+delta
        distances.append(float((dd/(dd.norm()+1e-30)-dp/(dp.norm()+1e-30)).norm()));ratios.append(stats['correction_ratio']);nonzero+=stats['correction_norm']>0
    after=(m.state_dict(),source.state_dict(),opt.state_dict(),torch.get_rng_state(),torch.cuda.get_rng_state_all());assert runtime.exact(state,after)
    cover=[a/max(1,b) for a,b in zip(coverage,denom)];fraction=sum(share[1:])/max(1e-30,sum(share))
    gates=dict(gradient_finite_nonzero=all(math.isfinite(v) and v>=1e-12 for v in sq.values()),lambda_finite=math.isfinite(weight) and 1e-6<=weight<=1e4,foreground_support=all(cover[c]>=.05 and len(patients[c])>=3 for c in (1,2,3)),foreground_loss_share=fraction>=.05,full_path=all(v>0 for v in groups.values()),direction_separation=sum(distances)/16>=.05,correction_active=nonzero>0,correction_cap=all(math.isfinite(v) and v<=.250001 for v in ratios),states_RNG_exact=True)
    write(root/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,lambda_D=weight,weights=dict(cfg['weights'],D=weight),gradient_squared=sq,gradient_groups=groups,coverage=cover,covered_patient_counts=[len(x) for x in patients],foreground_loss_share=fraction,D_DP_normalized_distance_mean=sum(distances)/16,predicted_correction_ratios=ratios,reused_probe=True,probe_step=256,measurement_optimizer_steps=0))


def self_check():
    for dev in ('cpu','cuda'):
        z=torch.zeros(3,device=dev,dtype=torch.float64);a=torch.eye(3,device=dev,dtype=torch.float64)[:2];a=torch.cat((a,torch.zeros((2,3),device=dev)),dim=0)
        for b in (torch.ones(4,device=dev,dtype=torch.float64),-torch.ones(4,device=dev,dtype=torch.float64)):
            d,stats=correction(z+1,a,z.new_zeros(4),b,torch.ones(4,device=dev,dtype=torch.bool));assert stats['correction_ratio']<=.250001
        d,stats=correction(z+1,a,z.new_zeros(4),z.new_ones(4)*10,torch.ones(4,device=dev,dtype=torch.bool));assert torch.equal(d,z)
        d,_=correction(z,a,z.new_zeros(4),-z.new_ones(4),torch.ones(4,device=dev,dtype=torch.bool));assert torch.equal(d,z)
        p=torch.nn.Parameter(torch.tensor([1.,-2.],device=dev));o=torch.optim.SGD([p],lr=.001,momentum=.9,nesterov=True,weight_decay=.0005)
        for i in range(2):
            p.grad=torch.tensor([.3,-.4],device=dev);before=p.detach().clone();expected=sgd_candidate(o,[p],p.grad.double());torch.nn.utils.clip_grad_norm_([p],1.);o.step();assert torch.allclose((p-before).double(),expected,atol=1e-7,rtol=1e-4)
        saved=copy.deepcopy(o.state_dict());old=p.detach().clone();p.grad=torch.ones_like(p);o.step();want=p.detach().clone();state=copy.deepcopy(o.state_dict());p.data.copy_(old);o.load_state_dict(saved);p.grad=torch.ones_like(p);o.step();assert torch.equal(want,p) and runtime.exact(state,o.state_dict())
    return dict(status='PASS',QP_KKT=True,inactive_identity=True,zero_candidate_identity=True,SGD_Nesterov_WD_displacement=True,candidate_momentum_restore_exact=True,synthetic_reverse=0)

def controller(cfg,guard):
    root=Path(cfg['root']);assert (Path(cfg['neighbor_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));geometry_check(cfg);write(root/'DECISION_SELF_CHECK.json',self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    parent_p0=json.loads((Path(cfg['neighbor_root'])/'P0_RESULT.json').read_text())
    common=dict(cfg,arm='A',weights={'R':parent_p0['weights']['R'],'D':0.},scales=parent_p0['scales'],target_seed=SEED,variant='D',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    c=dict(common,group='MEASURE',bucket='p0',phase='measure',output=str(root/'measure'))
    stage('compatibility',c);stage('measure',c);p0=json.loads((root/'P0_RESULT.json').read_text())
    if p0['status']!='PASS':write(root/'STATUS.json',dict(status='P0_STOP',formal_started=False,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
    common['weights']=p0['weights'];stage('detach',dict(common,group='DETACH',bucket='detach',phase='detach',output=str(root/'detach')))
    configs=[dict(common,group=f'{SEED}_{arm}',variant=arm) for arm in METHODS]
    for c in configs:stage('qualify',dict(c,bucket='engineering',phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,bucket='formal',phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',dict(c,bucket='evaluation'))
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))

def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if mode=='controller':controller(cfg,guard);return
        calls=Calls(cfg,mode);backward=torch.autograd.backward;grad=torch.autograd.grad
        def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
        def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
        with patch.object(base,'losses',losses),patch.object(shared,'CAPS',CAPS),patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(core,'update_ema',update_ema),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
            {'qualify':qualify,'train':train,'compatibility':compatibility,'measure':measure,'detach':detach,'evaluate':evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
