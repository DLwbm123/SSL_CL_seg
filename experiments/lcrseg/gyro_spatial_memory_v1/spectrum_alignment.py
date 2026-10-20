"""Two target streams; shared fixed calibration and native-engine spectral/alignment controls."""
import copy,fcntl,json,math,os,subprocess,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from . import core,target,runtime,n_contribution as prior
from .bounded_runtime import install
from .module import rows,centroid
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

SEEDS=(20261021,20261022)
METHODS=('S','G00','Aalign')
CAPS=dict(calibration_encoder=768,calibration_reverse=32,detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=576,engineering_encoder=768,engineering_reverse=240,formal_native=24300,formal_encoder=384600,formal_reverse=12024,diagnostic_reverse=60,evaluation_native=17352)


class Calls:
    def __init__(self,cfg,mode):self.cfg=cfg;self.mode=mode;self.diagnostic=False
    def kind(self,operation):
        family={'calibrate':'calibration','detach':'detach','qualify':'engineering','train':'formal','evaluate':'evaluation'}[self.mode]
        return ('diagnostic' if self.diagnostic and operation=='reverse' else family)+'_'+operation
    def call(self,kind,n,fn):
        root=Path(self.cfg['root']);runtime.guard(self.cfg)
        with (root/'calls.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);p=root/'calls_counter.json'
            c=json.loads(p.read_text()) if p.exists() else dict(executed={k:0 for k in CAPS},by_group={},pending=None,caps=CAPS)
            assert c['caps']==CAPS and c['pending'] is None
            if c['executed'][kind]+n>CAPS[kind]:raise PermissionError('frozen forward/reverse cap '+kind)
            c['pending']=dict(kind=kind,samples=n,group=self.cfg.get('group',self.mode));write(p,c)
            append(root/'calls_events.jsonl',dict(action='reserved',**c['pending'],time=time.time()))
        value=fn()
        with (root/'calls.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads(p.read_text());assert c['pending']['kind']==kind and c['pending']['samples']==n
            g=self.cfg.get('group',self.mode);key=g+':'+kind;c['executed'][kind]+=n;c['by_group'][key]=c['by_group'].get(key,0)+n;c['pending']=None;write(p,c)
            append(root/'calls_events.jsonl',dict(action='executed',kind=kind,samples=n,group=g,time=time.time()))
        return value


def spectral(c):return 1-torch.linalg.svdvals((2/3)*c).sum()/math.sqrt(len(c)*min(c.shape))
def alignment(c):return (1-c.square().sum(1)).mean()
def center(model,data,step,x=None):
    if x is None:x,_,_=data.batch('L',step)
    fill,_,_=data.batch('F',step,14);xx=torch.cat([x,fill]);seed=data.cfg['target_seed']
    pos=prior.positions(seed,step,16,x.device);fields=model.encode(torch.cat([xx,prior.strong(xx,seed,step,0),prior.strong(xx,seed,step,1)]))
    return centroid(*[rows(v,pos) for v in fields.chunk(3)])


class Current(prior.Current):
    def __init__(self,cfg,domain='Philips'):
        super().__init__(dict(cfg,group='S' if cfg['method']=='S' else 'G00'),domain)
    def batch(self,role,step,n=2):
        value=super().batch(role,step,n)
        if role=='L':self.last_supervised=(value[0].detach(),value[1].detach())
        return value


def group_norms(names,values):
    return {k:sum(float(g.detach().double().square().sum()) for n,g in zip(names,values) if g is not None and (n.startswith(('native.enc','native.bottleneck')) if k=='encoder' else n.startswith('native.decoder')))**.5 for k in ('encoder','decoder')}


def diagnostics(model,l,u,step):
    named=[(n,p) for n,p in model.parent.named_parameters() if p.requires_grad];names=[n for n,p in named];pars=[p for n,p in named]
    calls=model.calls;calls.diagnostic=True
    try:
        a=torch.autograd.grad(l,pars,retain_graph=True,allow_unused=True);b=torch.autograd.grad(u,pars,retain_graph=True,allow_unused=True)
    finally:calls.diagnostic=False
    combined=[(torch.zeros_like(p) if aa is None else aa)+(torch.zeros_like(p) if bb is None else bb) for p,aa,bb in zip(pars,a,b)]
    metrics=dict(step=step,CE_gradient=group_norms(names,a),aux_gradient=group_norms(names,b),combined_gradient=group_norms(names,combined))
    metrics['raw_global_gradient_norm']=sum(float(g.double().square().sum()) for g in combined)**.5
    metrics['theoretical_clip_coefficient']=min(1.,1/(metrics['raw_global_gradient_norm']+1e-6))
    metrics['CE_aux_inner_product']={}
    metrics['CE_aux_cosine']={}
    for k in ('encoder','decoder'):
        dot=sum(float((aa.double()*bb.double()).sum()) for n,aa,bb in zip(names,a,b) if aa is not None and bb is not None and (n.startswith(('native.enc','native.bottleneck')) if k=='encoder' else n.startswith('native.decoder')))
        den=metrics['CE_gradient'][k]*metrics['aux_gradient'][k];metrics['CE_aux_inner_product'][k]=dot;metrics['CE_aux_cosine'][k]=dot/den if den else None
    model.pending_diagnostic=metrics


def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);l=pixel_ce(model(x),y).mean();method=model.cfg['method']
    if method=='S':u=l*0;images=0
    else:
        if method=='STOP':
            with torch.no_grad():c=center(model,data,step,x);u=.1*spectral(c)
        else:
            c=center(model,data,step,x);u=.1*spectral(c) if method=='G00' else model.cfg['lambda_A']*alignment(c)
        images=48
    if getattr(model,'collect_stats',False) and step in (0,499,999,1499,1999):diagnostics(model,l,u,step)
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=64 if images else 0,geometry_images=images,historical_raw_reads=0)


def make_setup(cfg,calls):
    build,base_setup=prior.make_setup(core.build)
    def setup(c):
        source,model,ema,opt=base_setup(c);model.collect_stats=False;model.pending_diagnostic=None;model.calls=calls
        def before(*_):
            if model.pending_diagnostic is None:return
            model.diagnostic_before={n:p.detach().cpu().clone() for n,p in model.parent.named_parameters() if p.requires_grad}
            model.pending_diagnostic['observed_clipped_global_gradient_norm']=sum(float(p.grad.detach().double().square().sum()) for p in model.parent.parameters() if p.requires_grad and p.grad is not None)**.5
        def after(*_):
            if model.pending_diagnostic is None:return
            named=[(n,p) for n,p in model.parent.named_parameters() if p.requires_grad];delta=[p.detach().cpu().double()-model.diagnostic_before[n].double() for n,p in named]
            model.pending_diagnostic['actual_SGD_displacement']=group_norms([n for n,p in named],delta)
            append(Path(c['output'])/'gradient_diagnostics.jsonl',model.pending_diagnostic);model.pending_diagnostic=None;model.diagnostic_before=None
        opt.register_step_pre_hook(before);opt.register_step_post_hook(after)
        return source,model,ema,opt
    return build,setup


def preflight(cfg,out,source,model,ema,opt,data):
    prior.preflight(dict(cfg,group='S' if cfg['method']=='S' else 'G00'),out,source,model,ema,opt,data)
    model.collect_stats=cfg['phase']=='formal'


def qualify(cfg):
    target.main(dict(cfg,preflight_only=True));out=Path(cfg['output']);source,model,ema,opt=core.target_setup(cfg);data=Current(cfg)
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);target.engine.restore(initial,model,ema,opt,data);x,y,_=data.labeled(0);calls=0
    def step():
        nonlocal calls
        opt.zero_grad(set_to_none=True);v=pixel_ce(model(x),y).mean();v.backward();opt.step();core.update_ema(ema,model);calls+=1;return float(v)
    first=float(pixel_ce(model(x),y).mean())
    for i in range(30):
        before=copy.deepcopy(target.engine.snapshot(model,ema,opt,data,i)) if i==15 else None;v=step()
        if i==15:
            expected=copy.deepcopy(target.engine.snapshot(model,ema,opt,data,i+1));target.engine.restore(before,model,ema,opt,data)
            assert step()==v and runtime.exact(expected,target.engine.snapshot(model,ema,opt,data,i+1))
    last=float(pixel_ce(model(x),y).mean());threshold=min(1e-4,.01*first)
    write(out/'OVERFIT_MEASUREMENT.json',dict(first=first,final=last,threshold=threshold,floor=first<1e-6))
    assert last<=first+1e-8 if first<1e-6 else first-last>=threshold
    target.engine.restore(initial,model,ema,opt,data);opt.zero_grad(set_to_none=True);l,u,_=losses('A',model,data,0);(l+u).backward();opt.step();core.update_ema(ema,model);calls+=1
    assert calls==32 and all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if not k.startswith('parent.'))
    runtime.save(cfg,target.engine.snapshot(model,ema,opt,data,1),out/'engineering_terminal.pt')
    write(out/'ENGINEERING.json',dict(status='PASS',discarded=True,actual_calls=calls,first=first,final=last,CE_decrease_NA=first<1e-6,nonempty_restore_exact=True,peak_bytes=torch.cuda.max_memory_allocated()))


def calibrate(cfg):
    source,model,ema,opt=core.target_setup(cfg);del source,ema
    assert not opt.state;data=Current(cfg);before=copy.deepcopy(model.state_dict());pars=[p for n,p in model.parent.named_parameters() if p.requires_grad and n.startswith(('native.enc','native.bottleneck'))];items=[];gg=aa=0.
    for step in range(16):
        c=center(model,data,step);g=spectral(c);a=alignment(c)
        dg=torch.autograd.grad(g,pars,retain_graph=True);da=torch.autograd.grad(a,pars)
        assert all(torch.isfinite(v).all() for v in [g,a,*dg,*da])
        ng=sum(float(v.double().square().sum()) for v in dg);na=sum(float(v.double().square().sum()) for v in da);gg+=ng;aa+=na
        items.append(dict(batch=step,geometry_gradient_squared=ng,alignment_gradient_squared=na,geometry=float(g.detach()),alignment=float(a.detach())))
    assert gg>=1e-12 and aa>=1e-12;weight=.1*math.sqrt(gg/aa);assert math.isfinite(weight) and 1e-4<=weight<=10
    assert not opt.state and all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    write(Path(cfg['root'])/'CALIBRATION.json',dict(status='PASS',seed=20261024,batches=16,lambda_A=weight,sum_geometry_squared=gg,sum_alignment_squared=aa,per_batch=items,model_buffers_unchanged=True,optimizer_updates=0,shared_by_both_streams=True))


def detach_check(cfg):
    root=Path(cfg['root']);models=[]
    for group,method in [('DETACH_S','S'),('DETACH_STOP','STOP')]:
        c=dict(cfg,group=group,method=method,phase='engineering',target_seed=20261021);source,m,e,opt=core.target_setup(c);del source;models.append((m,e,opt,Current(c)))
    records=[]
    for step in range(32):
        inputs=[];rngs=[]
        for m,e,opt,data in models:
            opt.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,step);inputs.append(data.last_supervised)
            assert torch.isfinite(l+u);(l+u).backward();opt.step();core.update_ema(e,m);rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models;assert torch.equal(inputs[0][0],inputs[1][0]) and torch.equal(inputs[0][1],inputs[1][1])
        assert runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rngs[0],rngs[1])
        records.append(dict(step=step+1,model_opt_EMA_RNG_exact=True))
    write(root/'DETACH_CHECK.json',dict(status='PASS',updates=64,discarded=True,steps=records,compare_reads=False,reads=[dict(L=v[3].l_reads,extra_image_pairs=v[3].u_reads) for v in models]))


def self_check():
    g=torch.Generator().manual_seed(20261024);z=torch.nn.functional.normalize(torch.randn(3,8,32,generator=g,dtype=torch.float64),dim=-1).requires_grad_();w=(.5,.25,.25);c=sum(v*z[i] for i,v in enumerate(w));a=alignment(c)
    expected=sum(w[i]*w[j]*(z[i]-z[j]).square().sum(1).mean() for i in range(3) for j in range(i+1,3))
    assert torch.allclose(a,expected,atol=1e-12,rtol=1e-12) and torch.isfinite(torch.autograd.grad(a,z)[0]).all()
    assert torch.isfinite(torch.autograd.grad(spectral(sum(v*z[i] for i,v in enumerate(w))),z)[0]).all()
    assert torch.equal(prior.positions(20261021,0,2,'cpu'),prior.positions(20261021,0,16,'cpu')[:2])
    x=torch.linspace(-1,1,8192).reshape(2,1,64,64);assert torch.equal(prior.strong(x,20261021,0,0),prior.strong(torch.cat([x,torch.zeros(14,1,64,64)]),20261021,0,0)[:2])
    return dict(status='PASS',weighted_alignment_identity=True,finite_feature_gradients=True,slot_prefix_exact=True,medical_model_forwards=0,optimizer_updates=0)


def controller(cfg,guard):
    root=Path(cfg['root'])
    def stage(mode,c):
        guard(cfg,new_stage=True);path=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(path,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=16000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=p.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code))
        if code:raise RuntimeError(c['group']+' '+mode+' failed')
    cal=root/'calibration';cal.mkdir(exist_ok=False);stage('calibrate',dict(cfg,group='CAL',method='G00',target_seed=20261024,phase='engineering',output=str(cal)))
    weight=json.loads((root/'CALIBRATION.json').read_text())['lambda_A'];base=dict(cfg,lambda_A=weight)
    detach=root/'detach';detach.mkdir(exist_ok=False);stage('detach',dict(base,group='DETACH',method='S',phase='engineering',target_seed=20261021,output=str(detach)))
    configs=[dict(base,group=str(seed)+'_'+method,method=method,target_seed=seed,arm='A',stage=str(seed)+'_'+method,preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for method in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if mode=='controller':controller(cfg,guard);return
        calls=Calls(cfg,mode);build,setup=make_setup(cfg,calls);forward=core.Model.forward;encode=core.Model.encode;backward=torch.autograd.backward;grad=torch.autograd.grad
        def f(m,x):return calls.call(calls.kind('native'),len(x),lambda:forward(m,x))
        def e(m,x):return calls.call(calls.kind('encoder'),len(x),lambda:encode(m,x))
        def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
        def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
        with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight),patch.object(prior,'losses',losses),patch.object(core.Model,'forward',f),patch.object(core.Model,'encode',e),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
            {'calibrate':calibrate,'detach':detach_check,'qualify':qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
