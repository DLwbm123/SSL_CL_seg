"""S/G/T/GT factor experiment, using the existing frozen native training engine."""
import copy,json,math,os,time,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from . import core,target,runtime,spectrum_alignment as shared
from . import n_contribution as prior
from .module import rows,centroid
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

SEEDS=(20261025,20261026)
METHODS=('S','G','T','GT')
CAPS=dict(calibration_encoder=768,calibration_teacher_encoder=256,calibration_reverse=32,detach_native=160,detach_encoder=1632,detach_teacher_encoder=544,detach_reverse=64,engineering_native=768,engineering_encoder=1536,engineering_teacher_encoder=512,engineering_reverse=320,formal_native=32400,formal_encoder=770000,formal_teacher_encoder=257000,formal_reverse=16032,diagnostic_reverse=80,evaluation_native=23136)


def terms(model,data,step,x=None):
    if x is None:x,_,_=data.batch('L',step)
    fill,_,_=data.batch('F',step,14);weak=torch.cat([x,fill]);seed=data.cfg['target_seed'];pos=prior.positions(seed,step,16,x.device)
    teacher=model.temporal_teacher;assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    with torch.no_grad():zteacher=rows(teacher.encode(weak),pos)
    fields=model.encode(torch.cat([weak,prior.strong(weak,seed,step,0),prior.strong(weak,seed,step,1)]));z=[rows(v,pos) for v in fields.chunk(3)]
    g=.1*shared.spectral(centroid(*z));t=sum((1-(v*zteacher).sum(1)).mean() for v in z[1:])/2
    assert torch.isfinite(g+t);return g,t


def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);l=pixel_ce(model(x),y).mean();method=model.cfg['method']
    if method=='MINIMAL':g=t=l.detach()*0;images=0
    else:g,t=terms(model,data,step,x);images=64
    u=l*0
    if method in ('G','GT'):u=u+g
    if method in ('T','GT'):u=u+model.cfg['lambda_T']*t
    if getattr(model,'collect_stats',False) and step in (0,499,999,1499,1999):shared.diagnostics(model,l,u,step)
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),geometry=float(g.detach()),temporal=float(t.detach()),accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=64 if images else 0,geometry_images=48 if images else 0,teacher_images=16 if images else 0,historical_raw_reads=0)


def make_setup(cfg,calls):
    build,setup_base=shared.make_setup(cfg,calls)
    def setup(c):
        source,m,ema,opt=setup_base(c);object.__setattr__(m,'temporal_teacher',ema);object.__setattr__(ema,'is_temporal_teacher',True)
        assert not any('temporal_teacher' in k for k in m.state_dict()) and all(not p.requires_grad for p in ema.parameters())
        return source,m,ema,opt
    return build,setup


def calibrate(cfg):
    source,m,ema,opt=core.target_setup(cfg);del source;data=shared.Current(cfg);before=copy.deepcopy(m.state_dict());before_ema=copy.deepcopy(ema.state_dict());pars=[p for n,p in m.parent.named_parameters() if p.requires_grad and n.startswith(('native.enc','native.bottleneck'))];gg=tt=0.;batches=[]
    for step in range(16):
        g,t=terms(m,data,step);a=torch.autograd.grad(g,pars,retain_graph=True);b=torch.autograd.grad(t,pars);assert all(torch.isfinite(v).all() for v in [g,t,*a,*b])
        ng=sum(float(v.double().square().sum()) for v in a);nt=sum(float(v.double().square().sum()) for v in b);gg+=ng;tt+=nt;batches.append(dict(batch=step,weighted_geometry_grad_squared=ng,temporal_grad_squared=nt,geometry=float(g.detach()),temporal=float(t.detach())))
    assert gg>=1e-12 and tt>=1e-12;weight=.25*math.sqrt(gg/tt);assert math.isfinite(weight) and 1e-4<=weight<=10
    assert not opt.state and runtime.exact(before,m.state_dict()) and runtime.exact(before_ema,ema.state_dict()) and all(p.grad is None for p in ema.parameters())
    write(Path(cfg['root'])/'CALIBRATION.json',dict(status='PASS',seed=20261027,batches=16,lambda_T=weight,weighted_geometry_squared=gg,temporal_squared=tt,per_batch=batches,model_buffers_unchanged=True,teacher_unchanged=True,teacher_stop_gradient=True,optimizer_updates=0,shared_by_both_streams=True))


def detach_check(cfg):
    models=[]
    for group,method in [('DETACH_S','MINIMAL'),('DETACH_STOP','S')]:
        c=dict(cfg,group=group,method=method,phase='engineering',target_seed=20261025);source,m,e,opt=core.target_setup(c);del source;models.append((m,e,opt,shared.Current(c)))
    records=[]
    for step in range(32):
        xx=[];rng=[]
        for m,e,opt,data in models:
            opt.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,step);xx.append(data.last_supervised);(l+u).backward();opt.step();core.update_ema(e,m);rng.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()));assert all(p.grad is None for p in e.parameters())
        a,b=models
        assert runtime.exact(xx[0],xx[1]) and runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rng[0],rng[1])
        records.append(dict(step=step+1,model_opt_EMA_RNG_exact=True,teacher_stop_gradient=True))
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,discarded=True,steps=records))


def self_check():
    x=torch.nn.functional.normalize(torch.randn(2,64,32,dtype=torch.float64),dim=-1).requires_grad_();teacher=torch.nn.functional.normalize(torch.randn(64,32,dtype=torch.float64),dim=-1).requires_grad_();loss=(1-(x*teacher.detach()).sum(-1)).mean();a,b=torch.autograd.grad(loss,(x,teacher),allow_unused=True);assert torch.isfinite(a).all() and b is None
    return dict(status='PASS',teacher_detached=True,finite_temporal_feature_gradient=True,medical_model_forwards=0,optimizer_updates=0)


def controller(cfg,guard):
    root=Path(cfg['root'])
    preparation=root/'preparation'
    if not preparation.exists():preparation.symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert (preparation/'private_Philips_val.jsonl').is_file()
    def stage(mode,c):
        guard(cfg,new_stage=True);path=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(path,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=16000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=p.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    resumed=cfg.get('resume_completed_stages',False)
    if not resumed:
        out=root/'calibration';out.mkdir();stage('calibrate',dict(cfg,group='CAL',method='GT',target_seed=20261027,phase='engineering',output=str(out)))
    else:
        assert json.loads((root/'CALIBRATION.json').read_text())['status']=='PASS'
        assert json.loads((root/'DETACH_CHECK.json').read_text())['status']=='PASS'
    base=dict(cfg,lambda_T=json.loads((root/'CALIBRATION.json').read_text())['lambda_T'])
    if not resumed:
        out=root/'detach';out.mkdir();stage('detach',dict(base,group='DETACH',method='S',target_seed=20261025,phase='engineering',output=str(out)))
    configs=[dict(base,group=str(seed)+'_'+method,method=method,target_seed=seed,arm='A',stage=str(seed)+'_'+method,preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for method in METHODS]
    for c in configs:
        if resumed:
            record=json.loads((root/'qualification'/c['group']/'ENGINEERING.json').read_text());assert record['status']=='PASS' and record['actual_calls']==32
        else:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));out=Path(c['output'])
        if resumed and out.exists():
            assert json.loads((out/'RESULTS.json').read_text())['physical_updates']==2000
            assert json.loads((out/'EVALUATION_EXIT.json').read_text())['exit_code']==0
            assert len((out/'physical_updates.jsonl').read_text().splitlines())==2000
            continue
        stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21])) if mode=='controller' else None
    try:
        if mode=='controller':write(root/'SELF_CHECK.json',self_check());controller(cfg,guard);return
        with patch.object(shared,'CAPS',CAPS):
            calls=shared.Calls(cfg,mode);build,setup=make_setup(cfg,calls);forward=core.Model.forward;encode=core.Model.encode;backward=torch.autograd.backward;grad=torch.autograd.grad
            def f(m,x):return calls.call(calls.kind('native'),len(x),lambda:forward(m,x))
            def e(m,x):
                kind=calls.kind('encoder');kind=kind.replace('_encoder','_teacher_encoder') if getattr(m,'is_temporal_teacher',False) else kind
                return calls.call(kind,len(x),lambda:encode(m,x))
            def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
            def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
            with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(target,'Domain',shared.Current),patch.object(target,'preflight',shared.preflight),patch.object(prior,'losses',losses),patch.object(shared,'losses',losses),patch.object(core.Model,'forward',f),patch.object(core.Model,'encode',e),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
                {'calibrate':calibrate,'detach':detach_check,'qualify':shared.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
