"""Fixed input-rotation control of the learned projection; reuse native engine."""
import copy,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import core,target,runtime,spectrum_alignment as shared,n_contribution as prior
from .module import DeterministicPool,unit,rows,centroid
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

SEEDS=(20261101,20261102)
METHODS=('S','G','GR')
CAPS=dict(calibration_encoder=768,calibration_reverse=32,detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=576,engineering_encoder=1152,engineering_reverse=240,formal_native=24300,formal_encoder=578000,formal_reverse=12024,diagnostic_reverse=60,evaluation_native=17352)


def rotation(n,seed):
    a=torch.randn(n,n,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)
    q,r=torch.linalg.qr(a);return q*torch.where(r.diag()<0,-1.,1.)[None,:]


def rotated_projection(state):
    w=state['linear.weight'].double().reshape(32,128);q=rotation(128,20261104)
    wr=w@q;gram=w@w.T;den=gram.norm().clamp_min(1e-30)
    orth=float((q.T@q-torch.eye(128,dtype=q.dtype)).abs().max())
    error=float((wr@wr.T-gram).norm()/den)
    wr32=wr.float();error32=float((wr32.double()@wr32.double().T-gram).norm()/den)
    difference=float((wr-w).norm()/w.norm().clamp_min(1e-30))
    assert orth<=1e-10 and error<=1e-10 and error32<=1e-5 and difference>=.1
    return dict(rotation=q,rotated_weight=wr32.reshape(32,128,1,1)),dict(orthogonality_max_error=orth,row_gram_relative_error_float64=error,row_gram_relative_error_float32=error32,relative_input_direction_change=difference)


def form_control(cfg):
    state=torch.load(cfg['shared_projection'],map_location='cpu',weights_only=False)
    values,checks=rotated_projection(state);runtime.save(cfg,values,cfg['projection_control'])
    write(Path(cfg['root'])/'PROJECTION_CONTROL.json',dict(status='PASS',seed=20261104,dimension=[32,128],bias_unchanged=True,source_images_read=0,medical_model_forwards=0,optimizer_updates=0,**checks))


def native_encode(model,x):
    n=model.parent.native;h=n.enc1(x);h=n.enc2(n.pool(h));h=n.enc3(n.pool(h));h=n.bottleneck(n.pool(h))
    return DeterministicPool.apply(h)


def terms(model,data,step,x=None):
    if x is None:x,_,_=data.batch('L',step)
    fill,_,_=data.batch('F',step,14);weak=torch.cat([x,fill]);seed=data.cfg['target_seed'];pos=prior.positions(seed,step,16,x.device)
    h=model.encode(torch.cat([weak,prior.strong(weak,seed,step,0),prior.strong(weak,seed,step,1)]))
    out={}
    for name,w in [('G',model.projection.linear.weight),('GR',model.rotated_projection)]:
        z=unit(F.conv2d(h,w,model.projection.linear.bias));c=centroid(*[rows(v,pos) for v in z.chunk(3)])
        out[name]=shared.spectral(c)
    assert all(torch.isfinite(v) for v in out.values());return out


def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);l=pixel_ce(model(x),y).mean();method=model.cfg['method'];u=l*0
    if method=='MINIMAL':values={};images=0
    else:
        values=terms(model,data,step,x);images=48
        if method=='G':u=u+.1*values['G']
        elif method=='GR':u=u+model.cfg['lambda_R']*values['GR']
    if getattr(model,'collect_stats',False) and step in (0,499,999,1499,1999):shared.diagnostics(model,l,u,step)
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),auxiliary_terms={k:float(v.detach()) for k,v in values.items()},accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=64 if images else 0,geometry_images=images,historical_raw_reads=0)


def make_setup(cfg,calls):
    build_base,setup_base=shared.make_setup(cfg,calls)
    value=torch.load(cfg['projection_control'],map_location='cpu',weights_only=False)['rotated_weight']
    def attach(m):
        if 'rotated_projection' not in m._buffers:m.register_buffer('rotated_projection',value.to(next(m.parameters()).device))
        return m
    def build(c):return attach(build_base(c))
    def setup(c):
        source,m,ema,opt=setup_base(c);return attach(source),attach(m),attach(ema),opt
    return build,setup


def calibrate(cfg):
    source,m,ema,opt=core.target_setup(cfg);del source,ema;data=shared.Current(cfg);before=copy.deepcopy(m.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    pars=[p for n,p in m.parent.named_parameters() if p.requires_grad and n.startswith(('native.enc','native.bottleneck'))];energies={'G':0.,'GR':0.};batches=[]
    for step in range(16):
        values=terms(m,data,step);record={}
        for j,k in enumerate(energies):
            loss=.1*values[k] if k=='G' else values[k]
            grad=torch.autograd.grad(loss,pars,retain_graph=j==0);assert all(torch.isfinite(g).all() for g in grad)
            record[k]=sum(float(g.double().square().sum()) for g in grad);energies[k]+=record[k]
        batches.append(record)
    assert min(energies.values())>=1e-12;weight=math.sqrt(energies['G']/energies['GR']);assert math.isfinite(weight) and 1e-4<=weight<=10
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);assert not opt.state and runtime.exact(before,m.state_dict())
    write(Path(cfg['root'])/'CALIBRATION.json',dict(status='PASS',seed=20261103,batches=16,lambda_R=weight,energies=energies,per_batch=batches,optimizer_updates=0,model_buffers_RNG_exact=True,shared_by_both_streams=True))


def detach_check(cfg):
    models=[]
    for group,method in [('DETACH_S','MINIMAL'),('DETACH_STOP','S')]:
        c=dict(cfg,group=group,method=method,phase='engineering',target_seed=SEEDS[0]);source,m,e,opt=core.target_setup(c);del source;models.append((m,e,opt,shared.Current(c)))
    records=[]
    for step in range(32):
        inputs=[];rngs=[]
        for m,e,opt,data in models:
            opt.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,step);inputs.append(data.last_supervised);(l+u).backward();opt.step();core.update_ema(e,m);rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models;assert runtime.exact(inputs[0],inputs[1]) and runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rngs[0],rngs[1])
        records.append(dict(step=step+1,model_opt_EMA_RNG_exact=True))
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,discarded=True,steps=records))


def self_check():
    g=torch.Generator().manual_seed(20261104);w=torch.randn(32,128,1,1,generator=g);v,checks=rotated_projection({'linear.weight':w})
    z=F.normalize(torch.randn(3,64,32,generator=g,dtype=torch.float64),dim=-1).requires_grad_();q=rotation(32,20261105)
    c=sum(a*z[i] for i,a in enumerate((.5,.25,.25)));loss=shared.spectral(c)
    assert torch.allclose(loss,shared.spectral(c@q),atol=1e-12,rtol=1e-12)
    assert torch.isfinite(torch.autograd.grad(loss,z)[0]).all()
    assert torch.equal(prior.positions(SEEDS[0],0,2,'cpu'),prior.positions(SEEDS[0],0,16,'cpu')[:2])
    image=torch.linspace(-1,1,8192).reshape(2,1,64,64);assert torch.equal(prior.strong(image,SEEDS[0],0,0),prior.strong(torch.cat([image,torch.zeros(14,1,64,64)]),SEEDS[0],0,0)[:2])
    return dict(status='PASS',output_rotation_spectrum_invariant=True,finite_feature_gradient=True,slot_prefix_exact=True,medical_model_forwards=0,optimizer_updates=0,**checks)


def controller(cfg,guard):
    root=Path(cfg['root']);preparation=root/'preparation';preparation.symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True);assert (preparation/'private_Philips_val.jsonl').is_file()
    form_control(cfg)
    def stage(mode,c):
        guard(cfg,new_stage=True);path=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(path,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=p.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    cal=root/'calibration';cal.mkdir();stage('calibrate',dict(cfg,group='CAL',method='G',target_seed=20261103,phase='engineering',output=str(cal)))
    base=dict(cfg,lambda_R=json.loads((root/'CALIBRATION.json').read_text())['lambda_R'])
    detach=root/'detach';detach.mkdir();stage('detach',dict(base,group='DETACH',method='S',phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(base,group=f'{seed}_{method}',method=method,target_seed=seed,arm='A',stage=f'{seed}_{method}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for method in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if mode=='controller':write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]))
    try:
        if mode=='controller':write(root/'SELF_CHECK.json',self_check());controller(cfg,guard);return
        with patch.object(shared,'CAPS',CAPS):
            calls=shared.Calls(cfg,mode);build,setup=make_setup(cfg,calls);forward=core.Model.forward;backward=torch.autograd.backward;grad=torch.autograd.grad
            def f(m,x):return calls.call(calls.kind('native'),len(x),lambda:forward(m,x))
            def e(m,x):return calls.call(calls.kind('encoder'),len(x),lambda:native_encode(m,x))
            def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
            def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
            with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(target,'Domain',shared.Current),patch.object(target,'preflight',shared.preflight),patch.object(prior,'losses',losses),patch.object(shared,'losses',losses),patch.object(core.Model,'forward',f),patch.object(core.Model,'encode',e),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
                {'calibrate':calibrate,'detach':detach_check,'qualify':shared.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
