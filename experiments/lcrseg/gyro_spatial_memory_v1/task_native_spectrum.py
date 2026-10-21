"""Frozen source sensitivity and native-spectrum controls on the existing engine."""
import copy,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np,torch
from . import core,target,runtime,spectrum_alignment as shared,n_contribution as prior
from .module import DeterministicPool,unit,rows,centroid
from .bounded_runtime import install
from ..main_style_ssl_v1.data import Accessor
from ..di_dmpa_gate1.binding import safe_asset
from ..main_style_ssl_v1.runtime import write
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

SEEDS=(20261028,20261029)
METHODS=('S','G','NI','NP','NT')
CAPS=dict(source_native=230,source_reverse=230,calibration_encoder=768,calibration_reverse=64,detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=960,engineering_encoder=1920,engineering_reverse=400,formal_native=40200,formal_encoder=962000,formal_reverse=20040,diagnostic_reverse=100,evaluation_native=28920)


class Calls(shared.Calls):
    def kind(self,operation):
        if self.mode=='source':return 'source_'+operation
        return super().kind(operation)


def native_encode(model,x):
    n=model.parent.native;h=n.enc1(x);h=n.enc2(n.pool(h));h=n.enc3(n.pool(h));h=n.bottleneck(n.pool(h))
    return DeterministicPool.apply(h)


def native_rows(field,pos):
    assert field.ndim==4 and field.shape[1:]==(128,8,8)
    return field.flatten(2).transpose(1,2).gather(1,pos[...,None].expand(-1,-1,128)).reshape(-1,128)


def task_weights(s):
    assert s.shape==(128,) and torch.isfinite(s).all() and float(s.mean())>=1e-12,'source sensitivity energy below frozen floor'
    v=(s/s.mean()).clamp(.25,4);u=.5+.5*v;w=(u/u.mean()).sqrt()
    perm=torch.randperm(128,generator=torch.Generator().manual_seed(20261030))
    cv=float(w.square().std(unbiased=False)/w.square().mean());similarity=float(torch.nn.functional.cosine_similarity(w.square(),w[perm].square(),dim=0))
    assert cv>=.05,'source task weights insufficiently differentiated'
    assert similarity<=.99,'fixed permutation insufficiently different'
    return w.float(),perm,dict(weight_squared_CV=cv,permuted_weight_squared_cosine=similarity)


def terms(model,data,step,x=None):
    if x is None:x,_,_=data.batch('L',step)
    fill,_,_=data.batch('F',step,14);weak=torch.cat([x,fill]);seed=data.cfg['target_seed'];pos=prior.positions(seed,step,16,x.device)
    h=model.encode(torch.cat([weak,prior.strong(weak,seed,step,0),prior.strong(weak,seed,step,1)]))
    out={}
    z=unit(model.projection.linear(h));out['G']=.1*shared.spectral(centroid(*[rows(v,pos) for v in z.chunk(3)]))
    for method,w in [('NI',torch.ones_like(model.task_weight)),('NP',model.task_weight_permuted),('NT',model.task_weight)]:
        z=unit(h*w[None,:,None,None]);c=centroid(*[native_rows(v,pos) for v in z.chunk(3)]);assert c.shape==(64,128)
        out[method]=shared.spectral(c)
    assert all(torch.isfinite(v) for v in out.values());return out


def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);l=pixel_ce(model(x),y).mean();method=model.cfg['method'];u=l*0
    if method=='MINIMAL':values={};images=0
    else:
        values=terms(model,data,step,x);images=48
        if method=='G':u=u+values['G']
        elif method in ('NI','NP','NT'):u=u+model.cfg['lambdas'][method]*values[method]
    if getattr(model,'collect_stats',False) and step in (0,499,999,1499,1999):shared.diagnostics(model,l,u,step)
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),auxiliary_terms={k:float(v.detach()) for k,v in values.items()},accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=64 if images else 0,geometry_images=images,historical_raw_reads=0)


def make_setup(cfg,calls):
    build_base,setup_base=shared.make_setup(cfg,calls)
    def attach(m):
        if 'task_weight' not in m._buffers:
            w=torch.ones(128)
            if cfg.get('weights_path') and Path(cfg['weights_path']).exists():w=torch.load(cfg['weights_path'],map_location='cpu',weights_only=False)['weight']
            assert w.shape==(128,) and torch.isfinite(w).all()
            perm=torch.randperm(128,generator=torch.Generator().manual_seed(20261030))
            m.register_buffer('task_weight',w.to(next(m.parameters()).device));m.register_buffer('task_weight_permuted',w[perm].to(next(m.parameters()).device))
        return m
    def build(c):return attach(build_base(c))
    def setup(c):
        source,m,ema,opt=setup_base(c)
        return attach(source),attach(m),attach(ema),opt
    return build,setup


def source_statistics(cfg):
    source,m,ema,opt=core.target_setup(cfg);del source,ema;m.eval();assert not opt.state
    before=copy.deepcopy(m.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    access=Accessor(cfg['data'],'mnms','Siemens','train_labeled');assert len(access.patients)==10 and len(access.rows)==20 and sum(access.depths)==230
    allowed={str(safe_asset(Path(cfg['data']),r[k]).resolve()) for r in access.rows for k in ('image_h5_relpath','label_h5_relpath')}
    real=h5py.File
    def restricted(name,*a,**kw):
        assert str(Path(name).resolve()) in allowed,'source statistic attempted outside source train-L'
        return real(name,*a,**kw)
    result=[];present=[0]*4;count=0;captured=[]
    handle=m.parent.native.bottleneck.register_forward_hook(lambda module,args,value:captured.append(value))
    try:
        with patch.object(h5py,'File',restricted):
            for patient in access.patients:
                patient_scores=[]
                for i in access.by_patient[patient]:
                    for z in range(access.depths[i]):
                        item=access.read(i,z);x=item['image'][None].cuda();y=item['label'][None].cuda();captured.clear();logits=m(x)
                        assert len(captured)==1;h=captured[0];assert h.shape[1]==128
                        ce=pixel_ce(logits,y);classes=torch.unique(y).tolist();loss=torch.stack([ce[y==k].mean() for k in classes]).mean()
                        for k in classes:present[k]+=1
                        g=torch.autograd.grad(loss,h)[0];score=(h.detach().double()*g.detach().double()).square().mean((0,2,3)).cpu()
                        assert torch.isfinite(loss) and torch.isfinite(score).all();patient_scores.append(score);count+=1
                result.append(torch.stack(patient_scores).mean(0))
    finally:handle.remove()
    s=torch.stack(result).mean(0);w,perm,checks=task_weights(s)
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);assert runtime.exact(before,m.state_dict()) and not opt.state and all(p.grad is None for p in m.parameters())
    runtime.save(cfg,dict(score=s,weight=w,permutation=perm),cfg['weights_path'])
    write(Path(cfg['root'])/'SOURCE_STATISTICS.json',dict(status='PASS',patients=10,EDES=20,slices=count,classes_present_slices=present,score_mean=float(s.mean()),**checks,source_train_L_only=True,optimizer_updates=0,model_buffers_RNG_exact=True,gradient_stops_at_actual_decoder_bottleneck=True,no_per_image_features_saved=True))


def calibrate(cfg):
    source,m,ema,opt=core.target_setup(cfg);del source,ema;data=shared.Current(cfg);before=copy.deepcopy(m.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    pars=[p for n,p in m.parent.named_parameters() if p.requires_grad and n.startswith(('native.enc','native.bottleneck'))];energies={k:0. for k in ('G','NI','NP','NT')};batches=[]
    for step in range(16):
        values=terms(m,data,step);record={}
        for j,k in enumerate(energies):
            grad=torch.autograd.grad(values[k],pars,retain_graph=j<3);assert all(torch.isfinite(g).all() for g in grad)
            record[k]=sum(float(g.double().square().sum()) for g in grad);energies[k]+=record[k]
        batches.append(record)
    assert min(energies.values())>=1e-12;lambdas={k:math.sqrt(energies['G']/energies[k]) for k in ('NI','NP','NT')};assert all(1e-4<=v<=10 and math.isfinite(v) for v in lambdas.values())
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);assert not opt.state and runtime.exact(before,m.state_dict())
    write(Path(cfg['root'])/'CALIBRATION.json',dict(status='PASS',seed=20261031,batches=16,lambdas=lambdas,energies=energies,per_batch=batches,optimizer_updates=0,model_buffers_RNG_exact=True,shared_by_both_streams=True))


def detach_check(cfg):
    models=[]
    for group,method in [('DETACH_S','MINIMAL'),('DETACH_STOP','S')]:
        c=dict(cfg,group=group,method=method,phase='engineering',target_seed=20261028);source,m,e,opt=core.target_setup(c);del source;models.append((m,e,opt,shared.Current(c)))
    records=[]
    for step in range(32):
        inputs=[];rngs=[]
        for m,e,opt,data in models:
            opt.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,step);inputs.append(data.last_supervised);(l+u).backward();opt.step();core.update_ema(e,m);rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models;assert runtime.exact(inputs[0],inputs[1]) and runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rngs[0],rngs[1])
        records.append(dict(step=step+1,model_opt_EMA_RNG_exact=True))
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,discarded=True,steps=records))


def self_check():
    g=torch.Generator().manual_seed(20261030);h=torch.randn(3,16,128,8,8,generator=g,dtype=torch.float64,requires_grad=True);pos=prior.positions(20261028,0,16,'cpu')
    c=centroid(*[native_rows(unit(v),pos) for v in h]);assert c.shape==(64,128)
    loss=shared.spectral(c);assert torch.isfinite(torch.autograd.grad(loss,h)[0]).all()
    assert torch.equal(unit(h[0]),unit(h[0]*torch.ones(1,128,1,1)))
    w,p,_=task_weights(torch.linspace(.001,3,128,dtype=torch.float64));assert torch.equal(w.sort().values,w[p].sort().values)
    x=torch.randn(1,128,16,16,generator=g,dtype=torch.float64,requires_grad=True);a=DeterministicPool.apply(x);b=torch.nn.functional.adaptive_avg_pool2d(x,(8,8));assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a.sum(),x)[0],torch.autograd.grad(b.sum(),x)[0])
    assert torch.equal(prior.positions(20261028,0,2,'cpu'),pos[:2])
    image=torch.linspace(-1,1,8192).reshape(2,1,64,64);assert torch.equal(prior.strong(image,20261028,0,0),prior.strong(torch.cat([image,torch.zeros(14,1,64,64)]),20261028,0,0)[:2])
    return dict(status='PASS',finite_native_spectrum_gradient=True,pooling_value_and_adjoint_exact=True,unit_identity_and_permutation=True,slot_prefix_exact=True,medical_model_forwards=0,optimizer_updates=0)


def controller(cfg,guard):
    root=Path(cfg['root']);preparation=root/'preparation';preparation.symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True);assert (preparation/'private_Philips_val.jsonl').is_file()
    def stage(mode,c):
        guard(cfg,new_stage=True);path=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(path,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=p.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    source=root/'source_statistics';source.mkdir();stage('source',dict(cfg,group='SOURCE',method='S',target_seed=20261028,phase='engineering',output=str(source)))
    cal=root/'calibration';cal.mkdir();stage('calibrate',dict(cfg,group='CAL',method='G',target_seed=20261031,phase='engineering',output=str(cal)))
    base=dict(cfg,lambdas=json.loads((root/'CALIBRATION.json').read_text())['lambdas'])
    detach=root/'detach';detach.mkdir();stage('detach',dict(base,group='DETACH',method='S',phase='engineering',target_seed=20261028,output=str(detach)))
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
            calls=Calls(cfg,mode);build,setup=make_setup(cfg,calls);forward=core.Model.forward;backward=torch.autograd.backward;grad=torch.autograd.grad
            def f(m,x):return calls.call(calls.kind('native'),len(x),lambda:forward(m,x))
            def e(m,x):return calls.call(calls.kind('encoder'),len(x),lambda:native_encode(m,x))
            def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
            def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
            with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(target,'Domain',shared.Current),patch.object(target,'preflight',shared.preflight),patch.object(prior,'losses',losses),patch.object(shared,'losses',losses),patch.object(core.Model,'forward',f),patch.object(core.Model,'encode',e),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
                {'source':source_statistics,'calibrate':calibrate,'detach':detach_check,'qualify':shared.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
