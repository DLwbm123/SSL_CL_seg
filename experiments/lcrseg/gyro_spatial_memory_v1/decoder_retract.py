"""Output-filter norm retraction with free decoder direction and bias updates."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20261211,20261212)
METHODS=('GR','GRP','GRR','GRPR')
CAPS=engine.CAPS.copy()
POLAR_STEPS=8192
BASE_SETUP=engine.make_setup
BASE_PRIOR=engine.prior.make_setup
BASE_CHECK=engine.self_check
SHAPES=((128,64,2,2),(64,128,3,3),(64,64,3,3),(64,32,2,2),(32,64,3,3),(32,32,3,3),(32,16,2,2),(16,32,3,3),(16,16,3,3),(4,16,3,3))


def rows(w,transpose):
    return (w.transpose(0,1) if transpose else w).reshape(w.shape[1] if transpose else w.shape[0],-1)


def retract(w,rho,transpose):
    a=rows(w,transpose);norm=a.norm(dim=1);active=rho>0
    assert torch.isfinite(a).all() and torch.all(norm[active]>0)
    out=a*torch.where(active,rho/norm.clamp_min(torch.finfo(a.dtype).tiny),torch.zeros_like(norm))[:,None]
    assert torch.allclose(out.norm(dim=1),rho,rtol=2e-5,atol=1e-8)
    shape=w.transpose(0,1).shape if transpose else w.shape
    return out.reshape(shape).transpose(0,1) if transpose else out.reshape(shape)


def make_setup(cfg,calls):
    def with_retraction(original_build):
        build,base=BASE_PRIOR(original_build)
        def setup(c):
            source,m,ema,opt=base(c);selected=[]
            for name,mod in m.parent.named_modules():
                if name.startswith('native.decoder.') and isinstance(mod,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)) and mod.weight.requires_grad:
                    assert mod.groups==1;tr=isinstance(mod,torch.nn.ConvTranspose2d);selected.append((name+'.weight',mod.weight,tr,rows(mod.weight.detach(),tr).norm(dim=1).clone()))
            assert tuple(tuple(p.shape) for _,p,_,_ in selected)==SHAPES
            def after(*_):
                if not c['decoder_retract']:return
                stats=[]
                with torch.no_grad():
                    for n,p,tr,rho in selected:
                        before=p.clone();p.copy_(retract(p,rho,tr));nr=rows(p,tr).norm(dim=1)
                        stats.append(dict(module=n,transpose=tr,rows=len(rho),max_relative_norm_error=float(((nr-rho).abs()/rho.clamp_min(1e-12)).max()),correction_norm=float((p-before).norm())))
                append(Path(c['root'])/'retraction_events.jsonl',dict(group=c['group'],phase=c['phase'],matrices=stats))
            # Registered before the engine's diagnostic post-hook, so actual displacement includes retraction.
            opt.register_step_post_hook(after)
            return source,m,ema,opt
        return build,setup
    with patch.object(engine.prior,'make_setup',with_retraction):return BASE_SETUP(cfg,calls)


def self_check():
    value=BASE_CHECK();value['numerical_SVD_calls']=10
    checks=[]
    for device in ['cpu','cuda']:
        for tr in [False,True]:
            w=torch.arange(1,25,dtype=torch.float32,device=device).reshape(3,2,2,2);rho=rows(w,tr).norm(dim=1);a=retract(w*2,rho,tr)
            assert torch.allclose(a,w,rtol=2e-6) and torch.allclose(retract(a,rho,tr),a,rtol=2e-6)
            assert torch.equal(retract(w,torch.zeros_like(rho),tr),torch.zeros_like(w))
            try:retract(torch.zeros_like(w),rho,tr)
            except AssertionError:pass
            else:raise AssertionError('zero proposal must reject positive reference norm')
            checks.append(dict(device=device,transpose=tr,positive_scale_invariant=True,idempotent=True,zero_reference=True,collapsed_row_rejected=True))
    value['retraction_checks']=checks;return value


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['decoder_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,decoder_retract=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=name in ('GRP','GRPR'),decoder_retract=name in ('GRR','GRPR'),variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller,make_setup=make_setup):engine.main(cfg)
