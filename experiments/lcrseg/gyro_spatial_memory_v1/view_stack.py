"""Preserve individual-view second moments with fixed weighted row stacking."""
import json,math,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append
SEEDS=(20270401,20270402)
METHODS=('GRP','GRPV')
CAPS=dict(detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=384,engineering_encoder=768,engineering_reverse=160,formal_native=16200,formal_encoder=385000,formal_reverse=8016,diagnostic_reverse=40,evaluation_native=11568)
POLAR_STEPS=8192
WEIGHTS=(.5,.25,.25)
BASE_CHECK=engine.self_check
BASE_CENTROID=engine.orientation.centroid
BASE_LOSSES=engine.orientation.losses


def stacked(*z):return torch.cat([math.sqrt(3*w)*v for w,v in zip(WEIGHTS,z)],0)


def self_check():
    result=BASE_CHECK();checks=[]
    for device in ['cpu','cuda']:
        g=torch.Generator().manual_seed(163);base=torch.nn.functional.normalize(torch.randn(64,32,generator=g),dim=-1).to(device)
        z=[base.clone().requires_grad_() for _ in range(3)]
        a=engine.shared.spectral(BASE_CENTROID(*z));b=engine.shared.spectral(stacked(*z))
        ga=torch.autograd.grad(a,z);gb=torch.autograd.grad(b,z)
        assert torch.allclose(a,b,atol=2e-6,rtol=2e-6) and all(torch.allclose(x,y,atol=2e-6,rtol=2e-4) for x,y in zip(ga,gb))
        z=[torch.nn.functional.normalize(torch.randn(64,32,generator=g),dim=-1).to(device).requires_grad_() for _ in range(3)]
        c=BASE_CENTROID(*z);v=stacked(*z);a=engine.shared.spectral(c);b=engine.shared.spectral(v)
        cov=sum(w*(q-c).T@(q-c) for w,q in zip(WEIGHTS,z))
        assert v.shape==(192,32) and torch.allclose(v.T@v/3-c.T@c,cov,atol=2e-6,rtol=2e-5)
        assert b<a and torch.allclose(v.square().sum()/3,sum(w*q.square().sum() for w,q in zip(WEIGHTS,z)),rtol=2e-6)
        grads=torch.autograd.grad(b,z);assert all(torch.isfinite(t).all() and t.norm()>0 for t in grads)
        checks.append(dict(device=device,identical_view_value_gradient_match=True,second_moment_identity=True,weighted_energy=True,finite_all_view_gradient=True,distinct_from_centroid=True))
    result.update(numerical_SVD_calls=18,stack_checks=checks,synthetic_reverse_calls=6,medical_model_forwards=0,optimizer_updates=0)
    return result


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['center_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,stack_views=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=True,stack_views=name=='GRPV',variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    def aggregate(*z):
        c=BASE_CENTROID(*z);v=stacked(*z) if cfg['stack_views'] else c
        with torch.no_grad():
            second=sum(w*float(q.detach().double().square().sum()) for w,q in zip(WEIGHTS,z));mean=float(c.detach().double().square().sum());within=sum(w*float((q.detach().double()-c.detach().double()).square().sum()) for w,q in zip(WEIGHTS,z));error=abs(second-mean-within)/max(second,1e-30)
            assert error<1e-6 and len(v)==(192 if cfg['stack_views'] else 64)
            append(Path(cfg['root'])/'stack_events.jsonl',dict(group=cfg['group'],phase=cfg['phase'],rows=len(v),stack_views=cfg['stack_views'],second_energy=second,centroid_energy=mean,within_energy=within,decomposition_error=error))
        return v
    def losses(*args):
        l,u,stats=BASE_LOSSES(*args)
        if stats['geometry_images']:stats['geometry_rows']=192 if cfg['stack_views'] else 64
        return l,u,stats
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.orientation,'centroid',aggregate),patch.object(engine.orientation,'losses',losses):engine.main(cfg)
