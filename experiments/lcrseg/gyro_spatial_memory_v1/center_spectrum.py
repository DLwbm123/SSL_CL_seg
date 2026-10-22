"""Center the current centroid matrix before the existing spectral objective."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append
SEEDS=(20270201,20270202)
METHODS=('GRP','GRPC')
CAPS=dict(detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=384,engineering_encoder=768,engineering_reverse=160,formal_native=16200,formal_encoder=385000,formal_reverse=8016,diagnostic_reverse=40,evaluation_native=11568)
POLAR_STEPS=8192
BASE_SPECTRAL=engine.shared.spectral
BASE_CHECK=engine.self_check


def centered(c):return c-c.mean(0,keepdim=True)


def self_check():
    result=BASE_CHECK();checks=[]
    for device in ['cpu','cuda']:
        c=torch.randn(64,32,generator=torch.Generator().manual_seed(163)).to(device).requires_grad_()
        a=centered(c);value=BASE_SPECTRAL(a)
        shifted=BASE_SPECTRAL(centered(c+torch.linspace(-.5,.5,32,device=device)))
        constant=BASE_SPECTRAL(centered(torch.ones_like(c)))
        original=BASE_SPECTRAL(c)
        grad,=torch.autograd.grad(value,c)
        assert torch.allclose(value,shifted,atol=2e-6,rtol=2e-6) and constant.item()==1
        assert torch.isfinite(grad).all() and grad.abs().max()>0 and grad.mean(0).abs().max()<1e-7
        assert a.mean(0).abs().max()<1e-6 and not torch.allclose(value,original,atol=1e-7,rtol=1e-7)
        assert torch.allclose(c.square().sum(),a.square().sum()+len(c)*c.mean(0).square().sum(),rtol=2e-6)
        checks.append(dict(device=device,shift_invariant=True,constant_loss_one=True,gradient_mean_zero=True,nonzero_finite_gradient=True,energy_decomposition=True))
    result.update(numerical_SVD_calls=18,centering_checks=checks,synthetic_reverse_calls=2,medical_model_forwards=0,optimizer_updates=0)
    return result


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['anchor_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,center_spectrum=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=True,center_spectrum=name=='GRPC',variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    def spectral(c):
        value=centered(c) if cfg['center_spectrum'] else c
        with torch.no_grad():
            cc=centered(c.detach().double());total=float(c.detach().double().square().sum());common=float(len(c)*c.detach().double().mean(0).square().sum());residual=float(cc.square().sum())
            error=abs(total-common-residual)/max(total,1e-30)
            rowmean=float(value.detach().mean(0).abs().max())
            assert error<1e-6 and (not cfg['center_spectrum'] or rowmean<1e-6)
            append(Path(cfg['root'])/'centering_events.jsonl',dict(group=cfg['group'],phase=cfg['phase'],rows=len(c),center_spectrum=cfg['center_spectrum'],total_energy=total,common_energy=common,residual_energy=residual,decomposition_error=error,input_mean_max=rowmean))
        return BASE_SPECTRAL(value)
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.shared,'spectral',spectral):engine.main(cfg)
