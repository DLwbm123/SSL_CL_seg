"""Stop only the current weak-view branch in the existing spectral centroid."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20270121,20270122)
METHODS=('GRP','GRPA')
CAPS=dict(detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=384,engineering_encoder=768,engineering_reverse=160,formal_native=16200,formal_encoder=385000,formal_reverse=8016,diagnostic_reverse=40,evaluation_native=11568)
POLAR_STEPS=8192
BASE_CENTER=engine.orientation.centroid
BASE_CHECK=engine.self_check


def anchored(weak,strong1,strong2):
    return BASE_CENTER(weak.detach(),strong1,strong2)


def self_check():
    result=BASE_CHECK();checks=[]
    for device in ['cpu','cuda']:
        a=torch.linspace(-1,1,128,device=device).reshape(4,32).requires_grad_()
        b=torch.cos(a.detach()).requires_grad_();c=torch.sin(a.detach()).requires_grad_()
        reference=BASE_CENTER(a,b,c);changed=anchored(a,b,c)
        assert torch.equal(reference,changed)
        gr=torch.autograd.grad(reference.square().sum(),(a,b,c))
        ga=torch.autograd.grad(changed.square().sum(),(a,b,c),allow_unused=True)
        assert ga[0] is None and torch.equal(gr[1],ga[1]) and torch.equal(gr[2],ga[2])
        assert torch.allclose(gr[0],reference) and torch.allclose(gr[1],reference*.5)
        checks.append(dict(device=device,forward_exact=True,weak_gradient_absent=True,strong_gradients_exact=True))
    result.update(numerical_SVD_calls=10,anchor_checks=checks,synthetic_reverse_calls=4,medical_model_forwards=0,optimizer_updates=0)
    return result


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['view_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,weak_anchor=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=True,weak_anchor=name=='GRPA',variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    def center(weak,strong1,strong2,reference=None):
        assert reference is None
        value=anchored(weak,strong1,strong2) if cfg['weak_anchor'] else BASE_CENTER(weak,strong1,strong2)
        append(Path(cfg['root'])/'anchor_events.jsonl',dict(group=cfg['group'],phase=cfg['phase'],rows=len(weak),weak_anchor=cfg['weak_anchor']))
        return value
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.orientation,'centroid',center):engine.main(cfg)
