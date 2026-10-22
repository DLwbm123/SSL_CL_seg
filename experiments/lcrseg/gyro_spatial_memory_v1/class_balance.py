"""Per-image present-class macro CE crossed with full-support encoder polar updates."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write

SEEDS=(20261221,20261222)
METHODS=('GR','GRP','GRB','GRPB')
CAPS=engine.CAPS.copy()
POLAR_STEPS=8192
BASE_CE=engine.orientation.pixel_ce
BASE_CHECK=engine.self_check


def macro_ce(logits,labels):
    ce=BASE_CE(logits,labels)
    masks=labels[:,None]==torch.arange(logits.shape[1],device=labels.device)[None,:,None,None]
    counts=masks.sum((2,3));present=(counts>0).sum(1)
    weights=(labels.shape[-2]*labels.shape[-1])/(counts.clamp_min(1)*present[:,None])
    return ce*weights.gather(1,labels.flatten(1)).reshape_as(labels)


def self_check():
    value=BASE_CHECK();value['numerical_SVD_calls']=10
    checks=[]
    for device in ['cpu','cuda']:
        y=torch.tensor([[[0,0,0],[0,1,2]],[[3,3,3],[3,3,3]]],device=device)
        z=torch.arange(48,dtype=torch.float32,device=device).reshape(2,4,2,3)/17
        ce=BASE_CE(z,y);actual=macro_ce(z,y).mean()
        expected=torch.stack([torch.stack([ce[i][y[i]==k].mean() for k in y[i].unique()]).mean() for i in range(2)]).mean()
        assert torch.allclose(actual,expected,rtol=1e-6)
        assert torch.equal(macro_ce(z[1:],y[1:]),ce[1:])
        checks.append(dict(device=device,macro_formula=True,absent_classes_excluded=True,single_class_CE_exact=True))
    value['class_balance_checks']=checks;return value


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['retract_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,class_balance=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=name in ('GRP','GRPB'),class_balance=name in ('GRB','GRPB'),variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    ce=macro_ce if cfg.get('class_balance',False) else BASE_CE
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.orientation,'pixel_ce',ce),patch.object(engine.shared,'pixel_ce',ce):engine.main(cfg)
