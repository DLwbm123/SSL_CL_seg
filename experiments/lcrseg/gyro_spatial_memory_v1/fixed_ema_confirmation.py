"""Three predeclared GRP streams with fixed final EMA deployment; original engine."""
import copy,json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from . import update_spectrum as engine
from .single_target_audit import rows
from ..main_style_ssl_v1.runtime import write
SEEDS=(20270501,20270502,20270503)
METHODS=('GRP',)
CAPS=dict(engineering_native=288,engineering_encoder=576,engineering_reverse=120,formal_native=12200,formal_encoder=289000,formal_reverse=6012,diagnostic_reverse=30,evaluation_native=8676)
POLAR_STEPS=6096
SOURCE_PERCENT=80.36740663770262
BASE_CHECK=engine.self_check


def source_reference(cfg):
    source=rows(cfg['original_source_rows']);entry=rows(cfg['source_val_rows'])
    keys=lambda v:[(x['patient'],x['phase']) for x in v]
    assert len(source)==len(set(keys(source)))==46 and set(keys(source))==set(keys(entry))
    assert len({x['patient'] for x in source})==23
    assert all(sum(x['patient']==pid for x in source)==2 for pid in {x['patient'] for x in source})
    result=json.loads(Path(cfg['original_source_results']).read_text())
    score=100*float(np.mean([np.mean(x['dice']) for x in source]))
    assert result['primary']=='EMA' and abs(score-SOURCE_PERCENT)<1e-9 and abs(score-100*result['val']['primary_dice'])<1e-9
    return dict(status='PASS',patients=23,volumes=46,source_primary='EMA',source_percent=score,patient_EDES_match=True,additional_model_forwards=0)


def self_check():
    result=BASE_CHECK()
    class State(torch.nn.Module):
        def __init__(self):
            super().__init__();self.parent=torch.nn.Module();self.parent.register_buffer('weight',torch.tensor([2.,-3.]));self.parent.register_buffer('grad_update',torch.tensor([4.]));self.parent.sigma=torch.nn.Module();self.parent.sigma.register_buffer('weight',torch.tensor([5.]));self.register_buffer('fixed',torch.tensor([6.]));self.parent.register_buffer('count',torch.tensor([7]))
    ema=State();model=State()
    with torch.no_grad():
        for value in model.state_dict().values():value.add_(1)
    old=copy.deepcopy(ema.state_dict());expected={k:(v*.999+model.state_dict()[k]*.001 if k=='parent.weight' else model.state_dict()[k].clone()) for k,v in old.items()}
    engine.core.update_ema(ema,model)
    assert engine.runtime.exact(expected,ema.state_dict())
    result.update(numerical_SVD_calls=10,EMA_formula_exact=True,EMA_fixed_and_integer_state_copy_exact=True,synthetic_reverse_calls=0)
    return result


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['stack_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert (Path(cfg['stack_root'])/'DELIVERY.json').is_file()
    write(root/'SOURCE_REFERENCE_CHECK.json',source_reference(cfg))
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=True,variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller):engine.main(cfg)
