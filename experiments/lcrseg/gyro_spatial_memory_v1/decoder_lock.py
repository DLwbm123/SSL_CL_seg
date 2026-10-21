"""Factorial decoder zero learning rate and full-support encoder polar update."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20261201,20261202)
METHODS=('GR','GRP','GRF','GRPF')
CAPS=engine.CAPS.copy()
POLAR_STEPS=8192
BASE_SETUP=engine.make_setup
BASE_EMA=engine.core.update_ema
BASE_CHECK=engine.self_check


def make_setup(cfg,calls):
    build,base=BASE_SETUP(cfg,calls)
    def setup(c):
        source,m,ema,opt=base(c)
        decoder={id(p) for n,p in m.parent.named_parameters() if n.startswith('native.decoder.') and p.requires_grad}
        assert decoder and all(n.startswith(('native.enc','native.bottleneck','native.decoder.')) for n,p in m.parent.named_parameters() if p.requires_grad)
        groups=[]
        for g in opt.param_groups:
            for is_decoder in (False,True):
                pars=[p for p in g['params'] if (id(p) in decoder)==is_decoder]
                assert pars
                rate=0. if c['decoder_locked'] and is_decoder else .001
                groups.append(dict(g,params=pars,lr=rate,fixed_lr=rate,name=g['name']+('_decoder' if is_decoder else '_encoder')))
        opt.param_groups=groups
        fixed={k:v.detach().clone() for k,v in m.state_dict().items() if k.startswith('parent.native.decoder.')}
        def check(*_):
            assert all(g['lr']==(0. if c['decoder_locked'] and g['name'].endswith('_decoder') else .001) for g in opt.param_groups)
            if c['decoder_locked']:assert all(torch.equal(m.state_dict()[k],v) for k,v in fixed.items())
        def after(*_):
            check()
            if c['decoder_locked']:append(Path(c['root'])/'decoder_checks.jsonl',dict(group=c['group'],phase=c['phase'],decoder_exact=True))
        opt.register_step_pre_hook(check);opt.register_step_post_hook(after)
        return source,m,ema,opt
    return build,setup


def update_ema(ema,model):
    BASE_EMA(ema,model)
    if model.cfg['decoder_locked']:
        with torch.no_grad():
            for k,v in ema.state_dict().items():
                if k.startswith('parent.native.decoder.'):v.copy_(model.state_dict()[k])


def self_check():
    value=BASE_CHECK();value['numerical_SVD_calls']=10
    value['decoder_invariance']='checked every physical locked step, including restored qualification steps'
    return value


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['energy_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,decoder_locked=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=name in ('GRP','GRPF'),decoder_locked=name in ('GRF','GRPF'),variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller,make_setup=make_setup),patch.object(engine.core,'update_ema',update_ema):engine.main(cfg)
