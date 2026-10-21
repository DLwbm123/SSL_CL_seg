"""Fixed 90%-energy support versus full-support equal-energy polar steps."""
import json,math,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write

SEEDS=(20261121,20261122)
METHODS=('GR','GRP','GRT','GRTP')
CAPS=engine.CAPS.copy()
POLAR_STEPS=12256


def balanced(delta,mode):
    assert mode in ('full','truncate','energy_polar')
    a=delta.reshape(delta.shape[0],-1).float()
    u,s,vh=torch.linalg.svd(a,full_matrices=False,**({'driver':'gesvd'} if a.is_cuda else {}))
    assert torch.isfinite(s).all();energy=s.square().sum();norm=energy.sqrt()
    numerical=int((s>s[0]*max(a.shape)*torch.finfo(a.dtype).eps).sum())
    r=numerical
    if r and mode!='full':r=min(r,int(torch.searchsorted(s.square().cumsum(0),.9*energy))+1)
    if not r:out=torch.zeros_like(a);fraction=1.
    else:
        fraction=float(s[:r].square().sum()/energy)
        values=s[:r]*(norm/s[:r].norm()) if mode=='truncate' else torch.ones_like(s[:r])*(norm/math.sqrt(r))
        out=(u[:,:r]*values)@vh[:r,:]
        assert mode=='full' or fraction>=.9-1e-6
    assert torch.isfinite(out).all() and torch.allclose(out.norm(),a.norm(),rtol=2e-5,atol=1e-12)
    return out.reshape_as(delta),dict(rank=r,numerical_rank=numerical,retained_energy=fraction,mode=mode,proposal_norm=float(a.norm()),balanced_norm=float(out.norm()),stable_rank=float(energy/s[0].square()) if r else 0.,cosine=float((a*out).sum()/energy) if r else 1.)


def self_check():
    checks={}
    for device in ['cpu','cuda']:
        checks[device]={}
        for mode in ['full','truncate','energy_polar']:
            a=torch.diag(torch.tensor([4.,2.,1.,0.],device=device));b,m=balanced(a,mode)
            v=[math.sqrt(7)]*3+[0.] if mode=='full' else [4*math.sqrt(21/20),2*math.sqrt(21/20),0.,0.] if mode=='truncate' else [math.sqrt(10.5)]*2+[0.,0.]
            assert torch.allclose(b,torch.diag(torch.tensor(v,device=device)),atol=1e-6,rtol=1e-5)
            assert m['rank']==(3 if mode=='full' else 2) and torch.equal(b,balanced(a,mode)[0])
            assert torch.equal(balanced(torch.zeros_like(a),mode)[0],torch.zeros_like(a))
            assert torch.allclose(balanced(a.T,mode)[0],b.T,atol=1e-6)
            q=torch.arange(1,97,dtype=torch.float32,device=device).sin().reshape(4,3,2,4);x,_=balanced(q,mode);assert x.shape==q.shape and torch.allclose(x.norm(),q.norm(),rtol=2e-5)
            checks[device][mode]=dict(status='PASS',energy_preserved=True,known_support_exact=True,zero_exact=True,repeat_exact=True)
    return dict(status='PASS',checks=checks,numerical_SVD_calls=30,medical_model_forwards=0,optimizer_updates=0)


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['update_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,spectrum_mode='energy_polar',phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    modes=dict(GR='none',GRP='full',GRT='truncate',GRTP='energy_polar')
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=name!='GR',spectrum_mode=modes[name],variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    # Each worker has one fixed intervention; detach uses the same transform twice.
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller,balanced=lambda d:balanced(d,cfg.get('spectrum_mode','full'))):engine.main(cfg)
