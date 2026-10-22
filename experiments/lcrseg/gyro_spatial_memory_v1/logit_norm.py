"""Match macro-CE output gradient norm to ordinary CE, without extra model calls."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from . import update_spectrum as engine
from .class_balance import macro_ce,BASE_CE
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20261225,20261226)
METHODS=('GRP','GRPB','GRPBN')
CAPS=dict(detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=576,engineering_encoder=1152,engineering_reverse=240,formal_native=24300,formal_encoder=578000,formal_reverse=12024,diagnostic_reverse=60,evaluation_native=17352)
POLAR_STEPS=12256
BASE_CHECK=engine.self_check


def normalized_ce(logits,labels):
    ce=BASE_CE(logits,labels)
    with torch.no_grad():
        masks=labels[:,None]==torch.arange(logits.shape[1],device=labels.device)[None,:,None,None]
        counts=masks.sum((2,3));present=(counts>0).sum(1)
        weights=(labels.shape[-2]*labels.shape[-1])/(counts.clamp_min(1)*present[:,None])
        w=weights.gather(1,labels.flatten(1)).reshape_as(labels)
        # Analytic derivative of CE.mean() with respect to the existing logits.
        g=logits.detach().log_softmax(1).exp()-masks.to(logits.dtype)
        a=g.double().norm();b=(g.double()*w[:,None].double()).norm()
        assert torch.isfinite(a) and torch.isfinite(b) and ((a==0)==(b==0))
        factor=torch.ones_like(a) if a==0 else a/b
        factor=factor.to(ce.dtype);assert torch.isfinite(factor) and factor>0
        matched=(g.double()*w[:,None].double()*factor.double()).norm()
        error=float((matched-a).abs()/a.clamp_min(1e-30));assert error<=2e-5
        stats=dict(factor=float(factor),ordinary_logit_gradient_norm=float(a/labels.numel()),macro_logit_gradient_norm=float(b/labels.numel()),matched_logit_gradient_norm=float(matched/labels.numel()),relative_norm_error=error,zero_gradient=bool(a==0))
    return ce*w*factor,stats


def self_check():
    value=BASE_CHECK();value['numerical_SVD_calls']=10;checks=[]
    for device in ['cpu','cuda']:
        y=torch.tensor([[[0,0,0],[0,1,2]],[[3,3,3],[3,3,3]]],device=device)
        z=(torch.arange(48,dtype=torch.float32,device=device).reshape(2,4,2,3)/17).requires_grad_()
        ce=BASE_CE(z,y);out,stats=normalized_ce(z,y)
        ga=torch.autograd.grad(ce.mean(),z)[0];gb=torch.autograd.grad(out.mean(),z)[0]
        assert torch.allclose(ga.norm(),gb.norm(),rtol=2e-5,atol=1e-7)
        assert torch.allclose(out,macro_ce(z,y)*stats['factor'],rtol=2e-6)
        one,ss=normalized_ce(z[1:],y[1:]);assert ss['factor']==1 and torch.equal(one,ce[1:])
        saturated=torch.full((1,4,2,2),-1000.,device=device);saturated[:,0]=1000
        zero,zz=normalized_ce(saturated,torch.zeros((1,2,2),dtype=torch.long,device=device));assert zz['zero_gradient'] and zz['factor']==1 and torch.equal(zero,torch.zeros_like(zero))
        checks.append(dict(device=device,autograd_logit_norm_match=True,macro_direction_preserved=True,single_class_exact=True,zero_gradient_safe=True,**stats))
    value.update(normalization_checks=checks,synthetic_reverse_calls=4,extra_medical_forwards=0);return value


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['class_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,class_balance=True,logit_normalize=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=True,class_balance=name in ('GRPB','GRPBN'),logit_normalize=name=='GRPBN',variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    def ce(z,y):
        if not cfg.get('class_balance',False):return BASE_CE(z,y)
        if not cfg.get('logit_normalize',False):return macro_ce(z,y)
        value,stats=normalized_ce(z,y)
        append(Path(cfg['root'])/'normalization_events.jsonl',dict(group=cfg['group'],phase=cfg['phase'],**stats))
        return value
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.orientation,'pixel_ce',ce),patch.object(engine.shared,'pixel_ce',ce):engine.main(cfg)
