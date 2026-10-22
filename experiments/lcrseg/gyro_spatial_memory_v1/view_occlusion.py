"""Matched photometric strong views with only the final cutout removed."""
import json,os,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance
from . import update_spectrum as engine
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20270111,20270112)
METHODS=('GR','GRP','GRV','GRPV')
CAPS=engine.CAPS.copy()
POLAR_STEPS=8192
BASE_STRONG=engine.prior.strong
BASE_CHECK=engine.self_check


def photometric(images,seed,step,view):
    output=[];masks=[]
    for slot,image in enumerate(images.detach().cpu()):
        g=engine.prior.generator(seed,1,1,step,'strong:'+str(view)+':'+str(slot))
        rng=np.random.default_rng(int(torch.randint(2**62,(),generator=g)))
        im=Image.fromarray(((image[0].numpy()+1)*127.5).clip(0,255).round().astype(np.uint8))
        for _ in range(2):
            op=int(rng.integers(8));level=float(rng.uniform(0,1));sign=int(rng.choice([-1,1]))
            if op==1:im=ImageOps.autocontrast(im)
            elif op==2:im=ImageOps.equalize(im)
            elif op in (3,4,5):im={3:ImageEnhance.Brightness,4:ImageEnhance.Contrast,5:ImageEnhance.Sharpness}[op](im).enhance(1+sign*.9*level)
            elif op==6:im=ImageOps.posterize(im,max(1,8-round(4*level)))
            elif op==7:im=ImageOps.solarize(im,round(256*(1-level)))
        array=np.array(im,copy=True);h,w=array.shape;side=min(h,w)//2;cy,cx=int(rng.integers(h)),int(rng.integers(w))
        mask=np.zeros((h,w),dtype=bool);mask[max(0,cy-side//2):min(h,cy+side//2),max(0,cx-side//2):min(w,cx+side//2)]=True
        output.append(torch.from_numpy(array.astype(np.float32)/127.5-1)[None]);masks.append(torch.from_numpy(mask)[None])
    return torch.stack(output).to(images.device),torch.stack(masks).to(images.device)


def self_check():
    value=BASE_CHECK();value['numerical_SVD_calls']=10;checks=[]
    for device in ['cpu','cuda']:
        images=torch.linspace(-1,1,16*64*64,device=device).reshape(16,1,64,64)
        for seed in SEEDS:
            for view in [0,1]:
                photo,mask=photometric(images,seed,17,view);reference=BASE_STRONG(images,seed,17,view)
                fill=float(np.float32(128)/np.float32(127.5)-np.float32(1))
                reconstructed=torch.where(mask,torch.full_like(photo,fill),photo)
                assert torch.equal(reference,reconstructed) and torch.equal(photo[~mask],reference[~mask])
                assert torch.equal(photometric(images[:2],seed,17,view)[0],photo[:2])
                assert torch.isfinite(photo).all() and float(photo.min())>=-1 and float(photo.max())<=1
                assert mask.any() and not mask.all() and bool((photo[mask]!=reference[mask]).any())
                checks.append(dict(device=device,seed=seed,view=view,baseline_reconstruction_exact=True,outside_cutout_exact=True,slot_prefix_exact=True,removed_mask_fraction=float(mask.float().mean())))
    value.update(view_checks=checks,medical_model_forwards=0,optimizer_updates=0,synthetic_reverse_calls=0);return value


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['norm_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,no_cutout=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR',polar=name in ('GRP','GRPV'),no_cutout=name in ('GRV','GRPV'),variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    def strong(images,seed,step,view):
        value=photometric(images,seed,step,view)[0] if cfg['no_cutout'] else BASE_STRONG(images,seed,step,view)
        append(Path(cfg['root'])/'view_events.jsonl',dict(group=cfg['group'],phase=cfg['phase'],step=step,view=view,images=len(images),occlusion_removed=cfg['no_cutout']))
        return value
    with patch.multiple(engine,SEEDS=SEEDS,METHODS=METHODS,CAPS=CAPS,POLAR_STEPS=POLAR_STEPS,self_check=self_check,controller=controller),patch.object(engine.prior,'strong',strong):engine.main(cfg)
