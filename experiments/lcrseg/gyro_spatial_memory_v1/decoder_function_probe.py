"""Bounded decoder-function identifiability probe; reuse the original GRP loop."""
import copy,json,math,os,subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import update_spectrum as engine
from .runtime import exact,finite
from ..main_style_ssl_v1.runtime import write
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

CAPS=dict(formal_native=300,formal_encoder=3200,formal_reverse=70,diagnostic_reverse=2)
BASE_SETUP=engine.make_setup

def features(m,x):
    n=m.parent.native;e1=n.enc1(x);e2=n.enc2(n.pool(e1));e3=n.enc3(n.pool(e2));b=n.bottleneck(n.pool(e3))
    return b,e3,e2,e1

def decode(dec,h,shape):
    b,e3,e2,e1=h;z=dec.dec1(dec.dec2(dec.dec3(b,e3),e2),e1)
    return F.interpolate(dec.conv_logit(z,stochastic=False),size=shape,mode='bilinear',align_corners=True)

def measure(cfg):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['root']);out=root/'probe';load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    ini=load(out/'initial.pt');end=load(out/'final.pt');assert end['step']==64 and end['physical_updates']==64
    build,_=engine.prior.make_setup(engine.core.build);m=build(cfg);m.register_buffer('rotated_projection',load(cfg['projection_control'])['rotated_weight'].cuda());m.load_state_dict(end['model']);m.parent.configure_stage_training();m.projection.requires_grad_(False).eval()
    d0=copy.deepcopy(m.parent.native.decoder).eval().requires_grad_(False);de=copy.deepcopy(d0)
    prefix='parent.native.decoder.'
    d0.load_state_dict({k[len(prefix):]:v for k,v in ini['model'].items() if k.startswith(prefix)});de.load_state_dict({k[len(prefix):]:v for k,v in end['ema'].items() if k.startswith(prefix)})
    before=copy.deepcopy(m.state_dict());refs=[copy.deepcopy(d.state_dict()) for d in [d0,de]];rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    c=dict(cfg,output=str(root/'calibration'),method='GR',group='CALIBRATION',target_seed=20270303);Path(c['output']).mkdir();data=engine.shared.Current(c)
    pars=[p for p in m.parent.parameters() if p.requires_grad];names=[n for n,p in m.parent.named_parameters() if p.requires_grad];dix=[i for i,n in enumerate(names) if n.startswith('native.decoder')]
    items=[];support=torch.zeros(4,dtype=torch.float64);denom=support.clone();patients=[set() for _ in range(4)];squares={'CE':0.,'H':0.,'E':0.};dot=0.;qkl=0.;classkl={'H':[0.]*4,'E':[0.]*4};classgrad={'H':[0.]*4,'E':[0.]*4};calls=dict(encoder_samples=0,decoder_samples=0,native_samples=0,reverse=0)
    def call(kind,n,fn):
        caps=dict(encoder_samples=32,decoder_samples=96,native_samples=2,reverse=176)
        assert calls[kind]+n<=caps[kind]
        write(root/'measurement_calls.json',dict(executed=calls,pending=dict(kind=kind,samples=n),caps=caps))
        value=fn();calls[kind]+=n
        write(root/'measurement_calls.json',dict(executed=calls,pending=None,caps=caps));return value
    def gradients(loss):
        g=call('reverse',1,lambda:torch.autograd.grad(loss,pars,retain_graph=True,allow_unused=True))
        assert all(g[i] is None for i,n in enumerate(names) if not n.startswith('native.decoder'))
        assert all(v is None or torch.isfinite(v).all() for v in g)
        return [torch.zeros_like(pars[i]) if g[i] is None else g[i] for i in dix]
    norm=lambda gs:sum(float(g.detach().double().square().sum()) for g in gs)
    for step in range(16):
        engine.runtime.guard(cfg);x,y,ids=data.batch('L',step);h=call('encoder_samples',len(x),lambda:features(m,x));h=tuple(v.detach() for v in h)
        z=call('decoder_samples',len(x),lambda:decode(m.parent.native.decoder,h,x.shape[-2:]))
        with torch.no_grad():
            q0=call('decoder_samples',len(x),lambda:decode(d0,h,x.shape[-2:])).softmax(1);qe=call('decoder_samples',len(x),lambda:decode(de,h,x.shape[-2:])).softmax(1)
            if step==0:
                reference=call('native_samples',len(x),lambda:m(x));assert torch.equal(z.detach(),reference)
        mask=(q0.argmax(1)==y)&(qe.argmax(1)==y);logp=z.log_softmax(1)
        maps={k:(q*(q.clamp_min(1e-30).log()-logp)).sum(1) for k,q in [('H',q0),('E',qe)]};count=mask.sum().clamp_min(1)
        losses={k:(v*mask).sum()/count for k,v in maps.items()};gs={'CE':gradients(pixel_ce(z,y).mean()),**{k:gradients(v) for k,v in losses.items()}}
        for k,g in gs.items():squares[k]+=norm(g)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['H'],gs['E']))
        qkl+=float(((q0*(q0.clamp_min(1e-30).log()-qe.clamp_min(1e-30).log())).sum(1)*mask).sum()/count)
        for cls in range(4):
            chosen=mask&(y==cls);support[cls]+=int(chosen.sum());denom[cls]+=int((y==cls).sum())
            for i,pid in enumerate(ids):
                if chosen[i].any():patients[cls].add(pid)
            for k,v in maps.items():
                term=(v*chosen).sum()/count;classkl[k][cls]+=float(term.detach());classgrad[k][cls]+=norm(gradients(term))
        items.append(dict(batch=step,valid_pixels=int(mask.sum()),pixels=int(mask.numel()),losses={k:float(v.detach()) for k,v in losses.items()}))
    cosine=dot/math.sqrt(squares['H']*squares['E']) if min(squares['H'],squares['E'])>0 else None
    distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ['H','E']}
    coverage=[float(a/b) if b else 0. for a,b in zip(support,denom)];shares={k:sum(v[1:])/max(sum(v),1e-30) for k,v in classkl.items()}
    gates=dict(foreground_coverage=all(coverage[k]>=.05 and len(patients[k])>=3 for k in [1,2,3]),foreground_loss_share=all(v>=.05 for v in shares.values()),finite_calibration=all(v is not None and math.isfinite(v) and 1e-4<=v<=10 for v in weights.values()) and min(squares.values())>=1e-12,reference_separation=distance>=.05 and qkl/16>=1e-8)
    assert exact(before,m.state_dict()) and exact(refs,[d.state_dict() for d in [d0,de]]) and exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert calls==dict(encoder_samples=32,decoder_samples=96,native_samples=2,reverse=176)
    result=dict(status='PASS' if all(gates.values()) else 'NOT_IDENTIFIABLE',gates=gates,coverage_by_class=coverage,covered_patient_counts=[len(v) for v in patients],gradient_squared=squares,gradient_cosine=cosine,normalized_gradient_distance=distance,reference_KL_mean=qkl/16,lambda_reference=weights,foreground_KL_share=shares,class_KL=classkl,class_gradient_squared=classgrad,calls=calls,per_batch=items,decoder_forward_exact=True,encoder_direct_gradient_absent=True,model_buffers_RNG_unchanged=True,references_unchanged=True,probe_discarded=True,no_validation_used=True)
    write(root/'P0_RESULT.json',result)

def main(cfg):
    root=Path(cfg['root']);mode=os.environ.get('TASK_MODE','controller')
    if mode=='measure':engine.runtime.guard=engine.install(cfg);measure(cfg);return
    if mode=='controller':
        guard=engine.install(cfg);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True);write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));write(root/'SELF_CHECK.json',engine.self_check())
        for stage in ['train','measure']:
            c=dict(cfg,group='PROBE',method='GR',polar=True,target_seed=20270301,arm='A',phase='formal',output=str(root/'probe'),stage='PROBE',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False)
            guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000;p=root/('SETTINGS_'+stage+'.private.json');write(p,c)
            with (root/(stage+'.log')).open('x') as log:child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=stage),stdout=log,stderr=subprocess.STDOUT)
            a=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],mode=stage);write(root/'STATUS.json',dict(status='RUNNING',**a));write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+stage+'.private.json'),a);code=child.wait();write(root/('EXIT_'+stage+'.json'),dict(exit_code=code));
            if code:
                write(root/'STATUS.json',dict(status='FAILED',stage=stage,exit_code=code));raise RuntimeError(stage+' failed')
        result=json.loads((root/'P0_RESULT.json').read_text());write(root/'STATUS.json',dict(status='P0_'+result['status'],formal_started=False,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
    original=engine.target.engine.main
    def short(c):
        with patch.object(engine.target.engine,'STEPS',64):return original(c)
    with patch.multiple(engine,CAPS=CAPS,POLAR_STEPS=12320),patch.object(engine.target.engine,'main',short):engine.main(cfg)
