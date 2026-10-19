import copy
from pathlib import Path
import torch
from . import data as data_module
from ..gyro_native_v1.core import NativeGyro
from ..gyro_native_v1.module import spectral
from ..space_lora_v1.core import source_model
from ..semantic_condition_v1.core import budget_hooks
from ..siemens_ssl_reproduction_v1.methods import pixel_ce,strong_view
from ..main_style_ssl_v1.runtime import append

ARMS=('calibration','CE_high','CE_low','GYRO_low')
STEPS=69


def projection(model,images):
    m=model.parent.native
    h=m.enc1(images);h=m.enc2(m.pool(h));h=m.enc3(m.pool(h));h=m.bottleneck(m.pool(h))
    h=h.mean((-2,-1));z=model.projector(h)
    model.last_projection=dict(input_norm=float(h.detach().norm(dim=1).mean()),
        output_norm_min=float(z.detach().norm(dim=1).min()),output_norm_mean=float(z.detach().norm(dim=1).mean()),
        output_norm_max=float(z.detach().norm(dim=1).max()),
        bn_running_mean_norm=float(model.projector[1].running_mean.norm()),
        bn_running_var_mean=float(model.projector[1].running_var.mean()))
    return z


def configure_lr(opt,step):
    for group in opt.param_groups:group['lr']=group['fixed_lr']


def setup(cfg):
    torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    source=source_model(cfg)
    model=NativeGyro(source).cuda().train()
    cal=cfg['arm']=='calibration'
    if cal:
        model.parent.eval().requires_grad_(False)
    else:
        saved=torch.load(cfg['calibrated'],map_location='cpu',weights_only=False)
        assert saved['step']==saved['physical_updates']==69
        model.load_state_dict(saved['model'])
        # Deliberately do not load saved['optimizer']: both momentum histories reset.
        assert all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items())
    model.arm=cfg['arm'];model.cfg=cfg;model.query=None;model.last_projection={}
    eta=.03 if cal or cfg['arm']=='CE_high' else .001
    groups=[]
    for label,module,lr in [('native',model.parent,eta),('projector',model.projector,.03)]:
        for matrix in [True,False]:
            params=[p for p in module.parameters() if p.requires_grad and (p.ndim>1)==matrix]
            if params:groups.append(dict(params=params,lr=lr,fixed_lr=lr,name=label+('_matrix' if matrix else '_vector'),weight_decay=5e-4 if matrix else 0.))
    opt=torch.optim.SGD(groups,momentum=.9,nesterov=True)
    assert not opt.state
    ema=model.teacher()
    count=0;saved_update={}
    def before(*_):
        nonlocal saved_update
        pars=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in pars],1.,error_if_nonfinite=True)
        saved_update=dict(before={n:p.detach().clone() for n,p in pars},clip_before=float(norm),
            clip_after=sum(float(p.grad.double().square().sum()) for _,p in pars if p.grad is not None)**.5,
            lrs={g['name']:g['lr'] for g in opt.param_groups})
        if model.query is not None:
            with torch.no_grad():
                x,y=model.query;logits=model(x)
                saved_update['query_ce_before']=float(pixel_ce(logits,y).mean())
                pred=logits.argmax(1)
                saved_update['query_class_dice_before']=[float((2*((pred==c)&(y==c)).sum()+1e-5)/((pred==c).sum()+(y==c).sum()+1e-5)) for c in [1,2,3]]
    opt.register_step_pre_hook(before)
    budget_hooks(opt,cfg)
    def after(*_):
        nonlocal count
        count+=1
        row={k:v for k,v in saved_update.items() if k!='before'}
        for group,prefix in [('native','parent.'),('projector','projector.')]:
            row[group+'_update_norm']=sum(float((p.detach()-saved_update['before'][n]).double().square().sum()) for n,p in model.named_parameters() if n in saved_update['before'] and n.startswith(prefix))**.5
        if model.query is not None:
            with torch.no_grad():
                x,y=model.query;logits=model(x);row['query_ce_after']=float(pixel_ce(logits,y).mean())
                pred=logits.argmax(1)
                row['query_class_dice_after']=[float((2*((pred==c)&(y==c)).sum()+1e-5)/((pred==c).sum()+(y==c).sum()+1e-5)) for c in [1,2,3]]
        row.update(actual_call=count,phase=cfg['phase'],arm=cfg['arm'])
        append(Path(cfg['output'])/'private_update_diagnostics.jsonl',row)
    opt.register_step_post_hook(after)
    return source,model,ema,opt


def gradients(model,l,u):
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    gl=torch.autograd.grad(l,[p for _,p in params],allow_unused=True,retain_graph=True)
    gu=torch.autograd.grad(u,[p for _,p in params],allow_unused=True,retain_graph=True)
    report={}
    for group in ['enc1','enc2','enc3','bottleneck','decoder','projector','encoder_all']:
        def match(n):
            if group=='encoder_all':return n.startswith(('parent.native.enc','parent.native.bottleneck.'))
            return n.startswith('projector.' if group=='projector' else 'parent.native.'+group+'.')
        pairs=[(a,b) for (n,p),a,b in zip(params,gl,gu) if match(n)]
        a2=sum(float(a.double().square().sum()) for a,b in pairs if a is not None)
        b2=sum(float(b.double().square().sum()) for a,b in pairs if b is not None)
        dot=sum(float((a.double()*b.double()).sum()) for a,b in pairs if a is not None and b is not None)
        report[group]=dict(CE_norm=a2**.5,weighted_G_norm=b2**.5,cosine=dot/(a2*b2)**.5 if a2*b2>0 else None)
    assert all(g is None or torch.isfinite(g).all() for g in [*gl,*gu])
    return report


def losses(arm,model,data,step):
    cal=arm=='calibration'
    if cal:
        images=data.pool.epoch_batch(step);l=next(model.projector.parameters()).sum()*0
    else:
        x,y,_=data.labeled(step+69);l=pixel_ce(model(x),y).mean()
    u=l*0;stats={}
    if cal or arm=='GYRO_low':
        if not cal:images=data.pool.epoch_batch(step+69)
        z=projection(model,torch.cat([images,strong_view(images,step if cal else step+69)]))
        raw=spectral(torch.stack(z.chunk(2),dim=1),model.frames,torch.zeros(len(images),dtype=torch.long,device=z.device),0 if cal else 1/3)
        u=raw if cal else .01*raw
        stats.update(raw_spectral=float(raw.detach()),**model.last_projection)
    g=gradients(model,l,u)
    ratio=g['encoder_all']['weighted_G_norm']/max(g['encoder_all']['CE_norm'],1e-12)
    stats.update(gradient_groups=g,encoder_weighted_ratio=ratio)
    append(Path(model.cfg['output'])/'private_loss_diagnostics.jsonl',dict(step=step,arm=arm,**stats))
    if arm=='GYRO_low' and ratio>.25:raise RuntimeError('preset encoder gradient ratio gate: '+str(ratio))
    return l,u,dict(supervised=float(l.detach()),unsupervised=float(u.detach()),accepted_class_pixels=[0]*4,U_pixels=0,**stats)


@torch.no_grad()
def update_ema(ema,model):
    if model.arm=='calibration':ema.parent.load_state_dict(model.parent.state_dict())
    else:model.parent.update_dense_ema(ema.parent,decay=.999)
    ema.projector.load_state_dict(model.projector.state_dict());ema.frames.copy_(model.frames)
