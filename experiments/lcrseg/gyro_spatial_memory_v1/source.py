"""Bounded codec qualification and full second-order task-code condensation."""
import copy,json
from pathlib import Path
import torch
from torch.func import functional_call
from . import core
from .data import Domain
from .module import unit,retained_bytes
from .runtime import attach,event,snapshot,restore,exact,save,guard

LIVE = None
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce


def codec_loss(model,data,step,fixed=False):
    x,y,_=data.batch('L',0 if fixed else step,4);u,_,_=data.batch('U',0 if fixed else step,4)
    allx=torch.cat([x,u]);z=model.encode(allx);recon,mask=model.generator(z,allx.shape[-2:])
    mae=(recon-allx).abs().mean();ce=-mask[:4].clamp_min(1e-8).log().gather(1,y[:,None]).mean()
    truth=torch.nn.functional.one_hot(y,4).permute(0,3,1,2).float()
    dice=(2*(mask[:4,1:]*truth[:,1:]).sum((0,2,3))+1e-6)/(mask[:4,1:].sum((0,2,3))+truth[:,1:].sum((0,2,3))+1e-6)
    return mae+ce+1-dice.mean(),dict(mae=float(mae.detach()),soft_dice=float(dice.mean().detach()),ce=float(ce.detach()))


def load_codec(cfg):
    model=core.build(cfg);saved=torch.load(Path(cfg['root'])/'codec/final.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);return model


def code_loss(cfg,model,data,step,stage,phase,paths=False):
    indices=core.code_indices(step);z=unit(model.code[indices]);x,mask=model.generator(z,(384,384))
    params={k:p for k,p in model.named_parameters() if k.startswith('parent.') and p.requires_grad}
    # Recreate clean native parameter tensors; no momentum or state from another outer step.
    inner=core.soft_ce(model(x),mask)
    derivatives=torch.autograd.grad(inner,tuple(params.values()),create_graph=True)
    event(cfg,'reserved',stage,phase)
    updated={k:p-.001*g for (k,p),g in zip(params.items(),derivatives)}
    event(cfg,'functional',stage,phase)
    q,y,_=data.batch('L',step)
    logits=functional_call(model,updated,(q,),strict=False)
    utility=pixel_ce(logits,y).mean()
    with torch.no_grad():target=model(x).softmax(1)
    consistency=(target*(target.clamp_min(1e-8).log()-mask.clamp_min(1e-8).log())).sum(1).mean()
    total=utility+.1*consistency
    if stage=='C1':total=total+.1*core.geometry(model,x,step,z,len(x))
    stats=dict(utility=float(utility.detach()),mask_kl=float(consistency.detach()))
    if paths:
        gx,gm=torch.autograd.grad(utility,(x,mask),retain_graph=True)
        stats.update(image_hypergradient=float(gx.norm()),mask_hypergradient=float(gm.norm()))
        assert gx.norm()>0 and gm.norm()>0 and torch.isfinite(gx).all() and torch.isfinite(gm).all()
    return total,stats


def setup(cfg,stage,phase):
    model=core.build(cfg) if stage=='codec' else load_codec(cfg)
    if stage=='codec':
        model.requires_grad_(False).eval();model.projection.requires_grad_(True);model.generator.requires_grad_(True)
        pars=list(model.projection.parameters())+list(model.generator.parameters());lr=.001
    else:
        initial=torch.load(Path(cfg['root'])/'initial_codes.pt',map_location='cpu',weights_only=False)
        model.code.data.copy_(initial['code']);model.requires_grad_(False).eval();model.parent.configure_stage_training();model.code.requires_grad_(True)
        pars=[model.code];lr=.01
    opt=torch.optim.Adam(pars,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    attach(opt,cfg,stage,phase);return model,opt,Domain(cfg,'Siemens')


def run(cfg):
    stage,phase=cfg['stage'],cfg['phase'];out=Path(cfg['output']);out.mkdir(exist_ok=False,parents=True)
    global LIVE
    model,opt,data=setup(cfg,stage,phase);LIVE=(model,opt,data,out)
    initial=copy.deepcopy(snapshot(model,opt,data,0));save(cfg,initial,out/'initial.pt')
    calls=0
    def objective(step,fixed=False,paths=False):
        if stage=='codec':return codec_loss(model,data,step,fixed)
        return code_loss(cfg,model,data,0 if fixed else step,stage,phase,paths)
    def step(i,fixed=False,paths=False):
        nonlocal calls
        model.zero_grad(set_to_none=True);v,stats=objective(i,fixed,paths);v.backward();opt.step();calls+=1
        return float(v.detach()),stats
    if phase=='engineering':
        first,stats=objective(0,True,stage!='codec');first=float(first.detach())
        golden=copy.deepcopy(snapshot(model,opt,data,0))
        value,details=objective(0,True);value.backward();grads=[p.grad.clone() if p.grad is not None else None for p in opt.param_groups[0]['params']]
        restore(golden,model,opt,data);value2,_=objective(0,True);value2.backward()
        assert float(value)==float(value2) and all((a is None and p.grad is None) or (a is not None and torch.equal(a,p.grad)) for a,p in zip(grads,opt.param_groups[0]['params']))
        restore(golden,model,opt,data)
        count=30 if stage=='codec' else 11
        for i in range(count):
            before=copy.deepcopy(snapshot(model,opt,data,i)) if i==5 else None
            result=step(i,True)
            if i==5:
                expected=copy.deepcopy(snapshot(model,opt,data,i+1));restore(before,model,opt,data)
                assert step(i,True)==result and exact(expected,snapshot(model,opt,data,i+1))
        last,_=objective(0,True);last=float(last.detach())
        write(out/'MEASUREMENT.json',dict(first=first,final=last,**stats))
        assert last<first-1e-4,('fixed query learning gate',first,last)
        restore(initial,model,opt,data);assert not opt.state
        step(0,False,stage!='codec')
        frozen_prefixes=('parent.','code','frames') if stage=='codec' else ('parent.','projection.','generator.','frames')
        assert all(torch.equal(v.cpu(),initial['model'][k].cpu()) for k,v in model.state_dict().items() if k.startswith(frozen_prefixes))
        save(cfg,snapshot(model,opt,data,1),out/'engineering_terminal.pt')
        write(out/'ENGINEERING.json',dict(status='PASS',discarded=True,optimizer_calls=calls,nonempty_restore_exact=True,golden_loss_gradient_exact=True,first=first,final=last,peak_bytes=torch.cuda.max_memory_allocated(),**stats))
    else:
        count=500 if stage=='codec' else 256
        for i in range(count):
            v,stats=step(i);save(cfg,snapshot(model,opt,data,i+1),out/'latest.pt')
            if i==0 or (i+1)%25==0:
                write(out/'STATUS.json',dict(status='RUNNING',step=i+1,loss=v,**stats));append(out/'telemetry.jsonl',dict(step=i+1,loss=v,**stats))
        save(cfg,snapshot(model,opt,data,count),out/'final.pt')
        if stage=='codec':
            quality(cfg,model,data,out)
        else:
            memory=dict(projection=model.projection.state_dict(),generator=model.generator.state_dict(),code=unit(model.code).detach(),frames=model.frames.detach(),historical_images_retained=False)
            byte_count=retained_bytes(model.projection,model.generator,code=memory['code'])+memory['frames'].numel()*4
            assert byte_count<=8*1024**2
            save(cfg,memory,out/'memory.pt');write(out/'MEMORY.json',dict(bytes=byte_count,atoms=32,raw_images=0,individual_source_features=0))
    write(out/'STATUS.json',dict(status='COMPLETE',stage=stage,phase=phase));write(out/'EXIT.json',dict(exit_code=0))


@torch.no_grad()
def quality(cfg,model,data,out):
    # Gate batch fixed before training by sampler(step=0); all source train-L only.
    x,y,_=data.batch('L',0,4);image,mask=model.generator(model.encode(x),x.shape[-2:]);pred=mask.argmax(1)
    mae=float((image-x).abs().mean());dice=[float((2*((pred==c)&(y==c)).sum()+1e-6)/((pred==c).sum()+(y==c).sum()+1e-6)) for c in (1,2,3)]
    # One source stream computes mean pixel and code moments; no images/features retained.
    model.eval();count=0;pixel_sum=0.;pixels=0;mean=None;square=None
    for a in (data._l,data._u):
        for i,depth in enumerate(a.depths):
            for start in range(0,depth,8):
                xx=torch.stack([a.read(i,z)['image'] for z in range(start,min(start+8,depth))]).cuda()
                zz=model.encode(xx);s=zz.sum(0);q=zz.square().sum(0)
                mean=s if mean is None else mean+s;square=q if square is None else square+q;count+=len(xx)
                pixel_sum+=float(xx.double().sum());pixels+=xx.numel()
    baseline=float((x-pixel_sum/pixels).abs().mean());mu=mean/count;sigma=(square/count-mu.square()).clamp_min(0).sqrt()
    torch.manual_seed(163);code=mu[None]+.1*sigma[None]*torch.randn(32,32,8,8,device='cuda')
    passed=mae<=.8*baseline and sum(dice)/3>=.60 and all((pred==c).any() for c in (1,2,3))
    write(out/'QUALITY.json',dict(status='PASS' if passed else 'SCIENTIFIC_STOP',mae=mae,constant_mean_mae=baseline,foreground_dice=dice,mean_foreground_dice=sum(dice)/3,training_only=True,all_source_slices_streamed=count))
    if not passed:raise RuntimeError('fixed500 codec quality gate not met; target prohibited')
    save(cfg,dict(code=code.detach(),individual_features_retained=False),Path(cfg['root'])/'initial_codes.pt')


def preserve_dirty(cfg):
    if LIVE is None:return
    model,opt,data,out=LIVE
    from ..five_frameworks_v1.checkpoint import atomic_save
    guard(cfg,200_000_000)
    state=snapshot(model,opt,data,-1);state['resumable']=False
    state['reason']='exception state; not a clean committed recovery point'
    atomic_save(state,out/'exception_state.pt')
