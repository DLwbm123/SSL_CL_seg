"""Fixed 3000-step standard U-Net supervised qualification, no generative updates."""
import json
import os
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lcrseg.data import H5LabeledDataset, collate_labeled
from lcrseg.engine.checkpoint import capture_rng_state
from lcrseg.models.unet import UNet2D
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu, metrics, write, save, append
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.posterior_scale_readout import readout


def build_model():
    model = UNet2D(in_channels=1,num_classes=4,base_channels=16)
    # The projection branch has no role in supervised CE or the prediction interface.
    model.projection_head = nn.Identity()
    return model


@torch.no_grad()
def predict(model, image):
    return model(image).logits.softmax(1)


def run(s):
    root = Path(s['output'])
    root.mkdir(exist_ok=False)
    costs = dict(optimizer=0,reverse=0,train_image_forward=0,prediction_image_forward=0,
                 generator_forward=0,diagnostic_encoder_image_forward=0,diagnostic_posterior_batch_forward=0)
    active = {}
    started = time.time()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        storage(s,reserve=2_000_000_000)
        write(root/'CONTEXT.json',s)
        previous=Path(s['r_start'])
        assert json.loads((previous/'AUDIT.json').read_text())['status']=='PASS'
        source_history={p:p.read_text() for p in (previous/'RESULTS.json',previous/'physical_updates.jsonl',previous/'AUDIT.json')}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        assert len(sel['train_indices'])==2 and not set(sel['train_indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        batch=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(batch.image,(128,128),mode='bilinear',align_corners=False)
        y=F.interpolate(batch.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(image,k) for k in range(4)])
        y=torch.cat([flip(y,k) for k in range(4)])
        assert all(int((y[i]==k).sum())>0 for i in range(8) for k in range(4))
        diagnostic=readout(previous,image,root/'POSTERIOR_SCALE.json')
        costs['diagnostic_encoder_image_forward']=diagnostic['costs']['encoder_image_forward']
        costs['diagnostic_posterior_batch_forward']=diagnostic['costs']['posterior_batch_forward']
        torch.manual_seed(20261020)
        model=build_model().cuda()
        initial=cpu(model.state_dict())
        head=list(model.segmentation_head.parameters())
        head_ids={id(p) for p in head}
        body=[p for p in model.parameters() if id(p) not in head_ids]
        opt=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        assert not opt.state_dict()['state']
        write(root/'DESIGN.json',dict(architecture='existing UNet2D 16/32/64/128 GroupNorm, projection Identity',
              initialized_once_seed=20261020,parameters=sum(p.numel() for p in model.parameters()),
              class_counts_all_views=torch.bincount(y.flatten(),minlength=4).cpu().tolist(),
              no_pretrained_weights=True,no_old_Adam_loaded=True,formal_optimizer=3000,main_endpoint=3000))
        active.update(model=model,optimizer=opt,step=0)
        def post(optimizer,args,kwargs):
            costs['optimizer']+=1
            assert costs['optimizer']<=3000
            active['step']=costs['optimizer']
            append(root/'physical_updates.jsonl',dict(index=costs['optimizer'],step=costs['optimizer']))
        hook=opt.register_step_post_hook(post)
        def checkpoint(step):
            return dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng_state(),costs=costs)
        def evaluate(step):
            prob=predict(model,image)
            assert prob.shape==(8,4,128,128) and torch.isfinite(prob).all()
            costs['prediction_image_forward']+=8
            save(root/f'PRED_{step:04}.pt',dict(probability=prob))
            row=dict(step=step,metrics=metrics(prob.cpu(),y.cpu()))
            append(root/'metrics.jsonl',row)
            return row
        save(root/'S0000.pt',checkpoint(0))
        first=evaluate(0)
        clipped=dict(body=0,head=0)
        for step in range(1,3001):
            assert time.time()-s['original_started_unix']<85500
            opt.zero_grad(set_to_none=True)
            logits=model(image).logits
            costs['train_image_forward']+=8
            assert logits.shape==(8,4,128,128)
            loss=F.cross_entropy(logits,y,reduction='none').mean()
            assert torch.isfinite(loss)
            loss.backward()
            costs['reverse']+=1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            norms={}
            for name,params in (('body',body),('head',head)):
                norms[name]=float(nn.utils.clip_grad_norm_(params,1.))
                clipped[name]+=int(norms[name]>1.)
            opt.step()
            append(root/'steps.jsonl',dict(step=step,CE=float(loss.detach()),gradient_norms_before_clip=norms))
            if step%25==0:
                write(root/'STATUS.json',dict(status='RUNNING',step=step,costs=costs,
                      phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix']))
            if step%250==0:
                storage(s,reserve=2_000_000_000)
                save(root/f'S{step:04}.pt',checkpoint(step))
            if step in (250,500,1000,2000,3000):
                last=evaluate(step)
        hook.remove()
        assert costs==dict(optimizer=3000,reverse=3000,train_image_forward=24000,prediction_image_forward=48,
                          generator_forward=0,diagnostic_encoder_image_forward=72,diagnostic_posterior_batch_forward=9)
        assert all(int(v['step'])==3000 for v in opt.state_dict()['state'].values())
        for p,text in source_history.items():
            assert p.read_text()==text
        delta=sum(float((v.cpu().double()-initial[k].double()).square().sum()) for k,v in model.state_dict().items())**.5
        assert delta>0
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,initial=first,final=last,
              qualified24=last['metrics']['qualified_24'],clipping_counts=clipped,parameter_displacement_L2=delta,
              diagnostic='POSTERIOR_SCALE.json',source_history_unchanged=True,phase_elapsed_seconds=time.time()-started,
              original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s),
              interpretation='Training qualification only; not an architecture-only causal comparison or patient generalization'))
        write(root/'STATUS.json',dict(status='COMPLETE',step=3000,costs=costs))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:
            save(root/'failure_partial.pt',dict(step=active['step'],model=active['model'].state_dict(),
                 optimizer=active['optimizer'].state_dict(),rng=capture_rng_state(),costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()))
        write(root/'STATUS.json',dict(status='FAILED',costs=costs))
        write(root/'EXIT.json',dict(exit_code=1))
        raise


if __name__=='__main__':
    run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
