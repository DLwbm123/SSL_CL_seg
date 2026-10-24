"""Paired full-field versus low-bandwidth BayeSeg appearance posterior."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F

from ..budgets import ResourceLedger
from ..config import Config
from ..data_adapter import confined, restore_probability
from ..fullfield_bayeseg_a1.run import batch, evaluate, records
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ...sslcl_v1.prostate_metrics import binary_phase_metrics
from .model import BandwidthModel


def gradient_audit(task, variational, parameters):
    tg = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    vg = torch.autograd.grad(variational, parameters, retain_graph=True, allow_unused=True)
    zero = torch.zeros((), device='cuda', dtype=torch.float64)
    tn = sum((v.square().sum() for v in tg if v is not None), zero).sqrt()
    vn = sum((v.square().sum() for v in vg if v is not None), zero).sqrt()
    dot = sum((a.mul(b).sum() for a,b in zip(tg,vg) if a is not None and b is not None), zero)
    return dict(task_norm=float(tn),weighted_variational_norm=float(.01*vn),
                cosine=float(dot/(tn*vn)) if tn>0 and vn>0 else None,
                task_tensors=sum(v is not None for v in tg),
                variational_tensors=sum(v is not None for v in vg))


def swap_audit(p,model,ledger,out,arm):
    """A fixed source-L cross-patient swap after the sealed final update."""
    rows=records(p,'RUNMC','train_labeled')
    images=[];labels=[];spacings=[]
    for row in rows:
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['image_h5_relpath']),'r') as f:
            images.append(torch.from_numpy(f['image'][...].copy())[:,None].double())
            spacings.append(json.loads(f.attrs['processed_spacing']))
        with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['label_h5_relpath']),'r') as f:
            labels.append(f['label'][...].copy())
    assert len(rows)==4 and sum(len(v) for v in images)==80
    assert all(rows[i]['patient_id']!=rows[(i+1)%4]['patient_id'] for i in range(4))
    arm_rows=[];model.eval()
    with torch.no_grad():
        for i,(row,volume,truth,spacing) in enumerate(zip(rows,images,labels,spacings)):
            partner=images[(i+1)%4]
            predictions={key:[] for key in ('original','self','swap')}
            reconstruction_sse=0.;reconstruction_pixels=0
            swap_outside=0;swap_pixels=0
            for k,slice_ in enumerate(volume):
                ledger.check()
                image=F.interpolate(slice_[None],(128,128),mode='bilinear',
                                    align_corners=False).cuda()
                j=k*len(partner)//len(volume)
                other=F.interpolate(partner[j:j+1],(128,128),mode='bilinear',
                                    align_corners=False).cuda()
                shape=model.shape_to_image(torch.tanh(model.means(image)[0]))
                own_a=model.appearance_mean(image)
                other_a=model.appearance_mean(other)
                self_image=shape+own_a
                swap_image=shape+other_a
                assert torch.isfinite(self_image).all() and torch.isfinite(swap_image).all()
                reconstruction_sse+=float((self_image-image).square().sum())
                reconstruction_pixels+=image.numel()
                swap_outside+=int(((swap_image < -1)|(swap_image > 1)).sum())
                swap_pixels+=swap_image.numel()
                for key,value in (('original',image),('self',self_image),('swap',swap_image)):
                    prob=model.base.unet(value).logits.softmax(1).cpu()
                    predictions[key].append(prob)
            metrics={}
            for key,parts in predictions.items():
                prob=restore_probability(torch.cat(parts),volume.shape[-2:])
                metrics[key]=binary_phase_metrics(prob,truth,spacing)
                torch.save(prob,out/f'PRIVATE_{arm}_{i}_{key}.pt')
            arm_rows.append(dict(patient=row['patient_id'],metrics=metrics,
                reconstruction_mse=reconstruction_sse/reconstruction_pixels,
                swap_outside_fraction=swap_outside/swap_pixels,slices=len(volume)))
    write(out/f'PRIVATE_{arm}_PATIENTS.json',arm_rows)
    count=sum(v['slices'] for v in arm_rows)
    return dict(patients=4,slices=count,
        reconstruction_mse=sum(v['reconstruction_mse']*v['slices'] for v in arm_rows)/count,
        swap_outside_fraction=sum(v['swap_outside_fraction']*v['slices'] for v in arm_rows)/count,
        **{key+'_Dice':sum(v['metrics'][key]['classes'][0]['Dice'] for v in arm_rows)/4
           for key in ('original','self','swap')})


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='APPEARANCE_BANDWIDTH_N1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023)
    ledger=ResourceLedger(root/'results',max_seconds=24000,max_updates=3200,
                          max_bytes=3_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    def fresh(smooth):
        model=BandwidthModel(smooth).double().cuda()
        model.load_state_dict(torch.load(root/'results/preflight/PRIVATE_INITIAL.pt',
                                         map_location='cpu',weights_only=True))
        model.base.unet.requires_grad_(False)
        return model

    def unet_error(model):
        old=torch.load(Path(p['unet_root'])/'results/U1600/PRIVATE_CHECKPOINT.pt',
                       map_location='cpu',weights_only=True)['model']
        new=model.base.unet.state_dict()
        assert set(new)==set(old)
        return max(float((new[k].detach().cpu()-old[k]).abs().max()) for k in old)

    def score(arm,model,domain,split='val'):
        name=f'{arm}_{domain}'+('' if split=='val' else '_train')
        status('EVALUATE',arm=arm,domain=domain,split=split)
        with ledger.attempt(name,seconds=900) as out:
            result=evaluate(p,model,domain,split,out/'validation')
            result['predictor']='semantic_shape_posterior_mean'
            result['draws']=1
            (out/'validation/SUMMARY.json').write_text(json.dumps(result,indent=2))
        return result

    try:
        l1=Path(p['l1_root'])
        assert json.loads((l1/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L')
        assert source.counts=={'L':80,'U':0}
        assert cfg.height==cfg.width==128 and cfg.dtype=='float64' and cfg.outer_lr==3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0) as out:
            initial=BandwidthModel(False).double().cuda()
            prior=torch.load(l1/'results/preflight/PRIVATE_INITIAL.pt',
                             map_location='cpu',weights_only=True)
            initial.load_state_dict(BandwidthModel.wrapped_state(prior))
            initial.base.unet.requires_grad_(False)
            assert unet_error(initial)==0
            torch.save(initial.state_dict(),out/'PRIVATE_INITIAL.pt')
            full=fresh(False);smooth=fresh(True)
            assert all(torch.equal(full.state_dict()[k],smooth.state_dict()[k])
                       for k in full.state_dict())
            image,label=batch(source,0,223000)
            full.train();smooth.train()
            before={k:v.detach().clone() for k,v in full.state_dict().items()}
            outputs=[]
            for model in (full,smooth):
                task,var,logits,d=model.objectives(image,label,
                    torch.Generator(device='cuda').manual_seed(114000000))
                with torch.no_grad():
                    base=model.base.unet(image).logits
                    torch.testing.assert_close(logits.softmax(1),base.softmax(1),rtol=0,atol=1e-15)
                assert all(torch.isfinite(v).all() for v in (task,var,logits,d['image_shape']))
                outputs.append((task,var,logits))
                qa=list(model.base.appearance_posterior.parameters())
                qs=model.shape_parameters()
                gt=torch.autograd.grad(task,qs,retain_graph=True,allow_unused=True)
                ga=torch.autograd.grad(task,qa,retain_graph=True,allow_unused=True)
                gv=torch.autograd.grad(var,qs,retain_graph=True,allow_unused=True)
                gva=torch.autograd.grad(var,qa,retain_graph=True,allow_unused=True)
                gi=torch.autograd.grad(var,list(model.shape_to_image.parameters()),
                                       retain_graph=True,allow_unused=True)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gt)
                assert all(v is None for v in ga)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gv)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gva)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gi)
                (task+.01*var).backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all()
                           for v in model.parameters())
                assert all(v.grad is None for v in model.base.unet.parameters())
            for i in (0,2):torch.testing.assert_close(outputs[0][i],outputs[1][i],rtol=0,atol=0)
            raw=full.base.appearance_posterior(image)
            expected=F.interpolate(F.avg_pool2d(raw,32),size=(128,128),
                                   mode='bilinear',align_corners=False)
            filtered=smooth.base.appearance_posterior(image)
            torch.testing.assert_close(filtered,expected,rtol=0,atol=0)
            assert not torch.equal(raw,filtered)
            assert all(torch.equal(before[k],full.state_dict()[k]) for k in before)
            assert unet_error(full)==unet_error(smooth)==0
            write(root/'PREFLIGHT.json',dict(status='PASS',formal_updates=0,
                paired_initial_state_equal=True,paired_initial_outputs_equal=True,
                shape_prediction_equal_frozen_U=True,full_hierarchy_finite=True,
                task_to_qS_nonzero=True,task_to_qA_zero=True,
                both_variational_to_qS_qA_and_shape_decoder_nonzero=True,
                smooth_exact_fixed_4x4_filter=True,full_backward_deterministic=True,
                frozen_U_exact=True,
                state_restored=True))
            del full,smooth,initial

        cells={}
        swap={}
        for arm,smooth_appearance in (('Full',False),('Smooth',True)):
            model=fresh(smooth_appearance);model.train()
            optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                       lr=cfg.outer_lr)
            status('TRAIN',arm=arm)
            with ledger.attempt(arm,seconds=9000,updates=1600) as out:
                with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image,label=batch(source,step,223000)
                        generator=torch.Generator(device='cuda').manual_seed(114000000+step)
                        optimizer.zero_grad(set_to_none=True)
                        task,var,logits,d=model.objectives(image,label,generator)
                        audit=(gradient_audit(task,var,model.shape_parameters())
                               if step+1 in (1,400,800,1200,1600) else None)
                        loss=task+.01*var
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(v.grad is None for v in model.base.unet.parameters())
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.parameters())
                        optimizer.step()
                        row=dict(step=step+1,task=float(task.detach()),
                                 variational=float(var.detach()),
                                 reconstruction_mse=float((image-d['image_shape']-d['appearance'])
                                                          .detach().square().mean()),
                                 shape_mean_sd=float(logits[:,1].detach().std(unbiased=False)),
                                 shape_logsd=float(d['shape_logsd'].detach().mean()),
                                 appearance_logsd=float(d['appearance_logsd'].detach().mean()))
                        if audit is not None:row['gradient_audit_pre_update']=audit
                        log.write(json.dumps(row)+'\n');log.flush()
                        if audit is not None:
                            tmp=out/'checkpoint.tmp'
                            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                            step=step+1,audit=audit),tmp)
                            tmp.replace(out/f'PRIVATE_CHECKPOINT_{step+1}.pt')
                assert unet_error(model)==0
                write(out/'RECEIPT.json',dict(status='COMPLETE',steps=1600,
                    frozen_U_max_tensor_error=0,audit_steps=[1,400,800,1200,1600]))
            for domain in ('RUNMC','BMC'):
                cells[f'{arm}_{domain}']=score(arm,model,domain)
            cells[f'{arm}_RUNMC_train']=score(arm,model,'RUNMC','train_labeled')
            status('SWAP_AUDIT',arm=arm)
            with ledger.attempt(f'{arm}_SWAP',seconds=900,updates=0) as out:
                swap[arm]=swap_audit(p,model,ledger,out,arm)
            del model,optimizer
        write(root/'RESULTS.json',dict(cells=cells,swap=swap,formal_updates=3200,
            source_training_slices=80,target_training=False))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()
