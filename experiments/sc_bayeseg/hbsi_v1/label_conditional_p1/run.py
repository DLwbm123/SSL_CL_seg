"""Paired conditional shape priors within a full BayeSeg-like hierarchy."""
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
from ..data_adapter import confined
from ..fullfield_bayeseg_a1.run import batch, evaluate, records
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from .model import ConditionalReadout


def grad_audit(task, variational, parameters):
    tg = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    vg = torch.autograd.grad(variational, parameters, retain_graph=True, allow_unused=True)
    zero = torch.zeros((), device='cuda', dtype=torch.float64)
    tn = sum((g.square().sum() for g in tg if g is not None), zero).sqrt()
    vn = sum((g.square().sum() for g in vg if g is not None), zero).sqrt()
    dot = sum((a.mul(b).sum() for a, b in zip(tg, vg) if a is not None and b is not None), zero)
    cosine = float(dot / (tn * vn)) if tn > 0 and vn > 0 else None
    return dict(task_grad_norm=float(tn), weighted_variational_grad_norm=float(.01 * vn),
                cosine=cosine, task_gradient_parameters=sum(g is not None for g in tg),
                variational_gradient_parameters=sum(g is not None for g in vg))


def source_latent_audit(p,model,out):
    """Patient-level source-L qS label alignment and mean image reconstruction."""
    result=[];model.eval()
    with torch.no_grad():
        for row in records(p,'RUNMC','train_labeled'):
            with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['image_h5_relpath']),'r') as f:
                volume=torch.from_numpy(f['image'][...].copy())[:,None].double()
            with h5py.File(confined(Path(p['data_root'])/'h5/v1',row['label_h5_relpath']),'r') as f:
                labels=torch.from_numpy(f['label'][...].copy())[:,None].double()
            q=[];y=[];sse=0.;pixels=0
            for image,label in zip(volume,labels):
                x=F.interpolate(image[None],(128,128),mode='bilinear',
                                align_corners=False).cuda()
                z=F.interpolate(label[None],(128,128),mode='nearest').cpu()
                shape=model.base.shape_posterior(x)[:,:1]
                appearance=model.base.appearance_posterior(x)[:,:1]
                q.append(torch.tanh(shape).cpu().flatten())
                y.append(z.flatten())
                sse+=float((x-shape-appearance).square().sum())
                pixels+=x.numel()
            qs=torch.cat(q);ys=torch.cat(y)
            centered_q=qs-qs.mean();centered_y=ys-ys.mean()
            den=(centered_q.square().sum()*centered_y.square().sum()).sqrt()
            corr=float((centered_q*centered_y).sum()/den) if den>0 else 0.
            result.append(dict(patient=row['patient_id'],slices=len(volume),
                               abs_mask_pearson=abs(corr),signed_mask_pearson=corr,
                               mean_reconstruction_mse=sse/pixels))
    write(out/'PRIVATE_PATIENTS.json',result)
    return dict(patients=4,slices=sum(v['slices'] for v in result),
                mean_abs_mask_pearson=sum(v['abs_mask_pearson'] for v in result)/4,
                mean_reconstruction_mse=sum(v['mean_reconstruction_mse'] for v in result)/4)


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root / 'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose'] == 'LABEL_CONDITIONAL_P1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = replace(Config(**p['config']), seed=20261023)
    ledger = ResourceLedger(root / 'results', max_seconds=24000, max_updates=3200,
                            max_bytes=3_000_000_000)

    def status(stage, **extra):
        tmp = root / 'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage, time=time.time(), **extra), indent=2))
        tmp.replace(root / 'STATUS.json')

    def fresh(label_condition):
        model = ConditionalReadout(label_condition).double().cuda()
        model.load_state_dict(torch.load(root / 'results/preflight/PRIVATE_INITIAL.pt',
                                         map_location='cpu', weights_only=True))
        model.base.unet.requires_grad_(False)
        return model

    def unet_error(model):
        old = torch.load(Path(p['unet_root']) / 'results/U1600/PRIVATE_CHECKPOINT.pt',
                         map_location='cpu', weights_only=True)['model']
        new = model.base.unet.state_dict()
        assert set(new) == set(old)
        return max(float((new[k].detach().cpu() - old[k]).abs().max()) for k in old)

    def score(arm, model, domain, split='val'):
        name = f'{arm}_{domain}' + ('' if split == 'val' else '_train')
        status('EVALUATE', arm=arm, domain=domain, split=split)
        with ledger.attempt(name, seconds=900) as out:
            result = evaluate(p, model, domain, split, out / 'validation')
            result['predictor'] = 'frozen_U_plus_half_tanh_latent_head'
            result['draws'] = 1
            (out / 'validation/SUMMARY.json').write_text(json.dumps(result, indent=2))
        return result

    try:
        h1 = Path(p['h1_root'])
        assert json.loads((h1 / 'EXIT.json').read_text())['exit_code'] == 0
        source = data(p, cfg, 'RUNMC', 'L')
        assert source.counts == {'L': 80, 'U': 0}
        assert cfg.height == cfg.width == 128 and cfg.dtype == 'float64' and cfg.outer_lr == 3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight', seconds=300, updates=0) as out:
            initial=ConditionalReadout(False).double().cuda()
            initial.load_state_dict(torch.load(h1/'results/preflight/PRIVATE_INITIAL.pt',
                                               map_location='cpu',weights_only=True))
            initial.base.unet.requires_grad_(False)
            assert unet_error(initial)==0
            torch.save(initial.state_dict(),out/'PRIVATE_INITIAL.pt')
            prob=fresh(False);label_model=fresh(True)
            assert all(torch.equal(prob.state_dict()[k],label_model.state_dict()[k])
                       for k in prob.state_dict())
            image, label = batch(source, 0, 233000)
            prob.train();label_model.train()
            before = {k: v.detach().clone() for k,v in prob.state_dict().items()}
            noise = 124000000
            af, ad, av, al, ax = prob.objectives(image, label,
                torch.Generator(device='cuda').manual_seed(noise))
            sf, sd, sv, sl, sx = label_model.objectives(image, label,
                torch.Generator(device='cuda').manual_seed(noise))
            for a,b in ((af,sf),(ad,sd),(av,sv),(al,sl),
                        (ax['shape'],sx['shape']),(ax['appearance'],sx['appearance'])):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            with torch.no_grad():
                base_logits=prob.base.unet(image).logits
                torch.testing.assert_close(sl,base_logits,rtol=0,atol=0)
            assert torch.isfinite(af) and torch.isfinite(ad) and torch.isfinite(av)
            assert torch.isfinite(sv) and abs(float(sv-av))>0
            with torch.no_grad():
                for model in (prob,label_model):
                    model.head[-1].weight[0].fill_(.001)
                    model.head[-1].weight[1].fill_(-.001)
            apf,apd,apv,_,_=prob.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(noise))
            shf,shd,shv,_,_=label_model.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(noise))
            aptask=apf+apd; shtask=shf+shd
            qs=list(label_model.base.shape_posterior.parameters())
            qa=list(label_model.base.appearance_posterior.parameters())
            sqs=torch.autograd.grad(shtask,qs,retain_graph=True,allow_unused=True)
            sqa=torch.autograd.grad(shtask,qa,retain_graph=True,allow_unused=True)
            assert any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in sqs)
            assert all(g is None for g in sqa)
            for model,var in ((prob,apv),(label_model,shv)):
                for params in (list(model.base.shape_posterior.parameters()),
                               list(model.base.appearance_posterior.parameters())):
                    grads=torch.autograd.grad(var,params,retain_graph=True,allow_unused=True)
                    assert any(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in grads)
                (aptask+.01*apv if model is prob else shtask+.01*shv).backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all()
                           for v in model.parameters())
            prob.load_state_dict(torch.load(out/'PRIVATE_INITIAL.pt',map_location='cpu',weights_only=True))
            label_model.load_state_dict(torch.load(out/'PRIVATE_INITIAL.pt',map_location='cpu',weights_only=True))
            assert all(torch.equal(before[k],prob.state_dict()[k]) for k in before)
            assert all(torch.equal(prob.state_dict()[k],label_model.state_dict()[k])
                       for k in prob.state_dict())
            assert unet_error(prob)==unet_error(label_model)==0
            write(root/'PREFLIGHT.json',dict(status='PASS',production_updates=0,
                paired_initial_state_equal=True,initial_output_equal_U=True,
                paired_predictions_equal=True,conditional_prior_changed_bound=True,
                full_image_generator_finite=True,frozen_U_exact=True,
                label_to_qS_nonzero_label_to_qA_zero=True,
                both_variational_to_qS_qA_nonzero=True,
                full_backward_finite=True,probe_state_restored=True))
            del prob,label_model,initial

        cells={};audits={}
        for arm,label_condition in (('ProbZ',False),('LabelY',True)):
            model=fresh(label_condition)
            model.train()
            optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                       lr=cfg.outer_lr)
            status('TRAIN',arm=arm)
            with ledger.attempt(arm,seconds=12000,updates=1600) as out:
                with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image,label=batch(source,step,233000)
                        generator=torch.Generator(device='cuda').manual_seed(124000000+step)
                        optimizer.zero_grad(set_to_none=True)
                        fused,direct,variational,_,d=model.objectives(image,label,generator)
                        task=fused+direct
                        loss=task+.01*variational
                        audit=(grad_audit(task,variational,
                               list(model.base.shape_posterior.parameters()))
                               if step+1 in (1,400,800,1200,1600) else None)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(v.grad is None for v in model.base.unet.parameters())
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.parameters())
                        optimizer.step()
                        row=dict(step=step+1,fused_task=float(fused.detach()),
                                 direct_task=float(direct.detach()),
                                 variational=float(variational.detach()),
                                 reconstruction_mse=float((image-d['shape']-d['appearance'])
                                                          .detach().square().mean()),
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
                                             unet_max_tensor_error=0,
                                             audit_steps=[1,400,800,1200,1600]))
            for domain in ('RUNMC','BMC'):
                cells[f'{arm}_{domain}']=score(arm,model,domain)
            cells[f'{arm}_RUNMC_train']=score(arm,model,'RUNMC','train_labeled')
            status('LATENT_AUDIT',arm=arm)
            with ledger.attempt(f'{arm}_LATENT_AUDIT',seconds=900,updates=0) as out:
                audits[arm]=source_latent_audit(p,model,out)
            del model,optimizer
        write(root/'RESULTS.json',dict(cells=cells,latent_audits=audits,formal_updates=3200,
              target_training=False,source_training_slices=80))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
