"""Paired two-view BayeSeg training with a shape-invariance intervention."""
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..budgets import ResourceLedger
from ..config import Config
from ..fullfield_bayeseg_a1.run import batch, evaluate
from ..paired_semantic_v19.run import data
from ..real_diagnostic import write
from ..semantic_latent_h1.model import LatentReadout


def contrast(image, gamma):
    assert gamma in (.7, 1.3)
    if image.min() < -1.00000001 or image.max() > 1.00000001:
        raise ValueError('image outside frozen normalized range')
    return 2 * ((image + 1) / 2).pow(gamma) - 1


def consistency(model, image, augmented):
    shape = torch.tanh(model.base.shape_posterior(image)[:, :1])
    transformed = torch.tanh(model.base.shape_posterior(augmented)[:, :1])
    return (shape-transformed).square().mean(), shape.std(unbiased=False)


def grad_audit(task, variational, invariant, parameters):
    parts=[torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
           for value in (task,variational,invariant)]
    zero=torch.zeros((),device='cuda',dtype=torch.float64)
    norms=[sum((g.square().sum() for g in part if g is not None),zero).sqrt()
           for part in parts]
    def cosine(i,j):
        dot=sum((a.mul(b).sum() for a,b in zip(parts[i],parts[j])
                 if a is not None and b is not None),zero)
        return float(dot/(norms[i]*norms[j])) if norms[i]>0 and norms[j]>0 else None
    return dict(task_grad_norm=float(norms[0]),weighted_variational_grad_norm=float(.01*norms[1]),
                invariant_grad_norm=float(norms[2]),task_vs_variational_cosine=cosine(0,1),
                task_vs_invariant_cosine=cosine(0,2),
                variational_vs_invariant_cosine=cosine(1,2))


def main():
    root=Path(os.environ['RUN_ROOT'])
    p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='CONTRAST_INVARIANCE_K1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023)
    ledger=ResourceLedger(root/'results',max_seconds=36000,max_updates=3200,
                          max_bytes=3_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    h1=Path(p['h1_root'])
    initial=torch.load(h1/'results/preflight/PRIVATE_INITIAL.pt',
                       map_location='cpu',weights_only=True)

    def fresh():
        model=LatentReadout(True).double().cuda()
        model.load_state_dict(initial)
        model.base.unet.requires_grad_(False)
        return model

    def unet_error(model):
        old=torch.load(Path(p['unet_root'])/'results/U1600/PRIVATE_CHECKPOINT.pt',
                       map_location='cpu',weights_only=True)['model']
        new=model.base.unet.state_dict();assert set(new)==set(old)
        return max(float((new[k].detach().cpu()-old[k]).abs().max()) for k in old)

    def score(arm,model,domain,split='val'):
        name=f'{arm}_{domain}'+('' if split=='val' else '_train')
        status('EVALUATE',arm=arm,domain=domain,split=split)
        with ledger.attempt(name,seconds=900) as out:
            result=evaluate(p,model,domain,split,out/'validation')
            result['predictor']='original_image_frozen_U_plus_shape_mean_head'
            result['draws']=1
            (out/'validation/SUMMARY.json').write_text(json.dumps(result,indent=2))
        return result

    try:
        assert json.loads((h1/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L')
        assert source.counts=={'L':80,'U':0}
        assert cfg.height==cfg.width==128 and cfg.dtype=='float64' and cfg.outer_lr==3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            a=fresh();s=fresh()
            assert all(torch.equal(a.state_dict()[k],s.state_dict()[k]) for k in initial)
            assert unet_error(a)==unet_error(s)==0
            image,label=batch(source,0,193000)
            augmented=contrast(image,.7)
            assert augmented.shape==image.shape and torch.isfinite(augmented).all()
            assert augmented.min()>=-1 and augmented.max()<=1
            a.train();s.train()
            common=[]
            for model in (a,s):
                originals=model.objectives(image,label,
                    torch.Generator(device='cuda').manual_seed(94000000))
                changed=model.objectives(augmented,label,
                    torch.Generator(device='cuda').manual_seed(94000001))
                inv,spatial_sd=consistency(model,image,augmented)
                common.append((originals,changed,inv,spatial_sd))
                with torch.no_grad():
                    torch.testing.assert_close(originals[3],model.base.unet(image).logits,
                                               rtol=0,atol=0)
                    torch.testing.assert_close(changed[3],model.base.unet(augmented).logits,
                                               rtol=0,atol=0)
                assert all(torch.isfinite(v) for v in (*originals[:3],*changed[:3],inv,spatial_sd))
            for i in range(2):
                for j in range(4):
                    torch.testing.assert_close(common[0][i][j],common[1][i][j],rtol=0,atol=0)
            torch.testing.assert_close(common[0][2],common[1][2],rtol=0,atol=0)
            assert common[0][2]>0
            qs=list(s.base.shape_posterior.parameters())
            qa=list(s.base.appearance_posterior.parameters())
            gs=torch.autograd.grad(common[1][2],qs,retain_graph=True,allow_unused=True)
            ga=torch.autograd.grad(common[1][2],qa,retain_graph=True,allow_unused=True)
            assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gs)
            assert all(v is None for v in ga)
            assert all(v.grad is None for v in s.base.unet.parameters())
            assert unet_error(a)==unet_error(s)==0
            assert all(torch.equal(s.state_dict()[k],initial[k].to(s.state_dict()[k].device))
                       for k in initial)
            write(root/'PREFLIGHT.json',dict(status='PASS',production_updates=0,
                paired_initial_state_equal=True,paired_two_view_outputs_equal=True,
                both_initial_outputs_equal_U=True,source_label_same_both_views=True,
                monotonic_intensity_only_no_spatial_change=True,finite_full_generator=True,
                invariant_positive_and_to_shape_nonzero=True,
                invariant_to_appearance_direct_zero=True,frozen_U_exact=True,
                preflight_state_restored=True))
            del a,s

        cells={}
        for arm,shape_inv in (('AugOnly',False),('ShapeInv',True)):
            model=fresh();model.train()
            optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                       lr=cfg.outer_lr)
            status('TRAIN',arm=arm)
            with ledger.attempt(arm,seconds=12000,updates=1600) as out:
                with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image,label=batch(source,step,193000)
                        gamma=.7 if step%2==0 else 1.3
                        augmented=contrast(image,gamma)
                        optimizer.zero_grad(set_to_none=True)
                        fo,do,vo,_,xo=model.objectives(image,label,
                            torch.Generator(device='cuda').manual_seed(94000000+2*step))
                        fa,da,va,_,xa=model.objectives(augmented,label,
                            torch.Generator(device='cuda').manual_seed(94000001+2*step))
                        task=(fo+do+fa+da)/2
                        variational=(vo+va)/2
                        invariant,shape_sd=consistency(model,image,augmented)
                        audit=(grad_audit(task,variational,invariant,
                               list(model.base.shape_posterior.parameters()))
                               if step+1 in (1,400,800,1200,1600) else None)
                        loss=task+.01*variational+(invariant if shape_inv else 0)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(v.grad is None for v in model.base.unet.parameters())
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.parameters())
                        optimizer.step()
                        mse=(float((image-xo['shape']-xo['appearance']).detach().square().mean())+
                             float((augmented-xa['shape']-xa['appearance']).detach().square().mean()))/2
                        row=dict(step=step+1,gamma=gamma,task=float(task.detach()),
                                 variational=float(variational.detach()),
                                 invariant_mse=float(invariant.detach()),shape_spatial_sd=float(shape_sd.detach()),
                                 reconstruction_mse=mse,
                                 shape_logsd=float((xo['shape_logsd'].mean()+xa['shape_logsd'].mean()).detach()/2),
                                 appearance_logsd=float((xo['appearance_logsd'].mean()+xa['appearance_logsd'].mean()).detach()/2))
                        if audit is not None:row['gradient_audit_pre_update']=audit
                        log.write(json.dumps(row)+'\n');log.flush()
                        if audit is not None:
                            tmp=out/'checkpoint.tmp'
                            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                            step=step+1,audit=audit),tmp)
                            tmp.replace(out/f'PRIVATE_CHECKPOINT_{step+1}.pt')
                assert unet_error(model)==0
                write(out/'RECEIPT.json',dict(status='COMPLETE',steps=1600,
                    unet_max_tensor_error=0,audit_steps=[1,400,800,1200,1600]))
            for domain in ('RUNMC','BMC'):
                cells[f'{arm}_{domain}']=score(arm,model,domain)
            cells[f'{arm}_RUNMC_train']=score(arm,model,'RUNMC','train_labeled')
            del model,optimizer
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=3200,
            target_U_used_for_design_only=True,target_training=False,source_training_slices=80))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
