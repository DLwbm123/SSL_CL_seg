"""Frozen paired source experiment for a label-anchored BayeSeg shape field."""
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
from .model import SemanticShape


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


def main():
    root = Path(os.environ['RUN_ROOT'])
    p = json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='SEMANTIC_SHAPE_L1'
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

    def fresh(cut):
        model=SemanticShape(cut).double().cuda()
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
        c1=Path(p['c1_root'])
        assert json.loads((c1/'EXIT.json').read_text())['exit_code']==0
        source=data(p,cfg,'RUNMC','L')
        assert source.counts=={'L':80,'U':0}
        assert cfg.height==cfg.width==128 and cfg.dtype=='float64' and cfg.outer_lr==3e-4
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0) as out:
            initial=SemanticShape(True).double().cuda()
            prior=torch.load(c1/'results/Cut_source/PRIVATE_CHECKPOINT.pt',
                             map_location='cpu',weights_only=True)
            assert prior['step']==1600
            initial.base.load_state_dict(prior['model'])
            with torch.no_grad():
                initial.base.shape_posterior[-1].weight.zero_()
                initial.base.shape_posterior[-1].bias.zero_()
            initial.base.unet.requires_grad_(False)
            assert unet_error(initial)==0
            torch.save(initial.state_dict(),out/'PRIVATE_INITIAL.pt')
            cut=fresh(True);joint=fresh(False)
            assert all(torch.equal(cut.state_dict()[k],joint.state_dict()[k])
                       for k in cut.state_dict())
            image,label=batch(source,0,213000)
            cut.train();joint.train()
            before={k:v.detach().clone() for k,v in cut.state_dict().items()}
            outputs=[]
            for model in (cut,joint):
                task,var,logits,d=model.objectives(image,label,
                    torch.Generator(device='cuda').manual_seed(104000000))
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
                gi=torch.autograd.grad(var,list(model.shape_to_image.parameters()),
                                       retain_graph=True,allow_unused=True)
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gt)
                assert all(v is None for v in ga)
                assert (all(v is None for v in gv) if model.cut else
                        any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gv))
                assert any(v is not None and torch.isfinite(v).all() and v.norm()>0 for v in gi)
                assert all(v.grad is None for v in model.base.unet.parameters())
            for a,b in zip(outputs[0],outputs[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert all(torch.equal(before[k],cut.state_dict()[k]) for k in before)
            assert unet_error(cut)==unet_error(joint)==0
            write(root/'PREFLIGHT.json',dict(status='PASS',formal_updates=0,
                paired_initial_state_equal=True,paired_initial_outputs_equal=True,
                shape_prediction_equal_frozen_U=True,full_hierarchy_finite=True,
                task_to_qS_nonzero=True,task_to_qA_zero=True,
                Cut_variational_to_qS_zero=True,Joint_variational_to_qS_nonzero=True,
                both_variational_to_shape_decoder_nonzero=True,frozen_U_exact=True,
                state_restored=True))
            del cut,joint,initial

        cells={}
        for arm,cut_gradient in (('Cut',True),('Joint',False)):
            model=fresh(cut_gradient);model.train()
            optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],
                                       lr=cfg.outer_lr)
            status('TRAIN',arm=arm)
            with ledger.attempt(arm,seconds=9000,updates=1600) as out:
                with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image,label=batch(source,step,213000)
                        generator=torch.Generator(device='cuda').manual_seed(104000000+step)
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
            del model,optimizer
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=3200,
            source_training_slices=80,target_training=False))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()
