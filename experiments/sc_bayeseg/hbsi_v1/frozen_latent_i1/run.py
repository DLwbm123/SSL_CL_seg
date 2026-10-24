"""Matched direct semantic heads with the full BayeSeg generator frozen."""
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


def main():
    root=Path(os.environ['RUN_ROOT'])
    p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='FROZEN_LATENT_I1'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023)
    ledger=ResourceLedger(root/'results',max_seconds=36000,max_updates=3200,
                          max_bytes=3_000_000_000)

    def status(stage,**extra):
        tmp=root/'STATUS.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2))
        tmp.replace(root/'STATUS.json')

    h1=Path(p['h1_root'])
    initial=torch.load(h1/'results/preflight/PRIVATE_INITIAL.pt',
                       map_location='cpu',weights_only=True)

    def fresh(use_shape):
        model=LatentReadout(use_shape).double().cuda()
        model.load_state_dict(initial)
        model.base.requires_grad_(False)
        return model

    def frozen_error(model):
        current=model.state_dict()
        return max(float((current[k].detach().cpu()-v).abs().max())
                   for k,v in initial.items() if k.startswith('base.'))

    def score(arm,model,domain,split='val'):
        name=f'{arm}_{domain}'+('' if split=='val' else '_train')
        status('EVALUATE',arm=arm,domain=domain,split=split)
        with ledger.attempt(name,seconds=900) as out:
            result=evaluate(p,model,domain,split,out/'validation')
            result['predictor']='frozen_U_plus_half_tanh_frozen_latent_head'
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
            a=fresh(False);s=fresh(True)
            assert all(torch.equal(a.state_dict()[k],s.state_dict()[k])
                       for k in initial)
            assert frozen_error(a)==frozen_error(s)==0
            assert all(not v.requires_grad for v in a.base.parameters())
            assert all(not v.requires_grad for v in s.base.parameters())
            assert all(v.requires_grad for v in a.head.parameters())
            image,label=batch(source,0,183000)
            a.train();s.train()
            av,ad,ag,al,ax=a.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(74000000))
            sv,sd,sg,sl,sx=s.objectives(image,label,
                torch.Generator(device='cuda').manual_seed(74000000))
            for v,w in ((av,sv),(ad,sd),(ag,sg),(al,sl),
                        (ax['shape'],sx['shape']),(ax['appearance'],sx['appearance'])):
                torch.testing.assert_close(v,w,rtol=0,atol=0)
            with torch.no_grad():
                u=s.base.unet(image).logits
                torch.testing.assert_close(sl,u,rtol=0,atol=0)
            assert all(torch.isfinite(v) for v in (av,ad,ag))
            (av+ad).backward();(sv+sd).backward()
            assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0
                       for v in a.head.parameters())
            assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0
                       for v in s.head.parameters())
            assert all(v.grad is None for v in a.base.parameters())
            assert all(v.grad is None for v in s.base.parameters())
            assert frozen_error(a)==frozen_error(s)==0
            write(root/'PREFLIGHT.json',dict(status='PASS',production_updates=0,
                paired_initial_state_equal=True,initial_output_equal_U=True,
                complete_generator_finite=True,head_gradient_nonzero_both=True,
                full_generator_and_U_frozen_exact=True,optimizer_head_only=True,
                target_labels_not_loaded=True))
            del a,s

        cells={}
        for arm,use_shape in (('Appearance',False),('Shape',True)):
            model=fresh(use_shape);model.train()
            optimizer=torch.optim.Adam(model.head.parameters(),lr=cfg.outer_lr)
            status('TRAIN',arm=arm)
            with ledger.attempt(arm,seconds=12000,updates=1600) as out:
                with (out/'PRIVATE_TRAIN_LOG.jsonl').open('x') as log:
                    for step in range(1600):
                        ledger.check()
                        image,label=batch(source,step,183000)
                        generator=torch.Generator(device='cuda').manual_seed(74000000+step)
                        optimizer.zero_grad(set_to_none=True)
                        fused,direct,variational,_,d=model.objectives(image,label,generator)
                        loss=fused+direct
                        assert torch.isfinite(loss) and torch.isfinite(variational)
                        loss.backward()
                        assert all(v.grad is None for v in model.base.parameters())
                        assert all(v.grad is None or torch.isfinite(v.grad).all()
                                   for v in model.head.parameters())
                        optimizer.step()
                        row=dict(step=step+1,fused_task=float(fused.detach()),
                                 direct_task=float(direct.detach()),
                                 variational=float(variational.detach()),
                                 reconstruction_mse=float((image-d['shape']-d['appearance'])
                                                          .detach().square().mean()),
                                 shape_logsd=float(d['shape_logsd'].detach().mean()),
                                 appearance_logsd=float(d['appearance_logsd'].detach().mean()))
                        log.write(json.dumps(row)+'\n');log.flush()
                        if step+1 in (400,800,1200,1600):
                            assert frozen_error(model)==0
                            tmp=out/'checkpoint.tmp'
                            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                            step=step+1),tmp)
                            tmp.replace(out/f'PRIVATE_CHECKPOINT_{step+1}.pt')
                write(out/'RECEIPT.json',dict(status='COMPLETE',steps=1600,
                    complete_frozen_generator_max_error=0,
                    checkpoints=[400,800,1200,1600]))
            for domain in ('RUNMC','BMC'):
                cells[f'{arm}_{domain}']=score(arm,model,domain)
            cells[f'{arm}_RUNMC_train']=score(arm,model,'RUNMC','train_labeled')
            del model,optimizer
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=3200,
            target_training=False,source_training_slices=80))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
