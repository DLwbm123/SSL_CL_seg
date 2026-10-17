"""Reuse the existing source engine, ledger, snapshots and endpoint evaluator."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from .method import ARMS,STEPS,losses
from ..siemens_ssl_reproduction_v1 import run as engine
from ..siemens_ssl_reproduction_v1.methods import losses as reference_loss
from ..main_style_ssl_v1.runtime import write,append


def _main(cfg):
    if cfg['arm'] not in ARMS or cfg['method']!='fixmatch':raise PermissionError('source protocol mismatch')
    original_setup=engine.setup
    original_evaluate=engine.evaluate
    state=dict(calls=0,ema=None,stats=None,accepted=[0]*4,queries=0,relation_groups=0,before=[])
    out=Path(cfg['output'])
    def setup(settings,method):
        models,data,opts,ema=original_setup(settings,method)
        model=models[0];state.update(ema=ema,model=model,data=data)
        initial=torch.load(cfg['reference_initial'],map_location='cpu',weights_only=False)
        assert initial['step']==0 and initial['physical_updates']==0
        assert all(torch.equal(v.cpu(),initial['models'][0][k]) for k,v in model.state_dict().items())
        del initial
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        candidate,_=losses('dense_fixmatch',model,ema,data,0);candidate.backward()
        value=float(candidate.detach());grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
        del candidate;model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);data.l_reads=data.u_reads=0
        reference,_=reference_loss('fixmatch',models,data,0);reference.backward()
        assert float(reference.detach())==value
        assert all((p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad,g)) for p,g in zip(model.parameters(),grads))
        assert abs(value-1.06440103)<1e-5
        del reference,grads;model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);data.l_reads=data.u_reads=0
        write(out/'BASELINE_IDENTITY.json',dict(status='PASS',optimizer_updates=0,initial_state_matches=True,
            disabled_module_loss_and_gradient_exact=True,first_baseline_loss=value))
        def before(opt,args,kwargs):
            log=state['calls']==0 or (state['calls']+1)%100==0
            state['before']=[p.detach().clone() for p in model.parameters() if p.requires_grad] if log else []
        def after(opt,args,kwargs):
            state['calls']+=1;s=state['stats']
            state['accepted']=[a+b for a,b in zip(state['accepted'],s['extra_accepted_class_nodes'])]
            state['queries']+=s['relation_queries'];state['relation_groups']+=s['supervised_relation_groups']
            if state['before']:
                norm=sum(float((p.detach()-v).double().square().sum()) for p,v in zip(
                    [p for p in model.parameters() if p.requires_grad],state['before']))**.5
                append(out/'relation_telemetry.jsonl',dict(step=state['calls'],actual_update_norm=norm,**s))
                state['before']=[]
        opts[0].register_step_pre_hook(before);opts[0].register_step_post_hook(after)
        return models,data,opts,ema
    def objective(method,models,data,step):
        loss,stats=losses(cfg['arm'],models[0],state['ema'],data,step)
        state['stats']=stats
        return loss,stats
    def evaluate(model,settings,role,output):
        if role!='val':return original_evaluate(model,settings,role,output)
        from .evaluate import evaluate as endpoint
        return endpoint(model,state['model'],state['data'],settings,output)
    with patch.object(engine,'evaluate',evaluate),patch.object(engine,'STEPS',STEPS),patch.object(engine,'setup',setup),patch.object(engine,'losses',objective):
        engine.main(cfg)
    if cfg.get('preflight_only'):return
    assert state['calls']==STEPS
    result=json.loads((out/'RESULTS.json').read_text())
    result.update(arm=cfg['arm'],method='source_label_anchored_relation_v1' if cfg['arm']=='label_relation' else 'dense_fixmatch',
        interpretation='Source-only 10000-step pilot with matched initialization/L/U/optimizer; no CL efficacy claim or independent test.')
    write(out/'RESULTS.json',result)
    costs=json.loads((out/'COSTS.json').read_text())
    costs.update(extra_accepted_class_nodes=state['accepted'],relation_query_nodes=state['queries'],
                 supervised_relation_groups=state['relation_groups'],extra_teacher_image_forwards=16*STEPS if cfg['arm']=='label_relation' else 0)
    write(out/'COSTS.json',costs)
    (out/'REPORT.md').write_text(f"# {cfg['arm']} source pilot\n\nFixed final EMA; 10000 actual calls.\nTrain Dice: {100*result['train']['primary_dice']:.4f}%.\nValidation Dice: {100*result['val']['primary_dice']:.4f}%.\n\n{result['interpretation']}\n")


def main(cfg):
    existed=Path(cfg['output']).exists()
    try:
        _main(cfg)
    except BaseException as exc:
        out=Path(cfg['output'])
        if not existed and out.exists() and not (out/'FAILURE.json').exists():
            ledger=out/'physical_updates.jsonl'
            calls=len(ledger.read_text().splitlines()) if ledger.exists() else 0
            failure=dict(status='FAILED',physical_updates=calls,error=str(exc),retry=False)
            write(out/'FAILURE.json',failure);write(out/'STATUS.json',failure);write(out/'EXIT.json',dict(exit_code=1))
        raise


if __name__=='__main__':main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
