"""Finite synthetic controller; no scheduler, GPU submitter or real data reader."""
import json
from pathlib import Path
import torch
from .model import Model
from .parent_bridge import SyntheticParentBridge
from .recipes import SyntheticCurrentDomain
from .train_stage import StageTrainer
from .checkpoint import save,PhysicalLedger,require_predecessor,atomic_save
from .registry import validate_family


def synthetic_sequence(family,out,cwmi=None,steps=3,seed=161,order=1,options=None):
    validate_family(family)
    if type(steps) is not int or not 1<=steps<=128:raise ValueError('synthetic two-stage budget exceeds 256')
    torch.manual_seed(seed);root=Path(out);root.mkdir(parents=True,exist_ok=True)
    parent=SyntheticParentBridge();previous=None;predecessor=None;rows=[]
    for stage in (1,2):
        identity={'family':family,'candidate_id':family+'_SYNTHETIC','seed':seed,'order':order,'stage':stage}
        if predecessor:require_predecessor(identity,predecessor)
        try:
            model=Model(parent,family,ratio=(options or {}).get('rank_ratio',.5),previous=previous)
            provider=SyntheticCurrentDomain(seed,order,stage,size=128 if family=='F2' else 16)
            opt={'total_steps':steps,**(options or {})}
            trainer=StageTrainer(model,provider,opt,cwmi)
            stage_root=root/f'stage_{stage}';ledger=PhysicalLedger(stage_root/'physical.jsonl')
            for _ in range(steps):trainer.update(physical=ledger.append)
            save(trainer,stage_root/'checkpoint.pt',identity)
            deploy=model.deploy()
            atomic_save({'synthetic_only':True,'identity':identity,'student':deploy.state_dict(),
                         'd':parent.d,'rank':parent.adapters[0].a.shape[0],'step':trainer.step},stage_root/'student.pt')
            rows.append({'identity':identity,'status':'SYNTHETIC_COMPLETE','telemetry':trainer.telemetry,
                         'probe':trainer.probe,'data_access':{'synthetic_L_batches':provider.l_reads,'synthetic_U_batches':provider.u_reads},
                         'physical':ledger.summary(trainer.step),'spectral':model.spectral})
            parent=deploy.parent;previous=deploy.transform.detach().clone();predecessor=identity
            # Release stage optimizer, EMA, Q/R and current-domain prototypes.
            del trainer,model,provider,deploy
        except Exception as e:
            (root/'failure.json').write_text(json.dumps({'identity':identity,'error':str(e),'status':'SYNTHETIC_FAILED'}))
            raise
    (root/'summary.json').write_text(json.dumps({'synthetic_only':True,'stages':rows},indent=2))
    return rows
