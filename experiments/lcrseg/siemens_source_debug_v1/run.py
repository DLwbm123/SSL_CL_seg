"""One fixed source candidate; reuse the V1 update engine and paired Dice evaluator."""
import gc
import json
import os
import time
import traceback
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import setup, seal, write
from ..main_style_ssl_v1.protocol import options
from ..five_frameworks_v1.checkpoint import save, restore, PhysicalLedger
from ..siemens_lonly_v1.run import NODE, MAX_STEPS, evaluate, existing_validation, compare
from .model import HardPseudoTrainer, ConservativePseudoTrainer


def main(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
        raise PermissionError('physical GPU2 required')
    variant=cfg.get('variant','HARD_U_V1')
    if variant not in ('HARD_U_V1','CONSERVATIVE_U_V1'):
        raise ValueError('unregistered source candidate')
    limit_steps=MAX_STEPS if variant=='HARD_U_V1' else 43599
    trainer_type=HardPseudoTrainer if variant=='HARD_U_V1' else ConservativePseudoTrainer
    node={**NODE,'steps':limit_steps}
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    ledger=PhysicalLedger(out/'physical_updates.jsonl');trainer=None;started=time.time()
    identity=dict(dataset='mnms',seed=163,order=1,stage=0,domain='Siemens',arm=variant)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        write(out/'STATUS.json',dict(status='PREFLIGHT',max_physical_updates=limit_steps,variant=variant))
        trainer=setup(node,cfg['data'],cfg['reference'],'cuda',trainer_type=trainer_type)
        if any(trainer.options[k]!=v for k,v in options(limit_steps,True).items()):
            raise ValueError('source configuration changed')
        save(trainer,out/'initial.pt',identity)
        labeled,u,_=trainer.losses()
        if u is not None or abs(float(labeled.detach())-2.063466310157642)>1e-5:
            raise ValueError('initial labeled objective mismatch')
        labeled.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainer.model.parameters()):
            raise ValueError('nonfinite first backward')
        write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,first_loss=float(labeled.detach()),
                                      additional_labeled_batch_reads=1,backwards=1))
        del labeled,u
        restore(trainer,out/'initial.pt',identity)
        def limit(opt,args,kwargs):
            if trainer.physical_optimizer_updates>=limit_steps:
                raise PermissionError('candidate physical budget exhausted')
        trainer.optimizer.register_step_pre_hook(limit)
        while trainer.step<limit_steps:
            trainer.update(physical=ledger.append)
            if trainer.step==1 or trainer.step%100==0:
                row=dict(status='RUNNING',step=trainer.step,physical_updates=trainer.physical_optimizer_updates,
                         labeled_loss=trainer.last['labeled_loss'],unlabeled_loss=trainer.last.get('unlabeled_loss'),
                         active_U=trainer.last['active_U'],pseudo=trainer.last.get('pseudo'),
                         L_reads=trainer.provider.l_reads,U_reads=trainer.provider.u_reads,elapsed_seconds=time.time()-started)
                write(out/'STATUS.json',row);print(json.dumps(row),flush=True)
            if trainer.step%2000==0:save(trainer,out/'latest.pt',identity)
        save(trainer,out/'final.pt',identity)
        deployment=seal(trainer,{**node,'arm':variant},out/'sealed.pt')
        costs={**ledger.summary(trainer.step),'L_batch_reads':trainer.provider.l_reads,
               'U_batch_reads':trainer.provider.u_reads,'telemetry':trainer.telemetry,
               'max_physical_updates':limit_steps,'elapsed_seconds':time.time()-started}
        if costs['physical_optimizer_updates']!=limit_steps:raise ValueError('ledger mismatch')
        write(out/'COSTS.json',costs);write(out/'STATUS.json',dict(status='EVALUATING',step=trainer.step))
        del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
        train,train_rows=evaluate(deployment,cfg,'train_labeled',out/'private_train.jsonl')
        val,val_rows=evaluate(deployment,cfg,'val',out/'private_val.jsonl')
        old,old_rows=existing_validation(cfg)
        result=dict(status='COMPLETE',seed=163,variant=variant,train=train,val=val,
                    existing_ssl_val=old,paired_existing_minus_candidate=compare(val_rows,old_rows),
                    new_optimizer_updates=limit_steps,independent_test_access=False,
                    interpretation='Result-guided development candidate; not independent evaluation or a matched paper reproduction.')
        write(out/'RESULTS.json',result)
        objective=('class-balanced hard CE plus present-foreground Dice' if variant=='HARD_U_V1'
                   else 'confidence>0.95 hard CE averaged over the full grid, without class balancing or Dice')
        report=f'''# Siemens source candidate: {variant}

Training Dice: {train['primary_dice']*100:.4f}%.
Validation Dice: {val['primary_dice']*100:.4f}%.
Existing SSL validation Dice: {old['primary_dice']*100:.4f}%.
Candidate minus existing SSL: {(val['primary_dice']-old['primary_dice'])*100:.4f}pp.

The source U objective is {objective}. The network, initialization, data split,
augmentation, base PAS admission, LCTX, hierarchy and optimizer remain unchanged.
The fixed schedule has {limit_steps} steps. All updates count
toward the overall130,800-call budget. This is a selected development candidate with
reused validation, not an independent test or cross-paper ranking. Historical B and
this candidate differ in GPU/backend conditions. GitHub delivery pending verification.
'''
        (out/'REPORT.md').write_text(report)
        write(out/'STATUS.json',dict(status='COMPLETE',step=limit_steps));write(out/'EXIT.json',dict(exit_code=0))
    except BaseException as exc:
        step=0 if trainer is None else trainer.step
        write(out/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),**ledger.summary(step)))
        write(out/'STATUS.json',dict(status='FAILED',step=step,error=str(exc)));write(out/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
