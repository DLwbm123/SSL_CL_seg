"""One loop for both registered adaptations; count every optimizer separately."""
import copy
import json
import os
import time
import traceback
from pathlib import Path
import torch
from .methods import STEPS, Segmenter, optimizer, learning_rate, losses, update_ema
from ..main_style_ssl_v1.data import CurrentDomain
from ..main_style_ssl_v1.runtime import permit, write, append
from ..siemens_lonly_v1.run import NODE, evaluate
from ..five_frameworks_v1.checkpoint import atomic_save


def setup(cfg, method):
    torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    models = [Segmenter(cfg['reference'],branch=i).cuda() for i in range(2 if method=='cps' else 1)]
    if len(models)==2 and all(torch.equal(a,b) for a,b in zip(models[0].parameters(),models[1].parameters())):
        raise ValueError('CPS branches must have independent initialization')
    provider = CurrentDomain(cfg['data'],'mnms','Siemens',163,1,0,{'kind':'source'},'cuda',permit(NODE,'formal'))
    if method=='fullsupervised':
        from .fullsupervised import bind_training_labels
        bind_training_labels(provider,cfg)
        reference=torch.load(Path(cfg['fixmatch_result'])/'initial.pt',map_location='cpu',weights_only=False)
        if reference['step']!=0 or reference['physical_updates']!=0:
            raise ValueError('reference initialization is not step zero')
        for name,value in models[0].state_dict().items():
            if not torch.equal(value.cpu(),reference['models'][0][name]):
                raise ValueError('initialization differs from FixMatch')
    opts = [optimizer(model,'cps' if method=='cps' else 'fixmatch') for model in models]
    ema = copy.deepcopy(models[0]).eval().requires_grad_(False) if method!='cps' else None
    return models,provider,opts,ema


def snapshot(models,opts,ema,step,physical,method,provider):
    return dict(method=method,step=step,physical_updates=physical,models=[m.state_dict() for m in models],
                optimizers=[o.state_dict() for o in opts],ema=None if ema is None else ema.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                L_reads=provider.l_reads,U_reads=provider.u_reads)


def main(cfg):
    method = cfg['method']
    if method not in ('fixmatch','cps','fullsupervised') or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
        raise PermissionError('only registered methods on GPU2')
    out = Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    cap = STEPS*(2 if method=='cps' else 1)
    physical = 0;committed = 0;started = time.time()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        write(out/'STATUS.json',dict(status='PREFLIGHT',method=method,physical_cap=cap))
        models,provider,opts,ema=setup(cfg,method)
        initial=snapshot(models,opts,ema,0,0,method,provider)
        atomic_save(initial,out/'initial.pt')
        loss,stats=losses(method,models,provider,0);loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for m in models for p in m.parameters()):
            raise FloatingPointError('nonfinite native preflight')
        saved=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        for m,state in zip(models,saved['models']):m.load_state_dict(state);m.zero_grad(set_to_none=True)
        if ema is not None:ema.load_state_dict(saved['ema'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        provider.l_reads=provider.u_reads=0
        write(out/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,loss=float(loss.detach()),**stats,
                                       weak_strong_U_pairs=2 if method=='cps' else 14,checkpoint_restore=True))
        del loss,initial,saved
        if cfg.get('preflight_only',False):
            write(out/'STATUS.json',dict(status='PREFLIGHT_PASS',optimizer_updates=0))
            return
        def guard(opt,args,kwargs):
            if physical>=cap:raise PermissionError('physical budget exhausted')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params']):
                raise FloatingPointError('nonfinite gradient before optimizer')
        def record(branch):
            def hook(opt,args,kwargs):
                nonlocal physical
                physical+=1
                append(out/'physical_updates.jsonl',dict(physical_call=physical,iteration=committed+1,branch=branch,optimizer_executed=True))
            return hook
        for i,opt in enumerate(opts):opt.register_step_pre_hook(guard);opt.register_step_post_hook(record(i))
        for step in range(STEPS):
            for m in models:m.train()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=learning_rate('cps' if method=='cps' else 'fixmatch',step)
            loss,stats=losses(method,models,provider,step)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward()
            for opt in opts:opt.step()
            if ema is not None:update_ema(ema,models[0])
            committed=step+1
            if committed==1 or committed%100==0:
                status=dict(status='RUNNING',method=method,step=committed,physical_updates=physical,
                            L_reads=provider.l_reads,U_reads=provider.u_reads,elapsed_seconds=time.time()-started,**stats)
                write(out/'STATUS.json',status);print(json.dumps(status),flush=True)
            if committed%2000==0:atomic_save(snapshot(models,opts,ema,committed,physical,method,provider),out/'latest.pt')
        assert physical==cap and committed==STEPS
        atomic_save(snapshot(models,opts,ema,committed,physical,method,provider),out/'final.pt')
        costs=dict(physical_optimizer_updates=physical,iterations=committed,physical_cap=cap,engineering_optimizer_updates=0,
                   L_batch_reads=provider.l_reads,U_batch_reads=provider.u_reads,L_slice_exposure=2*provider.l_reads,
                   U_slice_exposure=2*provider.u_reads,elapsed_seconds=time.time()-started)
        write(out/'COSTS.json',costs);write(out/'STATUS.json',dict(status='EVALUATING',step=committed,physical_updates=physical))
        primary=ema if ema is not None else models[0]
        train,_=evaluate(primary,cfg,'train_labeled',out/'private_train.jsonl')
        val,val_rows=evaluate(primary,cfg,'val',out/'private_val.jsonl')
        result=dict(status='COMPLETE',method=method,seed=163,train=train,val=val,physical_updates=physical,
                    iterations=committed,primary='EMA' if ema is not None else 'branch1',independent_test_access=False,
                    interpretation='Siemens U-Net adaptation of author algorithms, not reproduction of original benchmark scores; prior L-only differs in objective/optimizer.')
        if method=='fullsupervised':
            from .fullsupervised import finish
            finish(primary,provider,cfg,out,result,val_rows)
        if method=='cps':
            secondary,_=evaluate(models[1],cfg,'val',out/'private_branch2_val.jsonl')
            result['branch2_val_secondary']=secondary
        write(out/'RESULTS.json',result)
        (out/'REPORT.md').write_text(f"# {method} Siemens adaptation\n\nTrain Dice: {100*train['primary_dice']:.4f}%.\nValidation Dice: {100*val['primary_dice']:.4f}%.\nPrimary: {result['primary']}; fixed final {STEPS} iterations, {physical} physical calls.\n\n{result['interpretation']}\n")
        write(out/'STATUS.json',dict(status='COMPLETE',step=committed,physical_updates=physical));write(out/'EXIT.json',dict(exit_code=0))
    except BaseException as exc:
        failure=dict(status='FAILED',step=committed,physical_updates=physical,error=str(exc),traceback=traceback.format_exc())
        write(out/'FAILURE.json',failure);write(out/'STATUS.json',failure);write(out/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
