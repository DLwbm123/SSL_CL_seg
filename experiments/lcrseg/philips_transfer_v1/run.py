import copy
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .model import ARMS, STEPS, setup, rate, losses, update_ema
from .data import provider, subset
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write, append
from ..five_frameworks_v1.checkpoint import atomic_save
from ..siemens_lonly_v1.run import aggregate, dice, compare

GPU = '2'


def snapshot(model, ema, opt, data, step):
    return dict(arm=data.arm, step=step, physical_updates=step,
                model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                reads=[data.l_reads, data.u_reads])


def restore(state, model, ema, opt, data):
    if state['arm'] != data.arm:
        raise ValueError('checkpoint arm mismatch')
    model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    data.l_reads, data.u_reads = state['reads']
    model.zero_grad(set_to_none=True)


def preflight(cfg, out, source, model, ema, opt, data):
    atomic_save(snapshot(model, ema, opt, data, 0), out/'initial.pt')
    state = torch.load(out/'initial.pt', map_location='cpu', weights_only=False)
    x, _, _ = data.labeled(0)
    with torch.no_grad():
        reference = source(x)
        entry_error = float((model(x)-reference).abs().max())
        assert torch.allclose(model(x), reference, atol=2e-5, rtol=2e-5)
        assert torch.allclose(ema(x), reference, atol=2e-5, rtol=2e-5)
        # Nonzero copy exercises effective-weight EMA and merge, without an optimizer call.
        probe = copy.deepcopy(model)
        for name, p in probe.named_parameters():
            if p.requires_grad:
                p.add_(torch.sin(torch.arange(p.numel(), device=p.device).reshape_as(p))*.0001)
        target = model.teacher()
        before = {k:v.clone() for k,v in target.parent.native.state_dict().items()}
        update_ema(target, probe)
        merged = probe.deploy()
        assert torch.allclose(probe(x), merged(x), atol=3e-5, rtol=3e-5)
        for k,v in target.parent.native.state_dict().items():
            if v.is_floating_point() and not k.endswith('grad_update'):
                expected = before[k]*.999+merged.parent.native.state_dict()[k]*.001
                assert torch.allclose(v, expected, atol=1e-7, rtol=1e-6), k
        if probe.sidecar is not None:
            expected = model.sidecar.effective()*.999+probe.sidecar.effective()*.001
            assert torch.allclose(target.sidecar.effective(), expected, atol=1e-7, rtol=1e-6)
        del probe, target, merged, before
    restore(state, model, ema, opt, data)
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    l, u, stats = losses(cfg['arm'], model, data, 0)
    grad_report = {}
    for kind, term in [('L', l)]+([] if cfg['arm']=='cl_lonly' else [('U', u)]):
        grads = torch.autograd.grad(term, [p for _,p in params], retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        grad_report[kind] = {group:sum(float(g.double().square().sum()) for (n,p),g in zip(params,grads)
                                     if (n.startswith('sidecar.') if group=='F5' else n.startswith('parent.'))) ** .5
                            for group in ('parent','F5')}
    (l+u).backward()
    saved_grads = [p.grad.detach().clone() for _,p in params]
    first_loss = float(l+u)
    assert all(p.grad is None for p in source.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if cfg['arm'] != 'dense_fixmatch':
        assert all(n.endswith(('.A','.B')) or n=='sidecar.r' for n,p in params)
    del l,u
    restore(state, model, ema, opt, data)
    l,u,_ = losses(cfg['arm'], model, data, 0)
    (l+u).backward()
    assert float(l+u)==first_loss
    assert all(torch.equal(p.grad,g) for (_,p),g in zip(params,saved_grads))
    assert not opt.state and rate(0)==.03
    # Identical parameters, gradients, empty momentum and LR imply identical next SGD update.
    restore(state, model, ema, opt, data)
    write(out/'PREFLIGHT.json', dict(status='PASS', optimizer_updates=0,
        source_entry_max_abs_error=entry_error, source_frozen=True,
        base_frozen=cfg['arm']!='dense_fixmatch', gradient_norms=grad_report,
        effective_weight_ema=True, F5_effective_ema=True if model.sidecar is not None else None,
        nonzero_merge_identity=True, checkpoint_restore_loss_and_gradient_exact=True,
        next_SGD_inputs_identical=True, trainable_parameters=sum(p.numel() for _,p in params),
        first_loss=first_loss, **stats))


@torch.inference_mode()
def evaluate(model, cfg, domain, role, output):
    access = Accessor(cfg['data'], 'mnms', domain, role)
    if role == 'train_labeled' and domain == 'Philips':
        subset(access, set(json.loads(Path(cfg['split']).read_text())['train']))
    model.eval().requires_grad_(False)
    rows=[]; count=0
    for i,meta in enumerate(access.rows):
        item=access.read(i); x=item['image']; truth=item['label'].numpy()
        pred=np.concatenate([model(x[z:z+2,None].cuda()).argmax(1).cpu().numpy() for z in range(0,len(x),2)])
        row=dict(patient=meta['patient_id'],phase=meta['phase'],dice=dice(pred,truth).tolist())
        append(output,row); rows.append(row); count+=len(x)
    return {**aggregate(rows),'forward_slices':count}, rows


def finalize(cfg,out,model,ema,committed,physical):
    train,_=evaluate(ema,cfg,'Philips','train_labeled',out/'private_train.jsonl')
    target,_=evaluate(ema,cfg,'Philips','val',out/'private_Philips_val.jsonl')
    old,rows= evaluate(ema,cfg,'Siemens','val',out/'private_Siemens_val.jsonl')
    old_rows=[json.loads(line) for line in Path(cfg.get('baseline_val_rows',cfg['source_val_rows'])).read_text().splitlines()]
    base=aggregate(old_rows)
    if not cfg.get('baseline_val_rows'):
        assert abs(base['primary_dice']-.8036740663770262)<1e-12
    result=dict(status='COMPLETE',arm=cfg['arm'],step=committed,physical_updates=physical,
        primary='fixed final EMA',train=train,Philips=target,Siemens=old,
        old_domain_delta_pp=100*(old['primary_dice']-base['primary_dice']),
        paired_old=compare(old_rows,rows),two_domain_mean=(old['primary_dice']+target['primary_dice'])/2,
        target_train_val_gap_pp=100*(train['primary_dice']-target['primary_dice']),
        independent_test_access=False,diagnostic_holdout_excluded=cfg.get('diagnostic_holdout_excluded', True))
    if cfg.get('audit_source_training', False):
        result['source_train'],_=evaluate(ema,cfg,'Siemens','train_labeled',out/'private_Siemens_train.jsonl')
    write(out/'RESULTS.json',result)
    if cfg['arm']=='cl_fixmatch':
        write(out/'STATUS.json',dict(status='DIAGNOSING',physical_updates=physical))
        from .diagnostic import run
        run(cfg,model,out)


def set_learning_rate(opt,step):
    for group in opt.param_groups: group["lr"]=rate(step)


def main(cfg):
    if cfg['arm'] not in ARMS or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise PermissionError('frozen arm/GPU mismatch')
    out=Path(cfg['output']); out.mkdir(parents=True,exist_ok=False)
    physical=0; committed=0; started=time.time()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try:
        source, model, ema, opt=setup(cfg)
        data=provider(cfg); data.arm=cfg['arm']
        preflight(cfg,out,source,model,ema,opt,data)
        del source
        if cfg.get('preflight_only'):
            write(out/'STATUS.json',dict(status='PREFLIGHT_PASS',physical_updates=0)); return
        start_step=0
        if cfg.get('resume_checkpoint'):
            state=torch.load(cfg['resume_checkpoint'],map_location='cpu',weights_only=False)
            if not (state['step']==state['physical_updates']==cfg['resume_step'] and 0<state['step']<STEPS):
                raise PermissionError('resume endpoint mismatch')
            restore(state,model,ema,opt,data)
            physical=committed=start_step=state['step']
            atomic_save(snapshot(model,ema,opt,data,start_step),out/'entry.pt')
        def guard(opt,args,kwargs):
            if physical >= STEPS: raise PermissionError('physical cap exhausted')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite gradient')
        def record(opt,args,kwargs):
            nonlocal physical
            physical+=1
            append(out/'physical_updates.jsonl',dict(physical_call=physical,iteration=committed+1,optimizer_executed=True))
        opt.register_step_pre_hook(guard); opt.register_step_post_hook(record)
        accepted=np.zeros(4,dtype=np.int64); total_pixels=0
        train_start=time.time()
        for step in range(start_step,STEPS):
            opt.zero_grad(set_to_none=True)
            set_learning_rate(opt,step)
            l,u,stats=losses(cfg['arm'],model,data,step)
            loss=l+u
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            log = step==start_step or (step+1)%100==0
            before=[p.detach().clone() for p in model.parameters() if p.requires_grad] if log else []
            opt.step(); update_ema(ema,model); committed=step+1
            if cfg.get('checkpoint_each_step',False):
                state=snapshot(model,ema,opt,data,committed)
                if any(not torch.isfinite(v).all() for values in [state['model'],state['ema']] for v in values.values() if torch.is_tensor(v)):
                    raise FloatingPointError('nonfinite post-update model/EMA')
                if any(not torch.isfinite(v).all() for values in state['optimizer']['state'].values() for v in values.values() if torch.is_tensor(v)):
                    raise FloatingPointError('nonfinite optimizer state')
                atomic_save(state,out/'latest.pt')
            accepted+=stats['accepted_class_pixels']; total_pixels+=stats['U_pixels']
            if log:
                norm=sum(float((p.detach()-b).double().square().sum()) for p,b in zip(
                         [p for p in model.parameters() if p.requires_grad],before))**.5
                status=dict(status='RUNNING',arm=cfg['arm'],step=committed,physical_updates=physical,
                    elapsed_seconds=time.time()-train_start,actual_parameter_update_norm=norm,
                    L_reads=data.l_reads,U_reads=data.u_reads,lr=rate(step),**stats)
                append(out/'telemetry.jsonl',status); write(out/'STATUS.json',status); print(json.dumps(status),flush=True)
            if committed%2000==0:
                atomic_save(snapshot(model,ema,opt,data,committed),out/'latest.pt')
        assert committed==physical==STEPS
        atomic_save(snapshot(model,ema,opt,data,committed),out/'final.pt')
        write(out/'COSTS.json',dict(physical_optimizer_updates=physical,engineering_optimizer_updates=0,
            prior_optimizer_updates=start_step,new_optimizer_updates=physical-start_step,
            L_batch_reads=data.l_reads,U_batch_reads=data.u_reads,L_slice_exposure=2*data.l_reads,
            U_slice_exposure=2*data.u_reads,accepted_class_pixels=accepted.tolist(),U_pixels=total_pixels,
            training_seconds=time.time()-train_start,preflight_and_training_seconds=time.time()-started))
        write(out/'STATUS.json',dict(status='EVALUATING',step=committed,physical_updates=physical))
        finalize(cfg,out,model,ema,committed,physical)
        write(out/'STATUS.json',dict(status='COMPLETE',step=committed,physical_updates=physical))
        write(out/'EXIT.json',dict(exit_code=0))
    except BaseException as exc:
        if cfg.get('save_exception_state',False) and 'data' in locals():
            try:
                state=snapshot(model,ema,opt,data,committed)
                state.update(resumable=False,reason='exception forward/BN/RNG may differ from last committed checkpoint')
                atomic_save(state,out/'exception_state.pt')
            except BaseException as save_error:
                write(out/'EXCEPTION_SAVE_FAILURE.json',dict(error=str(save_error)))
        failure=dict(status='FAILED',step=committed,physical_updates=physical,error=str(exc),traceback=traceback.format_exc())
        write(out/'FAILURE.json',failure); write(out/'STATUS.json',failure); write(out/'EXIT.json',dict(exit_code=1)); raise


if __name__=='__main__': main(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
