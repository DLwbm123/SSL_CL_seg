"""Common fixed-matrix loop. Native LCTX/Adam/EMA math is called by core.step."""
import argparse,hashlib,json,os,time
from collections import Counter
from pathlib import Path
import torch
from experiments.lcrseg.ams_seq_transfer_v0_1.engine import save
from experiments.lcrseg.cist_plasticity_v0_2.engine import rng_state,restore_rng
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations,flush
from . import core as c,contract as ct

def initial(base,task,reference,device,fixture=None):
    if fixture is None:
        inputs=ct.read(Path(base)/'private_inputs.json');entry=ct.read(Path(base)/'SOURCE_BINDING.json')['sources'][task['source_task_id']]
        payload=torch.load(Path(inputs['parent'])/'tasks'/task['source_task_id']/'deploy_student.pt',map_location=device,weights_only=False)
        if payload['source']!=ct.PARENT_SOURCE or payload['task']['task_id']!=task['source_task_id'] or not payload['complete'] or payload['head_mode']!=c.LINEAR:raise PermissionError('source identity')
        if c.hash_state(payload['student'])!=entry['student_hash']:raise PermissionError('source student hash')
        model=c.from_state(reference,device,payload.pop('student'),c.LINEAR);del payload
    else:model=c.parent.build(reference,device,task['seed'],task['source_domain'],'SUP_CE')
    source_hash=c.state_hash(model);c.configure(model,'F_CONV',None);ema=c.ema_from(model);opt=c.optimizer_for(model)
    boundary=dict(source_student_hash=source_hash,initial_EMA_hash=c.state_hash(ema),fixed_state_hash=c.hash_state(c.fixed_state(model)),
                  trainable_names=list(c.LAYERS),trainable_parameters=sum(p.numel() for p in c.parameters(model)),
                  whitelist=[dict(name=n,shape=list(p.shape),scalars=p.numel()) for n,p in model.named_parameters() if p.requires_grad],
                  source_optimizer_loaded=False,source_EMA_loaded=False,source_RNG_loaded=False,optimizer_state_empty=not opt.state)
    assert boundary['trainable_parameters']==438192 and boundary['initial_EMA_hash']==source_hash
    model.train();ema.eval();return model,ema,opt,boundary

@torch.no_grad()
def diagnose(model,ema,boundary,row):
    checks=dict(row['checks']);checks.update(fixed_subset=c.hash_state(c.fixed_state(model))==boundary['fixed_state_hash'],
                finite_models=all(bool(torch.isfinite(p).all()) for m in (model,ema) for p in m.parameters()),
                fixed_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),EMA_gradients_absent=all(p.grad is None for p in ema.parameters()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks)

def pending(root,model,ema,opt,p,counts,get_batches,recover=False):
    if p['state']!='POST_CORRECTION_PENDING_DIAGNOSTICS':return
    report=diagnose(model,ema,p['boundary'],p['row'])
    if p['transient']:
        # On recovery only, reopen the exact current batch using stored indices and keyed geometry.
        l,probe=get_batches();before=(c.state_hash(model),c.state_hash(ema),c.optimizer_hash(opt));rng=rng_state(next(model.parameters()).device)
        actual=c.actual_diagnostic(model,l,probe,p['task'],p['epoch'],p['index'],counts,p['transient'],p['row'])
        assert before==(c.state_hash(model),c.state_hash(ema),c.optimizer_hash(opt))
        after=rng_state(next(model.parameters()).device)
        assert torch.equal(rng['cpu'],after['cpu']) and (rng['cuda'] is None or torch.equal(rng['cuda'],after['cuda']))
        actual.update(position=p['position'],epoch=p['epoch'],index=p['index'])
        path=root/('actual_'+str(p['position'])+'.json')
        if not path.exists():c.write_json(path,actual)
    report.update(position=p['position'],source=p['source'],task_id=p['task']['task_id'])
    # Save evidence before raising. Failures preserve the exact pending checkpoint and row.
    if report['status']!='PASS':
        c.write_json(root/'FAILURE_STATE_EVIDENCE.json',dict(**report,row=p['row']));raise FloatingPointError('corrected pending diagnostics failed')
    if p['index']==p['steps']-1 and p['epoch'] in (20,100):c.write_json(root/('diagnostics_pass_'+str(p['position'])+'.json'),report)
    path=root/'steps.jsonl';n=(sum(1 for _ in path.open()) if path.exists() else 0) if recover else p['position']-1
    if n==p['position']-1:c.append(path,p['row'])
    elif n!=p['position']:raise RuntimeError('step log/checkpoint disagreement')

def train(base,task_id,data,reference,device,*,fixture=None,resume=False,stop_after=None):
    source=ct.verify() if fixture is None else 'SYNTHETIC';b=Path(base);task=ct.admit(b,task_id,source) if fixture is None else fixture['task']
    if fixture is None:
        if str(Path(data).resolve())!=ct.read(b/'private_inputs.json')['data'] or stop_after is not None:raise PermissionError('formal data/budget override')
        epochs=list(range(1,101));steps=task['steps_per_epoch'];expected=(c.MANIFEST_SHA,c.SPLIT_SHA);shape=(384,384)
    else:
        if not (Path(data)/'SYNTHETIC_FIXTURE.json').exists():raise PermissionError('fixture required')
        epochs=fixture['epochs'];steps=fixture['steps'];expected=fixture['expected'];shape=(24,24)
    root=b/'tasks'/task_id
    if (root/'receipt.json').exists():raise FileExistsError('completed task protected')
    root.mkdir(parents=True,exist_ok=resume);total=steps*len(epochs);position=0;counts=Counter();prior_l=prior_u=0;order=hashlib.sha256();uorder=hashlib.sha256();u=None
    if fixture is None:assert total==task['updates']
    if device.type=='cuda':torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    with c.training_access(task['domain']):
        ds=c.DomainData(data,task['domain'],'train_labeled',expected=expected,shape=shape);patients=c.patients_for(data,task['domain'],expected)
        model,ema,opt,boundary=initial(base,task,reference,device,fixture)
        def batches(epoch,index,ii,ui):
            nonlocal u
            l=c.batch(ds,ii,device,task['seed'],epoch,index,'labeled');probe=None
            if c.strength(epoch):
                if task['arm']=='DPR_L':probe={k:l[k] for k in ('image','geometry')}
                else:
                    if u is None:u=c.DomainData(data,task['domain'],'train_unlabeled',expected=expected,shape=shape)
                    probe=c.batch(u,ui,device,task['seed'],epoch,index,'unlabeled')
            return l,probe
        if resume:
            path=root/'latest.pt' if (root/'latest.pt').exists() else root/'pre_step.pt';p=torch.load(path,map_location=device,weights_only=False)
            if p['source']!=source or p['task']!=task or p['total']!=total or p['boundary']!=boundary:raise PermissionError('resume identity')
            if p['state'] not in ('PRE_UPDATE_RECOVERABLE','POST_CORRECTION_PENDING_DIAGNOSTICS','CORRECTED_COMMITTED'):raise PermissionError('raw proposal cannot resume')
            model.load_state_dict(p['student']);ema.load_state_dict(p['EMA']);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'],device)
            position=p['position'];counts=Counter(p['counts']);prior_l=p['L_opens'];prior_u=p['U_opens']
            pending(root,model,ema,opt,p,counts,lambda:batches(p['epoch'],p['index'],p['L_indices'],p['U_indices']),recover=True)
            del p
        else:c.write_json(root/'boundary.json',dict(status='PASS',source=source,**boundary))
        for k in range(position):
            ei,index=divmod(k,steps);epoch=epochs[ei];ii=c.orders(len(ds),task['seed'],task['domain'],epoch,'labeled_order',steps)[index];order.update(json.dumps([epoch,index,ii]).encode())
            if c.strength(epoch) and task['arm']!='DPR_L':
                n=c.COUNTS[task['domain']][1] if fixture is None else fixture['U_count'];ui=c.orders(n,task['seed'],task['domain'],epoch,'unlabeled_order',steps)[index];uorder.update(json.dumps([epoch,index,ui]).encode())
        started=time.time()
        def payload(state,pos,epoch,index,ii,ui,row=None,transient=None):
            return dict(source=source,task=task,total=total,steps=steps,position=pos,epoch=epoch,index=index,state=state,
                        student=model.state_dict(),EMA=ema.state_dict(),optimizer=opt.state_dict(),rng=rng_state(device),boundary=boundary,
                        counts=dict(counts),L_opens=prior_l+ds.opens,U_opens=prior_u+(u.opens if u else 0),label_order_hash=order.hexdigest(),U_order_hash=uorder.hexdigest(),
                        L_indices=ii,U_indices=ui,row=row,transient=transient or {},immutable_source_reference=task['source_task_id'])
        for ei,epoch in enumerate(epochs):
            if (ei+1)*steps<=position:continue
            lis=c.orders(len(ds),task['seed'],task['domain'],epoch,'labeled_order',steps);uis=[[]]*steps
            if c.strength(epoch) and task['arm']!='DPR_L':
                if u is None:u=c.DomainData(data,task['domain'],'train_unlabeled',expected=expected,shape=shape)
                uis=c.orders(len(u),task['seed'],task['domain'],epoch,'unlabeled_order',steps)
            for index,ii in enumerate(lis):
                k=ei*steps+index
                if k<position:continue
                ui=uis[index]
                for group in opt.param_groups:group['lr']=.001*(1-k/total)**.9
                save(root/'pre_step.pt',payload('PRE_UPDATE_RECOVERABLE',position,epoch,index,ii,ui))
                l,probe=batches(epoch,index,ii,ui)
                actual=(k+1 in task.get('actual_diagnostic_positions',[]))
                row,transient=c.step(model,ema,opt,l,probe,task,epoch,index,counts,[patients[i] for i in ii],diagnostic=actual)
                position=k+1;order.update(json.dumps([epoch,index,ii]).encode())
                if ui:uorder.update(json.dumps([epoch,index,ui]).encode())
                row.update(epoch=epoch,index=index,position=position,lr=opt.param_groups[0]['lr'])
                p=payload('POST_CORRECTION_PENDING_DIAGNOSTICS',position,epoch,index,ii,ui,row,transient);save(root/'latest.pt',p)
                # All potentially failing acceptance diagnostics follow the durable corrected checkpoint.
                pending(root,model,ema,opt,p,counts,lambda:(l,probe))
                p=payload('CORRECTED_COMMITTED',position,epoch,index,ii,ui,row);save(root/'latest.pt',p);del p,transient,l,probe
                flush('corrected_step_committed')
                if index==steps-1 and epoch in (20,100):
                    exact=root/('checkpoint_'+str(position)+'.pt')
                    if not exact.exists():os.link(root/'latest.pt',exact)
                if position==stop_after:return dict(status='SYNTHETIC_INTERRUPTED',updates=position)
            print(json.dumps(dict(task_id=task_id,epoch=epoch,updates=position)),flush=True)
        assert counts['optimizer_steps']==counts['backward']==counts['ema_updates']==total and c.hash_state(c.fixed_state(model))==boundary['fixed_state_hash']
        student_hash=c.state_hash(model);save(root/'deploy_student.pt',dict(student=model.state_dict(),student_hash=student_hash,task=task,source=source,head_mode=c.LINEAR,complete=True))
        P=438192;memory=dict(full_models=audit_live_models(2),student_parameter_bytes=c.tensor_bytes(dict(model.named_parameters())),EMA_bytes=c.tensor_bytes(dict(ema.named_parameters())),
               trainable_parameters=P,trainable_bytes=P*4,frozen_parameter_bytes=c.tensor_bytes(c.fixed_state(model)),Adam_bytes=c.tensor_bytes(opt.state_dict()),gradient_bytes=P*4,
               active_response_R_FP32_bytes=2*P*4,FP64_J_bytes=2*P*8,FP64_A_bytes=2*P*8,FP64_vector_bytes_each=P*8,
               proposal_snapshot_FP32_bytes_each=P*4,solver_and_reduction_workspace='additional to parameter/EMA/Adam snapshots; allocator peak includes actual tensor temporaries and activations',
               parameter_square_matrix_allocated=False,persistent_response_history_bytes=0,temporary_diagnostic_CPU_snapshot_bytes=2*P*4,
               peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        result=dict(status='TRAINING_COMPLETE',source=source,task=task,updates=position,student_hash=student_hash,EMA_hash=c.state_hash(ema),
             optimizer_hash=c.optimizer_hash(opt),label_order_hash=order.hexdigest(),U_order_hash=uorder.hexdigest(),counts=dict(counts),boundary=boundary,memory=memory,
             L_opens=prior_l+ds.opens,U_opens=prior_u+(u.opens if u else 0),fixed_subset_unchanged=True,EMA_updates=counts['ema_updates'],pseudo_labels=0,completed_unix=time.time(),seconds=time.time()-started)
        c.write_json(root/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','task_id','data','reference'):p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id'];op=Operations(root.parent/(root.name+'_operations'))
    if a['resume']:op.counts=Counter(ct.read(op.root/'operation_counts.json')['counts'])
    try:
        with op:train(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),recovery='corrected pending first; raw proposal never continued'));raise
