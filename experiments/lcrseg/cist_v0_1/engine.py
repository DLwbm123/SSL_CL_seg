"""One controller-only optimizer loop for all six arms; exact recovery references immutable core."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
import torch
from experiments.lcrseg.ams_seq_transfer_v0_1.engine import save
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations,flush
from . import core as c,contract as ct

def rng_state(device):
    return dict(python=c.random.getstate(),numpy=c.np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)

def restore_rng(r,device):
    c.random.setstate(r['python']);c.np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu())
    if device.type=='cuda':torch.cuda.set_rng_state(r['cuda'].cpu(),device)

def initial(base,task,reference,device,fixture=None):
    if fixture is None:
        tensors,entry=ct.load_basis(base,task);inputs=ct.read(Path(base)/'private_inputs.json')
        p=torch.load(Path(inputs['parent'])/'tasks'/task['source_task_id']/'deploy_student.pt',map_location=device,weights_only=False)
        if p['source']!=ct.PARENT_SOURCE or p['task']['task_id']!=task['source_task_id'] or not p['complete'] or p['head_mode']!=c.LINEAR:raise PermissionError('source payload identity')
        if c.hash_state(p['student'])!=entry['student_hash']:raise PermissionError('source hash mismatch')
        core=c.from_state(reference,device,p.pop('student'),c.LINEAR);del p
    else:
        core=c.parent.build(reference,device,task['seed'],task['source_domain'],'SUP_CE')
        tensors,meta=c.readout_basis(core.decoder.conv_logit.mu.weight,task['seed'],task['source_domain']);entry=dict(student_hash=c.state_hash(core),basis=meta)
    m=c.Student(core,c.Transport(tensors,task['arm'],task['seed'],task['source_domain']).to(device));opt=c.optimizer_for(m.transport)
    boundary=dict(source_student_hash=c.state_hash(core),controller_initial_hash=c.state_hash(m.transport),
                  basis_buffer_hash=c.hash_state(dict(m.transport.named_buffers())),optimizer_state_empty=not opt.state,
                  source_optimizer_loaded=False,source_RNG_loaded=False,EMA_models=0,full_models=audit_live_models(1))
    assert boundary['source_student_hash']==entry['student_hash'] and boundary['optimizer_state_empty']
    if task['arm']=='SCALE_LU':assert torch.count_nonzero(m.transport.scale)==torch.count_nonzero(m.transport.bias)==0
    else:assert torch.count_nonzero(m.transport.mlp[-1].weight)==torch.count_nonzero(m.transport.mlp[-1].bias)==0
    return m,opt,entry,boundary

@torch.no_grad()
def diagnose(model,boundary):
    device=next(model.transport.parameters()).device
    with c.rng(device,'CIST_V0_1','fixed_geometric_probe'):
        h=torch.randn(2,16,8,8,device=device);hp,detail=model.transport(h);row=c.mechanism(h,hp,detail)
    checks=dict(core_hash=c.state_hash(model.core)==boundary['source_student_hash'],
                basis_hash=c.hash_state(dict(model.transport.named_buffers()))==boundary['basis_buffer_hash'],
                finite=all(bool(torch.isfinite(p).all()) for p in model.transport.parameters()),
                core_gradients_absent=all(p.grad is None for p in model.core.parameters()))
    if model.transport.arm.startswith('ISO_'):
        checks.update(orthogonality=row['orthogonality64_max']<=c.ISO64_LIMIT,distance=row['distance_relative_max']<=c.DISTANCE32_LIMIT)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,probe=row,scope='fixed synthetic features; no image/GT forward')

def pending(root,model,payload):
    if payload['state']!='POST_UPDATE_PENDING_DIAGNOSTICS':return None
    path=Path(root)/'latest.pt';checksum=ct.sha256(path)
    try:report=diagnose(model,payload['boundary'])
    except Exception as exc:report=dict(status='FAIL',error=repr(exc))
    report.update(position=payload['position'],source=payload['source'],task_id=payload['task']['task_id'],checkpoint_sha256=checksum)
    c.write_json(Path(root)/('diagnostics_'+str(payload['position'])+'.json'),report)
    if report['status']!='PASS':
        c.write_json(Path(root)/'FAILURE_STATE_EVIDENCE.json',dict(phase='post_update_diagnostics',**report))
        raise FloatingPointError('post-update diagnostics failed; exact pending state retained')
    c.write_json(Path(root)/('diagnostics_pass_'+str(payload['position'])+'.json'),report)
    return report

def train(base,task_id,data,reference,device,*,fixture=None,resume=False,stop_after=None):
    source=ct.verify() if fixture is None else 'SYNTHETIC';b=Path(base);task=ct.admit(b,task_id,source) if fixture is None else fixture['task']
    if fixture is None:
        if str(Path(data).resolve())!=ct.read(b/'private_inputs.json')['data'] or stop_after is not None:raise PermissionError('formal data or budget override')
        epochs=list(range(1,101));steps=task['steps_per_epoch'];expected=(c.MANIFEST_SHA,c.SPLIT_SHA);shape=(384,384)
    else:
        if not (Path(data)/'SYNTHETIC_FIXTURE.json').exists():raise PermissionError('fixture required')
        epochs=fixture['epochs'];steps=fixture['steps'];expected=fixture['expected'];shape=(24,24)
    root=b/'tasks'/task_id
    if (root/'receipt.json').exists():raise FileExistsError('completed task protected')
    root.mkdir(parents=True,exist_ok=resume);total=steps*len(epochs);position=0;counts=Counter();prior_l=prior_u=0;order=hashlib.sha256();diagnostics=[]
    if fixture is None:assert total==task['updates']
    if device.type=='cuda':torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    with c.training_access(task['domain'],task['arm']):
        ds=c.DomainData(data,task['domain'],'train_labeled',expected=expected,shape=shape);patients=c.patients_for(data,task['domain'],expected)
        m,opt,entry,boundary=initial(base,task,reference,device,fixture);u=None
        if resume:
            path=root/'latest.pt' if (root/'latest.pt').exists() else root/'pre_step.pt';p=torch.load(path,map_location=device,weights_only=False)
            if p['source']!=source or p['task']!=task or p['total']!=total or p['boundary']!=boundary:raise PermissionError('resume identity')
            position=p['position'];m.transport.load_state_dict(p['controller']);opt.load_state_dict(p['optimizer']);restore_rng(p['rng'],device)
            counts=Counter(p['counts']);prior_l=p['L_opens'];prior_u=p['U_opens'];diagnostics=p['diagnostics']
            assert len((root/'steps.jsonl').read_text().splitlines())==position if (root/'steps.jsonl').exists() else position==0
            check=pending(root,m,p)
            if check:diagnostics.append(check)
            del p
        else:
            check=diagnose(m,boundary);c.write_json(root/'initial_diagnostics.json',check)
            if check['status']!='PASS':raise RuntimeError('initial numerical diagnosis')
            c.write_json(root/'boundary.json',dict(source=source,status='PASS',**boundary))
        for k in range(position):
            ei,index=divmod(k,steps);ii=c.orders(len(ds),task['seed'],task['domain'],epochs[ei],'labeled_order',steps)[index];order.update(json.dumps([epochs[ei],index,ii]).encode())
        started=time.time()
        def payload(state,pos,epoch,index):
            return dict(source=source,task=task,total=total,position=pos,epoch=epoch,index=index,state=state,
                        controller=m.transport.state_dict(),optimizer=opt.state_dict(),rng=rng_state(device),boundary=boundary,
                        counts=dict(counts),operation_counts={} if c.telemetry.ACTIVE is None else dict(c.telemetry.ACTIVE.counts),
                        L_opens=prior_l+ds.opens,U_opens=prior_u+(u.opens if u else 0),label_order_hash=order.hexdigest(),diagnostics=diagnostics,
                        immutable_core_reference=task['source_task_id'],config_sha256=ct.sha256(ct.DOC/'PROTOCOL.json') if fixture is None else 'SYNTHETIC')
        for ei,epoch in enumerate(epochs):
            if (ei+1)*steps<=position:continue
            if epoch>20 and task['arm']!='ISO_COND_L' and u is None:u=c.DomainData(data,task['domain'],'train_unlabeled',expected=expected,shape=shape)
            lis=c.orders(len(ds),task['seed'],task['domain'],epoch,'labeled_order',steps)
            uis=c.orders(len(u),task['seed'],task['domain'],epoch,'unlabeled_order',steps) if u is not None else None
            for index,ii in enumerate(lis):
                k=ei*steps+index
                if k<position:continue
                for group in opt.param_groups:group['lr']=.001*(1-k/total)**.9
                save(root/'pre_step.pt',payload('PRE_UPDATE_RECOVERABLE',position,epoch,index));prehash=ct.sha256(root/'pre_step.pt');solve_number=0
                def journal(phase,small):
                    nonlocal solve_number
                    if phase=='PRE_SOLVE':solve_number+=1
                    save(root/'current_solve.pt',dict(phase=phase,solve_number=solve_number,epoch=epoch,index=index,committed_position=position,
                         pre_step_sha256=prehash,small_matrix=small,recovery='replay the entire uncommitted optimizer step from pre_step; no mid-autograd resume'))
                m.transport.journal=journal
                lb=c.batch(ds,ii,device,task['seed'],epoch,index,'labeled')
                ub=c.batch(u,uis[index],device,task['seed'],epoch,index,'unlabeled',augment=False) if u is not None else None
                try:row=c.step(m,opt,lb,ub,task,epoch,index,counts,[patients[i] for i in ii])
                finally:m.transport.journal=None
                position=k+1;order.update(json.dumps([epoch,index,ii]).encode());c.append(root/'steps.jsonl',dict(epoch=epoch,index=index,position=position,lr=opt.param_groups[0]['lr'],**row));flush('step_committed')
                needs_check=(index==steps-1 and (epoch in (20,100) or fixture is not None))
                state='POST_UPDATE_PENDING_DIAGNOSTICS' if needs_check else 'POST_UPDATE_COMMITTED'
                p=payload(state,position,epoch,index);save(root/'latest.pt',p)
                if needs_check:
                    exact=root/('checkpoint_'+str(position)+'.pt')
                    if not exact.exists():os.link(root/'latest.pt',exact)
                    check=pending(root,m,p);diagnostics.append(check)
                del p
                (root/'current_solve.pt').unlink(missing_ok=True)
                if position==stop_after:return dict(status='SYNTHETIC_INTERRUPTED',updates=position)
            print(json.dumps(dict(task_id=task_id,epoch=epoch,updates=position)),flush=True)
        assert counts['optimizer_steps']==total and c.state_hash(m.core)==boundary['source_student_hash']
        assert c.hash_state(dict(m.transport.named_buffers()))==boundary['basis_buffer_hash']
        controller_hash=c.state_hash(m.transport);student_hash=c.state_hash(m)
        save(root/'deploy_student.pt',dict(core=m.core.state_dict(),controller=m.transport.state_dict(),student_hash=student_hash,
             controller_hash=controller_hash,task=task,source=source,source_student_hash=boundary['source_student_hash'],head_mode=c.LINEAR,complete=True))
        memory=dict(full_models=audit_live_models(1),core_parameter_bytes=c.tensor_bytes(dict(m.core.named_parameters())),
                    controller_parameters=sum(p.numel() for p in m.transport.parameters()),controller_parameter_bytes=c.tensor_bytes(dict(m.transport.named_parameters())),
                    basis_buffer_bytes=c.tensor_bytes(dict(m.transport.named_buffers())),Adam_bytes=c.tensor_bytes(opt.state_dict()),
                    gradient_bytes=c.tensor_bytes([p.grad for p in m.transport.parameters() if p.grad is not None]),EMA_bytes=0,
                    peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        result=dict(status='TRAINING_COMPLETE',source=source,task=task,updates=position,student_hash=student_hash,controller_hash=controller_hash,
                    optimizer_hash=c.optimizer_hash(opt),label_order_hash=order.hexdigest(),counts=dict(counts),boundary=boundary,memory=memory,
                    L_opens=prior_l+ds.opens,U_opens=prior_u+(u.opens if u else 0),core_frozen=True,EMA_updates=0,pseudo_labels=0,
                    completed_unix=time.time(),seconds=time.time()-started)
        c.write_json(root/'receipt.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','task_id','data','reference'):p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id'];op=Operations(root.parent/(root.name+'_operations'))
    if a['resume']:op.counts=Counter(ct.read(op.root/'operation_counts.json')['counts'])
    try:
        with op:train(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),latest_checkpoint='latest.pt' if (root/'latest.pt').exists() else 'pre_step.pt',current_solve_present=(root/'current_solve.pt').exists()))
        raise
