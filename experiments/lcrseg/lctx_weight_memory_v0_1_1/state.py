"""Exact post-update evidence; pending means recheck before another update."""
import hashlib
import json
import os
from pathlib import Path
import torch
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c
from experiments.lcrseg.di_dmpa_gate1.binding import sha256
from experiments.lcrseg.ams_seq_transfer_v0_1.engine import save
from . import diagnostics as d

def value_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():visit(k);visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif isinstance(x,c.np.ndarray):h.update(x.tobytes())
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def metadata(base,task,source,fixture=None):
    from .contract import read, DOC
    if fixture is not None:return dict(attempt_id=task['task_id'],config_hash='SYNTHETIC',source_identity='SYNTHETIC',basis_identity='SYNTHETIC')
    inputs=read(Path(base)/'RECOVERY_MANIFEST.json')
    return dict(attempt_id=inputs['attempt_id']+'/'+task['task_id'], config_hash=sha256(DOC/'EXECUTION_PLAN.md'),
                protocol_hash=inputs['protocol_hash'], source_identity=task['source_task_id'],
                basis_identity=inputs['binding_hash'], training_source_commit=source)

def checkpoint(root,payload,model,load,epoch,index,order_hash,frozen,identity,pending):
    from experiments.lcrseg.ssl_anchored_mix_v0_1 import telemetry
    payload.update(identity, epoch=epoch,index=index,label_order_hash=order_hash,frozen_hash=frozen,
                   checkpoint_state='POST_UPDATE_PENDING_DIAGNOSTICS' if pending else 'POST_UPDATE_COMMITTED',
                   operation_counts={} if telemetry.ACTIVE is None else dict(telemetry.ACTIVE.counts),
                   last_log_update=payload['position'], completed_phase='optimizer_and_EMA', backend=d.backend())
    path=Path(root)/'latest.pt';save(path,payload)
    special=payload['task']['task_id']=='O1_s62_LR_SRC_AB' and payload['position'] in (399,420)
    if special:
        exact=Path(root)/('checkpoint_'+str(payload['position'])+'.pt')
        if not exact.exists():os.link(path,exact)
    rows=[]
    if pending:rows=check_pending(root,model,load,payload)
    return rows

def check_pending(root,model,load,payload):
    if payload.get('checkpoint_state')!='POST_UPDATE_PENDING_DIAGNOSTICS':return []
    from experiments.lcrseg.lctx_weight_memory_v0_1.engine import frozen_hash
    root=Path(root);position=payload['position'];checkpoint=root/'latest.pt'
    checksum=sha256(checkpoint)
    rows=d.collect(model,load,payload['epoch'],payload['task']['arm'])
    frozen_ok=frozen_hash(model)==payload['frozen_hash']
    if not frozen_ok:
        for row in rows:row['failures'].append('frozen_state');row['passed']=False;row['status']='FAIL'
    evidence=dict(task_id=payload['task']['task_id'],attempt_id=payload['attempt_id'],
                  position=position,epoch=payload['epoch'],index=payload['index'],
                  phase='diagnostics_after_completed_optimizer_and_EMA',last_log_update=payload['last_log_update'],
                  checkpoint=str(checkpoint),checkpoint_sha256=checksum,frozen_pass=frozen_ok,
                  completed_operation_counts=payload['operation_counts'],rows=rows,
                  status='PASS' if all(r['passed'] for r in rows) else 'FAIL')
    c.write_json(root/('diagnostics_'+str(position)+'.json'),evidence)
    if evidence['status']=='FAIL':
        c.write_json(root/'FAILURE_STATE_EVIDENCE.json',evidence)
        d.require(rows)
    c.write_json(root/('diagnostics_pass_'+str(position)+'.json'),dict(status='PASS',position=position,checkpoint_sha256=checksum))
    return rows

def compare_399(base,root,payload):
    from .contract import read
    b=Path(base);old=Path(read(b/'RECOVERY_MANIFEST.json')['old_run'])/'tasks'/payload['task']['task_id']/'latest.pt'
    original=torch.load(old,map_location='cpu',weights_only=False)
    exact={k:value_hash(original[k])==value_hash(payload[k]) for k in
           ['student','EMA','optimizer','counts','boundary','L_opens','position','total','task']}
    rng={k:value_hash(original['rng'][k])==value_hash(payload['rng'][k]) for k in original['rng']}
    old_lines=(old.parent/'steps.jsonl').read_text().splitlines()[:399]
    new_lines=(Path(root)/'steps.jsonl').read_text().splitlines()
    exact['steps_and_losses']=old_lines==new_lines
    # Keyed streams restore the caller's process-global RNG. Across cold processes that
    # ambient state is not seeded by the protocol and is not a comparable stream state.
    evidence=dict(status='PASS' if all(exact.values()) else 'FAIL',position=399,exact=exact,
                  ambient_rng_equal=rng, comparable_RNG='same frozen keyed seeds/orders; all 399 step records and complete train states compared',
                  ambient_RNG_scope='unseeded process-global state restored by keyed fork_rng; raw equality recorded, not presumed',
                  old_checkpoint_sha256=sha256(old),new_checkpoint_sha256=sha256(Path(root)/'checkpoint_399.pt'),
                  old_training_source=original['source'],new_training_source=payload['source'],
                  original_420_state_available=False)
    c.write_json(Path(root)/'PREFIX_399_COMPARISON.json',evidence)
    if evidence['status']!='PASS':raise RuntimeError('399 prefix mismatch; preserve evidence and investigate before further training')

def gate_420(base,root):
    from .contract import read
    root=Path(root);p=read(root/'diagnostics_pass_420.json');prefix=read(root/'PREFIX_399_COMPARISON.json')
    assert p['status']==prefix['status']=='PASS'
    assert sha256(root/'checkpoint_420.pt')==p['checkpoint_sha256']
    c.write_json(Path(base)/'RECOVERY_420_GATE.json',dict(status='PASS',task_id='O1_s62_LR_SRC_AB',
                 prefix=prefix,checkpoint_sha256=p['checkpoint_sha256'],position=420,
                 scope='exact new attempt state; original unsaved 420 state was not restored'))

def incomplete_update(root,m,t,opt,task,source,epoch,index,position,counts,identity,exc):
    from experiments.lcrseg.ssl_anchored_mix_v0_1 import telemetry
    oc={} if telemetry.ACTIVE is None else dict(telemetry.ACTIVE.counts)
    phase='optimizer' if oc.get('optimizer_steps_attempts',0)>oc.get('optimizer_steps',0) else 'EMA' if oc.get('optimizer_steps',0)>oc.get('ema_updates',0) else 'forward_or_backward'
    path=Path(root)/('incomplete_update_'+str(position+1)+'.pt')
    save(path,dict(student=m.state_dict(),EMA=t.state_dict(),optimizer=opt.state_dict(),task=task,source=source,
                   epoch=epoch,index=index,last_committed_position=position,counts=dict(counts),operation_counts=oc,
                   checkpoint_state='INCOMPLETE_UPDATE_NOT_RESUMABLE',identity=identity,phase=phase,
                   rng=dict(python=c.random.getstate(),numpy=c.np.random.get_state(),cpu=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state() if next(m.parameters()).device.type=='cuda' else None)))
    c.write_json(Path(root)/'FAILURE_STATE_EVIDENCE.json',dict(task_id=task['task_id'],attempt_id=identity['attempt_id'],
                 epoch=epoch,index=index,last_committed_position=position,phase=phase,error=repr(exc),
                 checkpoint=str(path),checkpoint_sha256=sha256(path),operation_counts=oc,automatic_resume=False))
