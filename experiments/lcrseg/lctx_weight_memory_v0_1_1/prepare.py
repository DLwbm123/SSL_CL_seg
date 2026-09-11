"""No-GT frozen-input binding and zero-update review of eighteen old successes."""
import gc
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as c,engine as e
from experiments.lcrseg.di_dmpa_gate1.binding import sha256,check_hash
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
from . import contract as ct,diagnostics as d

def prepare(base,old,reference,device):
    source=ct.verify();b=ct.nas(base);old=Path(old);b.mkdir(exist_ok=True)
    if (b/'RECOVERY_MANIFEST.json').exists():raise FileExistsError('create-only recovery')
    reservation=ct.read(old/'reservation.json');assert reservation['source']==ct.OLD_SOURCE
    for n,h in reservation['input_hashes'].items():check_hash(old/n,h)
    for n in ['SOURCE_BINDING_AND_LAYER_MANIFEST.json','private_inputs.json']:
        shutil.copyfile(old/n,b/n)
    (b/'bases').symlink_to(old/'bases',target_is_directory=True)
    reused=[];new=[];rows=[]
    binding=ct.read(old/'SOURCE_BINDING_AND_LAYER_MANIFEST.json')
    for task in ct.protocol()['tasks']:
        root=old/'tasks'/task['task_id']
        if (root/'receipt.json').exists():reused.append(task['task_id'])
        else:new.append(task['task_id'])
    assert len(reused)==len(new)==18 and 'O1_s62_LR_SRC_AB' in new
    assert sum(t['updates'] for t in ct.protocol()['tasks'] if t['task_id'] in new)==49900
    (b/'reaudit').mkdir()
    # Runtime W0+D is audited on the same CUDA backend. No model forward or GT is allowed.
    if device.type=='cuda':torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    from experiments.lcrseg.di_dmpa_jascl.modeling import LCRSegUNet2DJASCL
    with Operations(b/'reaudit_operations',update_cap=0) as op, patch.object(LCRSegUNet2DJASCL,'forward',side_effect=PermissionError('no forward in reaudit')), patch.object(c.CurrentData,'__getitem__',side_effect=PermissionError('no GT/image access in reaudit')):
        for task in ct.protocol()['tasks']:
            tid=task['task_id']
            if tid not in reused:continue
            root=old/'tasks'/tid;r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
            assert r['source']==ev['source']==ct.OLD_SOURCE and r['task']==task and r['updates']==task['updates']
            assert r['status']=='TRAINING_COMPLETE' and ev['status']=='COMPLETE' and r['memory']['models']==2 and ev['models']==1
            for phase in ('train','eval'):
                assert ct.read(old/'logs'/(tid+'_'+phase+'_exit.json'))['exit_code']==0
                oc=ct.read(old/'tasks'/(tid+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json')
                assert oc['status']=='PASS'
                if phase=='train':assert oc['counts']['optimizer_steps']==oc['counts']['backward']==oc['counts']['ema_updates']==task['updates']
            m,t,opt,load,entry,boundary=e.initial(b,task,reference,device);initial_frozen=e.frozen_hash(m)
            p=torch.load(root/'latest.pt',map_location=device,weights_only=False)
            assert p['source']==ct.OLD_SOURCE and p['task']==task and p['position']==p['total']==task['updates'] and p['boundary']==boundary==r['boundary']
            m.load_state_dict(p['student']);t.load_state_dict(p['EMA']);opt.load_state_dict(p['optimizer'])
            assert e.frozen_hash(m)==initial_frozen and r['frozen_unchanged'] and r['U_opens']==0 and r['merge_max_abs']<=1e-5
            assert c.state_hash(t)==r['EMA_hash'] and c.optimizer_hash(opt)==r['optimizer_hash']
            effective=c.hash_state(dict(c.dense_items(m)))
            deploy=torch.load(root/'deploy_student.pt',map_location='cpu',weights_only=False)
            assert c.hash_state(deploy['student'])==deploy['student_hash']==r['student_hash']==ev['student_hash']==effective
            assert deploy['source']==ct.OLD_SOURCE and deploy['complete'] and deploy['head_mode']==c.LINEAR and deploy['task']==task
            steps=(root/'steps.jsonl').read_text().splitlines();assert len(steps)==task['updates']
            import hashlib
            order=hashlib.sha256()
            for k,line in enumerate(steps):
                row=json.loads(line);epoch,index=divmod(k,task['steps_per_epoch']);assert row['update']==k+1
                ii=c.orders(c.COUNTS[task['domain']][0],task['seed'],task['domain'],epoch+1,'labeled_order',task['steps_per_epoch'])[index]
                order.update(json.dumps([epoch+1,index,ii]).encode())
            assert order.hexdigest()==r['label_order_hash']
            diag=d.collect(m,load,100,task['arm'])
            evidence=dict(task_id=tid,status='PASS' if all(z['passed'] for z in diag) else 'FAIL',training_source_commit=ct.OLD_SOURCE,
                          reaudit_source_commit=source,checkpoint_sha256=sha256(root/'latest.pt'),deploy_sha256=sha256(root/'deploy_student.pt'),
                          receipt_sha256=sha256(root/'receipt.json'),student_hash=effective,frozen_hash=initial_frozen,
                          reused_evaluation_sha256=sha256(root/'evaluation/private_patient_metrics.csv'),rows=diag,
                          coverage='final unmerged checkpoint and deployment; unavailable historical epoch20 checkpoints not reconstructed',
                          GT_reads=0,model_forwards=0,optimizer_updates=0)
            c.write_json(b/'reaudit'/(tid+'.json'),evidence);d.require(diag);rows.append(evidence)
            del m,t,opt,p,deploy,load;gc.collect()
    assert op.counts['optimizer_steps']==op.counts['model_forward']==op.counts['sample_access']==op.counts['hdf5_open']==0
    manifest=dict(status='PASS',source=source,attempt_id=b.name,old_run=str(old),old_training_source_commit=ct.OLD_SOURCE,
                  reused_task_ids=reused,new_task_ids=new,rerun_task_id='O1_s62_LR_SRC_AB',
                  old_success_updates=45500,old_failed_updates=420,new_updates=49900,scientific_updates=95400,physical_updates_if_complete=95820,
                  protocol_hash=sha256(ct.OLD_DOC/'PROTOCOL.json'),binding_hash=sha256(b/'SOURCE_BINDING_AND_LAYER_MANIFEST.json'),
                  original399_sha256=sha256(old/'tasks/O1_s62_LR_SRC_AB/latest.pt'),original420_exact_state_available=False,
                  reaudit_counts=dict(op.counts),basis_recomputed=False,source_retrained=False,
                  original_terminal=ct.read(old/'TERMINAL.json'))
    c.write_json(b/'RECOVERY_MANIFEST.json',manifest)
    print(json.dumps(dict(status='PASS',reaudited=len(reused),new_tasks=len(new),new_updates=49900)),flush=True)
