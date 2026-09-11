"""One fixed current-L trainer; parent objective is unchanged for every arm."""
import argparse,json,time,hashlib
from collections import Counter
from pathlib import Path
import torch
from . import core as c
from .contract import verify,admit,read,PARENT_SOURCE
from experiments.lcrseg.ams_seq_transfer_v0_1.engine import save
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations,flush
from experiments.lcrseg.single_teacher_scd_v0_1.engine import audit_live_models

def basis_loader(base,task):
    entry=read(Path(base)/'SOURCE_BINDING_AND_LAYER_MANIFEST.json')['sources'][task['source_task_id']]
    def load(name):
        t=torch.load(Path(base)/'bases'/task['source_task_id']/(name+'.pt'),map_location='cpu',weights_only=False)
        expected=next(r['tensor_hashes'] for r in entry['layers'] if r['layer']==name)
        if {k:c.hash_state({k:v}) for k,v in t.items()}!=expected:raise PermissionError('basis identity mismatch')
        return t
    return load,entry

def initial(base,task,reference,device,fixture=None):
    if fixture is None:
        inputs=read(Path(base)/'private_inputs.json');parent=Path(inputs['parent'])/'tasks'/task['source_task_id']
        load,entry=basis_loader(base,task);p=torch.load(parent/'deploy_student.pt',map_location=device,weights_only=False)
        if p['source']!=PARENT_SOURCE or p['task']!=entry['receipt']['task'] or not p['complete'] or p['head_mode']!=c.LINEAR:raise PermissionError('parent identity')
        if c.hash_state(p['student'])!=entry['student_hash']:raise PermissionError('source state hash')
        m=c.from_state(reference,device,p.pop('student'),c.LINEAR);del p
    else:
        m=c.build(reference,device,task['seed'],task['source_domain'],'SUP_CE');basis={n:c.basis(m.get_parameter(n),task['seed'],task['source_domain'],n)[0] for n in c.LAYERS};load=lambda n:basis[n];entry=dict(student_hash=c.state_hash(m))
    teacher=c.ema_from(m);c.configure(m,task['arm'],load);opt=c.optimizer_for(m)
    boundary=dict(source_student_hash=entry['student_hash'],initial_effective_hash=c.hash_state(dict(c.dense_items(m))),initial_EMA_hash=c.state_hash(teacher),optimizer_state_empty=not opt.state,source_EMA_loaded=False,source_Adam_loaded=False,source_RNG_loaded=False)
    assert boundary['source_student_hash']==boundary['initial_effective_hash']==boundary['initial_EMA_hash'] and boundary['optimizer_state_empty']
    return m,teacher,opt,load,entry,boundary

def frozen_hash(model):return c.hash_state({k:v for k,v in model.named_parameters() if not v.requires_grad}|dict(model.named_buffers()))

def train(base,task_id,data,reference,device,*,resume=False,fixture=None,stop_after=None,recovery=False):
    verify_run,admit_run=verify,admit
    if recovery:
        from experiments.lcrseg.lctx_weight_memory_v0_1_1 import contract as rc, state as rs, diagnostics as rd
        verify_run,admit_run=rc.verify,rc.admit
    source=verify_run() if fixture is None else 'SYNTHETIC';b=Path(base);task=admit_run(b,task_id,source) if fixture is None else fixture['task']
    if fixture is None:
        if str(Path(data).resolve())!=read(b/'private_inputs.json')['data']:raise PermissionError('data binding mismatch')
        if stop_after is not None:raise PermissionError('formal shortening forbidden')
        epochs=100;steps=task['steps_per_epoch'];shape=(384,384);expected=(c.MANIFEST_SHA,c.SPLIT_SHA)
    else:
        if not (Path(data)/'SYNTHETIC_FIXTURE.json').exists():raise PermissionError('synthetic fixture only')
        epochs=fixture['epochs'];steps=fixture['steps'];shape=(24,24);expected=fixture['expected']
    root=b/'tasks'/task_id
    if (root/'receipt.json').exists():raise FileExistsError('complete task protected')
    root.mkdir(parents=True,exist_ok=resume);total=steps*epochs;position=0;counts=Counter();priorL=0;diag=[]
    if fixture is None:assert total==task['updates']
    if device.type=='cuda':
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    with c.training_access(task['domain']):
        ds=c.DomainData(data,task['domain'],'train_labeled',expected=expected,shape=shape);patients=c.patients_for(data,task['domain'],expected)
        m,t,opt,load,entry,boundary=initial(base,task,reference,device,fixture);frozen=frozen_hash(m)
        identity=rs.metadata(base,task,source,fixture) if recovery else None
        if resume:
            p=torch.load(root/'latest.pt',map_location=device,weights_only=False)
            if p['task']!=task or p['source']!=source or p['boundary']!=boundary or p['total']!=total:raise PermissionError('resume identity')
            position=p['position'];assert len((root/'steps.jsonl').read_text().splitlines())==position,'uncommitted tail; preserve attempt, do not replay'
            m.load_state_dict(p.pop('student'));t.load_state_dict(p.pop('EMA'));opt.load_state_dict(p.pop('optimizer'));counts=Counter(p['counts']);priorL=p['L_opens'];diag=p['diagnostics'];rng=p['rng']
            c.random.setstate(rng['python']);c.np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'].cpu(),device)
            if frozen_hash(m)!=frozen:raise PermissionError('resume frozen source or basis differs')
            if recovery:
                if any(p.get(k)!=v for k,v in identity.items()):raise PermissionError('recovery resume identity')
                diag+=rs.check_pending(root,m,load,p)
            del p,rng
        else:
            diag=rd.require(rd.collect(m,load,0,task['arm'])) if recovery else c.diagnostics(m,load,0,task['arm']);c.write_json(root/'boundary.json',dict(status='PASS',source=source,**boundary))
        orderhash=hashlib.sha256()
        for k in range(position):
            ep,idx=divmod(k,steps);ii=c.orders(len(ds),task['seed'],task['domain'],ep+1,'labeled_order',steps)[idx];orderhash.update(json.dumps([ep+1,idx,ii]).encode())
        m.train();t.eval();started=time.time()
        for epoch in range(position//steps+1,epochs+1):
            lis=c.orders(len(ds),task['seed'],task['domain'],epoch,'labeled_order',steps)
            for index,ii in enumerate(lis):
                k=(epoch-1)*steps+index
                if k<position:continue
                for g in opt.param_groups:g['lr']=.001*(1-k/total)**.9
                lb=c.batch(ds,ii,device,task['seed'],epoch,index,'labeled')
                try:
                    row=c.step(m,t,opt,lb,task['seed'],task['domain'],epoch,index,counts,[patients[i] for i in ii],task['arm'])
                except BaseException as exc:
                    if recovery:rs.incomplete_update(root,m,t,opt,task,source,epoch,index,position,counts,identity,exc)
                    raise
                assert row['U_image_records']==0 and row['labeled_source_scoring_multiplicity']==1
                position=k+1;orderhash.update(json.dumps([epoch,index,ii]).encode());c.append(root/'steps.jsonl',dict(epoch=epoch,step=index,update=position,lr=opt.param_groups[0]['lr'],**row));flush('step_committed')
                if not recovery and index==steps-1 and epoch in (20,100):
                    assert frozen_hash(m)==frozen,'frozen parameters changed';diag+=c.diagnostics(m,load,epoch,task['arm'])
                if index==steps-1 or position==stop_after:
                    rng=dict(python=c.random.getstate(),numpy=c.np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
                    payload=dict(task=task,source=source,total=total,position=position,student=m.state_dict(),EMA=t.state_dict(),optimizer=opt.state_dict(),counts=dict(counts),boundary=boundary,L_opens=priorL+ds.opens,rng=rng,diagnostics=diag)
                    if recovery:
                        pending=(epoch in (20,100)) or bool(fixture and fixture.get('diagnose_every_checkpoint'))
                        diag+=rs.checkpoint(root,payload,m,load,epoch,index,orderhash.hexdigest(),frozen,identity,pending)
                        if fixture is None and task_id=='O1_s62_LR_SRC_AB':
                            if position==399:rs.compare_399(base,root,payload)
                            if position==420:rs.gate_420(base,root)
                    else:save(root/'latest.pt',payload)
                    del payload
                if position==stop_after:return dict(status='SYNTHETIC_INTERRUPTED',updates=position)
            if epoch==20:c.write_json(root/'warmup.json',dict(student_hash=c.hash_state(dict(c.dense_items(m))),EMA_hash=c.state_hash(t),optimizer_hash=c.optimizer_hash(opt),label_order_hash=orderhash.hexdigest(),U_opens=0))
            print(json.dumps(dict(task_id=task_id,epoch=epoch,updates=position)),flush=True)
        assert counts['optimizer_steps']==total and frozen_hash(m)==frozen
        effective_hash=c.hash_state(dict(c.dense_items(m)));ema_hash=c.state_hash(t);optimizer_hash=c.optimizer_hash(opt)
        if fixture is None and task['arm']=='F_FULL':
            got=dict(student_hash=effective_hash,EMA_hash=ema_hash,optimizer_hash=optimizer_hash,label_order_hash=orderhash.hexdigest())
            if got!=entry['parent_F_FULL_reference']:raise RuntimeError('F_FULL differs from parent trajectory')
        memory=dict(models=audit_live_models(2),frozen_parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters() if not p.requires_grad),trainable_parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters() if p.requires_grad),EMA_bytes=c.tensor_bytes(t.state_dict()),thin_basis_buffer_bytes=c.tensor_bytes(dict(m.named_buffers())),Adam_bytes=c.tensor_bytes(opt.state_dict()),gradient_bytes=c.tensor_bytes([p.grad for p in m.parameters() if p.grad is not None]),activations='included in measured CUDA peaks; no separate allocator attribution',peak_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,peak_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        # Merge in the existing student, never create a third full model.
        m.eval()
        with torch.no_grad(),c.rng(device,'LCTX_WEIGHT_MEMORY_V0_1','merge_probe'):
            x=torch.rand(1,3,24,24,device=device);before=m(x,stochastic_classifier=False)[0];c.merge(m);after=m(x,stochastic_classifier=False)[0];error=float((before-after).abs().max())
        if error>1e-5:raise FloatingPointError('merge output error')
        assert c.state_hash(m)==effective_hash
        save(root/'deploy_student.pt',dict(student=m.state_dict(),student_hash=effective_hash,task=task,source=source,head_mode=c.LINEAR,complete=True))
        receipt=dict(status='TRAINING_COMPLETE',source=source,task=task,updates=total,student_hash=effective_hash,EMA_hash=ema_hash,optimizer_hash=optimizer_hash,label_order_hash=orderhash.hexdigest(),boundary=boundary,counts=dict(counts),L_opens=priorL+ds.opens,U_opens=0,frozen_unchanged=True,memory=memory,merge_max_abs=error,merge_probe_forwards=2,seconds=time.time()-started,completed_unix=time.time())
        c.write_json(root/'PARAMETER_DIAGNOSTICS.json',diag);c.write_json(root/'receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','task_id','data','reference']:p.add_argument('--'+k.replace('_','-'),required=True)
    p.add_argument('--recovery',action='store_true');p.add_argument('--resume',action='store_true');a=vars(p.parse_args());root=Path(a['base'])/'tasks'/a['task_id'];op=Operations(root.parent/(root.name+'_operations'))
    if a['resume']:
        old=read(op.root/'operation_counts.json');op.counts=Counter(old['counts']);c.write_json(op.root/('resume_prior_'+str(time.time_ns())+'.json'),old)
    try:
        with op:train(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc)));raise
