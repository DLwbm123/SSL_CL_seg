"""Frozen SRC_CE loop and common final-pending target loop, with new-only lineage."""
import argparse,time,os
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch
from . import core as c,contract as ct
from experiments.lcrseg.ams_seq_transfer_v0_1 import engine as src
from experiments.lcrseg.dpr_v0_1 import engine as target
from experiments.lcrseg.dpr_finite_v0_2 import engine as finite_engine
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations
save=src.save
rng_state=target.rng_state

def initial(base,task,reference,device,fixture=None):
    if fixture is None:
        source=ct.verify();entry=ct.read(Path(base)/'SOURCE_BINDING.json')['sources'][task['source_task_id']]
        path=Path(base)/'tasks'/task['source_task_id']/'deploy_student.pt'
        payload=torch.load(path,map_location=device,weights_only=False);st=payload['task']
        if not payload['complete'] or payload['source']!=source or st['arm']!='SRC_CE' or st['seed']!=task['seed'] or st['domain']!=task['source_domain'] or st['task_id']!=task['source_task_id']:raise PermissionError('new source identity')
        assert c.hash_state(payload['student'])==entry['student_hash']
        model=c.from_state(reference,device,payload.pop('student'),c.LINEAR);del payload
    else:
        path=Path(base)/'tasks'/task['source_task_id']/'deploy_student.pt'
        payload=torch.load(path,map_location=device,weights_only=False)
        assert payload['source']=='SYNTHETIC' and payload['task']['seed']==task['seed']
        model=c.from_state(reference,device,payload.pop('student'),c.LINEAR);del payload
    sh=c.state_hash(model);c.configure(model,'F_CONV',None);ema=c.ema_from(model);opt=c.optimizer_for(model)
    boundary=dict(source_student_hash=sh,initial_EMA_hash=c.state_hash(ema),fixed_state_hash=c.hash_state(c.fixed_state(model)),trainable_names=list(c.LAYERS),trainable_parameters=sum(x.numel() for x in c.parameters(model)),whitelist=[dict(name=n,shape=list(p.shape),scalars=p.numel()) for n,p in model.named_parameters() if p.requires_grad],source_optimizer_loaded=False,source_EMA_loaded=False,source_RNG_loaded=False,optimizer_state_empty=not opt.state)
    assert boundary['trainable_parameters']==438192 and boundary['initial_EMA_hash']==sh
    model.train();ema.eval();return model,ema,opt,boundary

def pending(root,model,ema,opt,p,counts,get_batches,**kw):
    if p['epoch']==20 and p['index']==p['steps']-1:
        actual=dict(student_hash=c.state_hash(model),EMA_hash=c.state_hash(ema),optimizer_hash=c.optimizer_hash(opt),label_order_hash=p['label_order_hash'],U_opens=p['U_opens'])
        path=root/'warmup.json'
        if not path.exists():c.write_json(path,actual)
        if p['task']['arm']==c.MAIN:
            expected=ct.read(root.parent/p['task']['task_id'].replace(c.MAIN,'F_CONV')/'warmup.json')
            p['row']['checks']['paired_F_CONV_warmup_exact']=actual==expected
    return target.pending(root,model,ema,opt,p,counts,get_batches,**kw)

def train(base,task_id,data,reference,device,*,fixture=None,**kwargs):
    task=ct.task_by_id(task_id) if fixture is None else fixture['task'];root=Path(base)/'tasks'/task_id
    if fixture is None and str(Path(data).resolve())!=ct.read(Path(base)/'private_inputs.json')['data']:raise PermissionError('data identity')
    if task['arm']=='SRC_CE':
        def checkpoint(path,payload):
            save(path,payload)
            if Path(path).name=='latest.pt' and payload['position'] in (payload['total']//5,payload['total']):
                exact=Path(path).parent/('checkpoint_'+str(payload['position'])+'.pt')
                if not exact.exists():os.link(path,exact)
        with patch.multiple(src,verify=ct.verify,admit=ct.admit,read=ct.read,save=checkpoint):
            return src.train(base,task_id,data,reference,device,fixture=fixture,**kwargs)
    def failure(model,ema,opt,counts,phase):
        save(root/('FAILED_PHASE_'+str(time.time_ns())+'.pt'),dict(state='FORENSIC_ONLY_NOT_RESUMABLE',phase=phase,student=model.state_dict(),EMA=ema.state_dict(),optimizer=opt.state_dict(),rng=rng_state(device),counts=dict(counts)))
    with patch.multiple(target,c=c,ct=ct,initial=initial),patch.object(c,'RUN_ARM',task['arm']),patch.object(c.sign,'FAIL_HOOK',failure):
        # Keep the original pending function available for the wrapper, without recursive patching.
        native=target.pending
        def wrapped(*a,**kw):
            with patch.object(target,'pending',native):return pending(*a,**kw)
        with patch.object(target,'pending',wrapped):return target.train(base,task_id,data,reference,device,fixture=fixture,**kwargs)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','task_id','data','reference'):p.add_argument('--'+k.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());ct.neutral_subprocess_paths();root=Path(a['base'])/'tasks'/a['task_id']
    try:
        with Operations(root.parent/(root.name+'_operations')):train(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),automatic_retry=False));raise
