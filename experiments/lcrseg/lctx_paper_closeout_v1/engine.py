"""Reuse the pending-first target loop; reset targets from existing source students."""
import argparse,contextlib,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from . import core as c,contract as ct
from experiments.lcrseg.dpr_v0_1 import engine as common
from experiments.lcrseg.ams_seq_transfer_v0_1 import evaluate as scoring
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations,ACTIVE

def initial(base,task,reference,device,fixture=None):
    b=Path(base)
    if fixture is None:
        old=Path(ct.read(b/'private_inputs.json')['parent']);entry=ct.read(b/'REUSE_BINDING.json')['sources'][task['source_task_id']]
        payload=torch.load(old/'tasks'/task['source_task_id']/'deploy_student.pt',map_location=device,weights_only=False)
        st=payload['task']
        if not payload['complete'] or payload['source']!=ct.PARENT_SOURCE or payload['head_mode']!=c.LINEAR or st!=entry['task'] or st['seed']!=task['seed'] or st['domain']!=task['source_domain'] or st['arm']!='SRC_CE' or payload['student_hash']!=entry['student_hash']:raise PermissionError('source deployment/binding identity')
        source_hash=entry['student_hash']
        model=c.from_state(reference,device,payload.pop('student'),c.LINEAR);del payload
    else:
        model=c.parent.build(reference,device,task['seed'],task['source_domain'],'SUP_CE');source_hash=c.state_hash(model)
    c.configure(model,'F_CONV',None);ema=c.ema_from(model);opt=c.optimizer_for(model)
    boundary=dict(source_student_hash=source_hash,initial_EMA_hash=source_hash,fixed_state_hash=c.hash_state(c.fixed_state(model)),trainable_names=list(c.LAYERS),trainable_parameters=sum(p.numel() for p in c.parameters(model)),whitelist=[dict(name=n,shape=list(p.shape),scalars=p.numel()) for n,p in model.named_parameters() if p.requires_grad],source_optimizer_loaded=False,source_EMA_loaded=False,source_RNG_loaded=False,optimizer_state_empty=not opt.state)
    assert boundary['trainable_parameters']==438192
    model.train();ema.eval();return model,ema,opt,boundary

@torch.no_grad()
def diagnose(model,ema,boundary,row):
    checks=dict(row['checks'],finite_models=all(bool(torch.isfinite(p).all()) for m in (model,ema) for p in m.parameters()),fixed_gradients_absent=all(p.grad is None for p in model.parameters() if not p.requires_grad),EMA_gradients_absent=all(p.grad is None for p in ema.parameters()))
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks)

def train(base,task_id,data,reference,device,fixture=None,**kw):
    task=ct.task_by_id(task_id) if fixture is None else fixture['task'];b=Path(base)
    if fixture is None:subset=ct.read(b/'private_subsets.json')[task['domain']][task['budget']]
    else:subset=fixture.get('subset')
    with contextlib.ExitStack() as stack:
        if subset is not None:
            stack.enter_context(c.selected_data([r['case_id'] for r in subset]))
            stack.enter_context(patch.object(c,'patients_for',lambda *a:[r['patient_id'] for r in subset]))
        stack.enter_context(patch.multiple(common,c=c,ct=ct,initial=initial,diagnose=diagnose))
        result=common.train(base,task_id,data,reference,device,fixture=fixture,**kw)
    if result['status']=='TRAINING_COMPLETE':
        for name in list(result['memory']):
            if any(x in name for x in ('response','FP64','proposal','solver','diagnostic')):del result['memory'][name]
        result['memory'].update(response_workspace_bytes=0,training_full_models=2,deployment_students=1)
        result.update(budget=task.get('budget','synthetic'),L_patients=len(subset) if subset else None,source_training_updates=0,reused_LCTX_training_updates=0)
        c.write_json(b/'tasks'/task_id/'receipt.json',result)
    return result

def evaluate(base,task_id,data,reference,device):
    source=ct.verify();task=ct.admit(base,task_id,source);b=Path(base)
    seal=ct.read(b/'TARGET_WEIGHT_SEAL.json')
    if seal['source']!=source or len(seal['students'])!=30 or seal['students'][task_id]!=ct.read(b/'tasks'/task_id/'receipt.json')['student_hash']:raise PermissionError('all final targets must be sealed')
    if str(Path(data).resolve())!=ct.read(b/'private_inputs.json')['data']:raise PermissionError('data identity')
    with patch.multiple(scoring,verify=ct.verify,admit=ct.admit,read=ct.read):return scoring.evaluate(base,task_id,data,reference,device)

def main():
    p=argparse.ArgumentParser()
    for name in ('base','task_id','data','reference','phase'):p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--resume',action='store_true');a=vars(p.parse_args());phase=a.pop('phase');resume=a.pop('resume');ct.neutral_subprocess_paths()
    root=Path(a['base'])/'tasks'/a['task_id'];ops=root.parent/(root.name+('_operations' if phase=='train' else '_eval_operations'))
    if resume:raise PermissionError('preserve attempt and reconcile physical operation counts before an authorized recovery')
    try:
        with Operations(ops):
            (train if phase=='train' else evaluate)(**a,device=torch.device('cuda:0'))
    except BaseException as exc:
        c.write_json(root.parent/(root.name+'_'+phase+'_failure.json'),dict(status='INCOMPLETE_ENGINEERING',error=repr(exc),automatic_retry=False));raise

if __name__=='__main__':main()
