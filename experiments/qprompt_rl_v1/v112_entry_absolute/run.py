"""Measure frozen entry retention once; decompose existing complete trajectories."""
import fcntl
import importlib.util
import json
import os
import random
import statistics as st
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def differences(row,entryold):
    assert all(np.isfinite(row[k]) for k in ('new','mid_new','new_entry','old','old_memory_reference','gain','utility')) and np.isfinite(entryold)
    gain=.25*(row['mid_new']-row['new_entry'])+.75*(row['new']-row['new_entry'])
    assert abs(gain-row['gain'])<1e-12
    frozen=-max(0.,row['old_memory_reference']-entryold-.005)
    return dict(final_new_minus_entry=row['new']-row['new_entry'],mid_new_minus_entry=row['mid_new']-row['new_entry'],weighted_new_gain=gain,old_final_minus_entry=row['old']-entryold,old_final_minus_memory=row['old']-row['old_memory_reference'],frozen_entry_utility=frozen,utility_minus_frozen=row['utility']-frozen)


def selfcheck():
    r=dict(new=.8,mid_new=.8,new_entry=.8,old=.85,old_memory_reference=.86,gain=0.,utility=-.005)
    a=differences(r,.85);assert abs(a['utility_minus_frozen'])<1e-12 and a['final_new_minus_entry']==a['old_final_minus_entry']==0
    r.update(new=.79,mid_new=.82,gain=-.0025,utility=-.0075);a=differences(r,.85)
    assert abs(a['weighted_new_gain']+.0025)<1e-12 and abs(a['final_new_minus_entry']+.01)<1e-12 and abs(a['mid_new_minus_entry']-.02)<1e-12


def main(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);selfcheck()
    prior=Path(cfg['v111']);assert D.read(prior/'FINAL.json')['status']=='COMPLETE' and D.read(prior/'EXIT.json')['exit_code']==0
    assert D.read(prior/'ENDPOINT_LOCK.json')['trajectories']==264
    data=D.read(prior/'DEVELOPMENT_RESULTS.json');assert len(data['rows'])==264
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    roles=c.split_roles(cfg['data']);assert roles==D.read(Path(cfg['action_root'])/'ROLES.private.json') and len(roles['Q_dev_old'])==4
    counts=Counter();sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role=='train_labeled' and ds.rows[i]['case_id'] in set(roles['Q_dev_old'])
        counts['query_item_attempts']+=1;value=sample(ds,i);counts['query_item_success']+=1;return value
    c.e.primitive.CurrentData.__getitem__=guarded;ledger=b.JobLedger(root,{})
    registered=D.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'];entries=[]
    for i,(m,n,kind,factor) in enumerate(registered):
        ctx=(m,n,(kind,factor),f'dev{i}');t=b.make(cfg,roles,ledger,ctx,300)
        entry=torch.load(Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt',map_location='cpu',weights_only=False);c.restore(t,entry);assert t.step==100
        before=c.snapshot(t);counts['query_calls_attempted']+=1;record=dict(context=ctx[3],role='Q_dev_old',step=100,images=4)
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:old=scores(t,roles,'Q_dev_old',('identity',1.))['macro']
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));counts['query_calls_success']+=1
        assert c.e.same(before,c.snapshot(t)) and not ledger.count
        refs=[r for r in data['rows'] if r['context']==ctx[3]];assert len(refs)==66 and len({r['new_entry'] for r in refs})==len({r['old_memory_reference'] for r in refs})==1
        row=dict(context=ctx[3],new_entry=refs[0]['new_entry'],old_entry=old,old_memory_reference=refs[0]['old_memory_reference'],frozen_utility=-max(0.,refs[0]['old_memory_reference']-old-.005),snapshot_RNG_preserved=True)
        entries.append(row);D.write(root/f'ENTRY_{i}.json',row);del t;torch.cuda.empty_cache()
    assert dict(counts)==dict(query_calls_attempted=4,query_calls_success=4,query_item_attempts=16,query_item_success=16) and not ledger.count
    lookup={r['context']:r for r in entries};rows=[]
    for stage,values in [('V111',data['rows']),('V108',D.read(Path(cfg['v108'])/'DEVELOPMENT_RESULTS.json')['rows'])]:
        for r in values:
            e=lookup[r['context']];assert r['new_entry']==e['new_entry'] and r['old_memory_reference']==e['old_memory_reference']
            seq='->'.join('OFF' if a==12 else 'ON' for a in r['actions'])
            rows.append(dict(stage=stage,context=r['context'],stream=r['stream'],method=r['method'],arm=r['method'].rsplit('_',1)[0] if r['method'].startswith(('SAMPLE_','ARGMAX_')) else r['method'],action_sequence=seq,mid_new=r['mid_new'],new=r['new'],old=r['old'],new_entry=r['new_entry'],old_memory_reference=r['old_memory_reference'],utility=r['utility'],**differences(r,e['old_entry'])))
    assert len(rows)==504;metrics=list(differences(data['rows'][0],entries[0]['old_entry']))
    def aggregate(fields):
        groups={}
        for r in rows:groups.setdefault(tuple(r[f] for f in fields),[]).append(r)
        return [dict(zip(fields,key),n=len(rs),**{k:st.mean(r[k] for r in rs) for k in metrics}) for key,rs in sorted(groups.items())]
    sequence=aggregate(['stage','method','action_sequence'])
    present={(r['stage'],r['method'],r['action_sequence']) for r in sequence}
    for stage,method in sorted({(r['stage'],r['method']) for r in rows}):
        for category in ('OFF->OFF','OFF->ON','ON->OFF','ON->ON'):
            if (stage,method,category) not in present:sequence.append(dict(stage=stage,method=method,action_sequence=category,n=0,**{k:None for k in metrics}))
    for name,values in [('ENTRY_BASELINES',entries),('ABSOLUTE_RESULTS',rows),('METHOD_SUMMARY',aggregate(['stage','method'])),('ARM_SUMMARY',aggregate(['stage','arm'])),('CONTEXT_METHOD_SUMMARY',aggregate(['stage','context','method'])),('SEQUENCE_SUMMARY',sequence)]:D.write(root/(name+'.json'),values);D.table(root/(name+'.csv'),values)
    D.write(root/'COSTS.json',dict(native_updates=0,actor_optimizer_updates=0,linear_solves=0,development_query_images=16,query_calls=4,counts=dict(counts),reused_V111_rows=264,reused_V108_rows=240,new_annotation_cases=0))
    D.write(root/'QUALIFICATION.json',dict(status='PASS',synthetic_arithmetic=True,all4_query_snapshot_RNG_preserved=True,query_role_guard=True,no_optimizer_capability=True,optimizer_calls=0))
    D.write(root/'FINAL.json',dict(status='COMPLETE',decision='ABSOLUTE_DIAGNOSTIC_NO_ORIGINAL_GATE_CHANGE',V111_original_decision=data['status'],publication='PENDING',time=time.time()))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    spec=importlib.util.spec_from_file_location('diagnostic',cfg['diagnostic_entry']);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS absolute/frozen-baseline arithmetic; zero model calls')
    else:
        D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:main(root,cfg)
        except BaseException as exc:D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
