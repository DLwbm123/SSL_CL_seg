"""Audit completed paired stage receipts and publish anonymous metrics."""
import json,csv,sys
from pathlib import Path


def main(raw,readout,finite,out):
    raw,readout,out=Path(raw),Path(readout),Path(out);out.mkdir(exist_ok=True)
    get=lambda n:json.loads((raw/n).read_text());lines=lambda n:[json.loads(v) for v in (raw/n).read_text().splitlines()]
    r=get('RESULTS.json');c=r['costs'];assert r['status']=='COMPLETE' and c['optimizer']==208 and c['extra_reverse']==3536
    assert get('EXIT.json')['exit_code']==0 and not (raw/'FAILURE.json').exists()
    for name,key,n in [('physical_updates.jsonl','physical',208),('attempts.jsonl','attempt',208),('reverse_calls.jsonl','index',3536),('candidate_evaluations.jsonl','index',c['candidate_evaluations'])]:assert [x[key] for x in lines(name)]==list(range(1,n+1))
    steps={};metrics={}
    for name in ['single_full','single_quarter','robust_full','robust_quarter']:
        steps[name]=lines(name+'/steps.jsonl');metrics[name]=lines(name+'/metrics.jsonl');assert len(steps[name])==50
        assert len(get(name+'/GOLDEN.json'))==2 and all(x['full_state_and_decision_equal'] for x in get(name+'/GOLDEN.json'))
        for row,prev,next_ in zip(steps[name],metrics[name],metrics[name][1:]):
            assert row['vector']==next_['vector'] and row['metrics']==next_['metrics']
            assert all(v<=b+1e-7+1e-6*abs(b) for v,b in zip(row['vector'],prev['vector']))
            if row['full_model_rollback']:assert row['joint_before']==row['joint_after'] and row['actual_semantic_delta_norm']==0
            else:assert all(a<=b+1e-5 for a,b in zip(row['joint_after'],row['joint_before'])) and sum(row['joint_after'])<sum(row['joint_before'])-2e-5
    post=json.loads((readout/'RESULTS.json').read_text());assert post['status']=='COMPLETE' and post['optimizer_updates']==0 and post['reverse_calls']==0
    assert json.loads((readout/'EXIT.json').read_text())['exit_code']==0 and not (readout/'FAILURE.json').exists()
    f=json.loads(Path(finite).read_text());assert f['training_qualified'] and f['candidate_evaluations']==12 and r['paired_rng_exact'] and r['source_cursor']==3125
    audit=dict(status='PASS',costs=c,finite_candidates=12,ledgers_contiguous=True,all_eight_golden_exact=True,all_step_semantic_constraints_pass=True,accepted_joint_objectives_improve=True,rollback_cost_preserved=True,posterior_readout_completed=True,space=r['space'],no_failure=True,exit_code=0)
    r['runtime_seconds']=r.pop('finished')-r.pop('started');f.pop('started',None);f.pop('finished',None)
    for n,obj in [('RESULTS',r),('AUDIT',audit),('FINITE',f),('STEPS',steps),('METRICS',metrics),('POSTERIOR',post)]:
        (out/(n+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    rows=[dict(arm=arm,step=row['step'],alpha=row['alpha'],rollback=row['full_model_rollback'],semantic_delta=row['actual_semantic_delta_norm'],joint_before=sum(row['joint_before'])/2,joint_after=sum(row['joint_after'])/2,**{ch+'_Dice_pct':100*sum(map(sum,v['dice']))/6 for ch,v in row['metrics'].items()}) for arm,ss in steps.items() for row in ss]
    with (out/'TRAJECTORY.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
