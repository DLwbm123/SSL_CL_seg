"""Finite-table optimistic policy bounds; no model, image or optimizer access."""
import json
import math
import os
from collections import Counter
from pathlib import Path

METHODS=('BASE','IGNORE_C','RANDOM_MATCHED','UNIFORM_MATCHED','OFF')
CONTROLS=('BASE','RANDOM_MATCHED','UNIFORM_MATCHED','OFF')
METRICS=('gain','final_new_gain','old_gain','reward')


def average(rows,key):return sum(r[key] for r in rows)/len(rows)


def analyze(rows):
    lookup={(r['context'],r['stream'],r['method']):r for r in rows}
    assert len(rows)==len(lookup)==160 and set(lookup)=={(i,s,m) for i in range(16) for s in (3,4) for m in METHODS}
    assert all(math.isfinite(r[k]) for r in rows for k in METRICS)
    units=[];oracle=[]
    for i in range(16):
        for stream in (3,4):
            choices=[lookup[i,stream,m] for m in METHODS];base=choices[0]
            for r in choices:
                assert all(r[k]==base[k] for k in ('fold','source_step','labeled_images','condition','factor'))
                assert abs(r['reward']-r['gain']-r['old_gain'])<1e-12
            unit={k:base[k] for k in ('context','stream','fold','source_step','labeled_images','condition','factor')}
            unit.update({k:max(r[k] for r in choices) for k in METRICS});units.append(unit)
            for objective in ('reward','final_new_gain','old_gain'):
                best=max(choices,key=lambda r:r[objective])
                oracle.append(dict(unit,objective=objective,selected_method=best['method'],**{k:best[k] for k in METRICS}))
    groups=[dict(group=f'{key}={value}',**{k:average([r for r in units if r[key]==value],k) for k in METRICS})
            for key in ('fold','stream','source_step','labeled_images','condition','context') for value in sorted({r[key] for r in units})]
    absolute_failures=[]
    for r in groups:
        if r['group'].split('=')[0] not in ('fold','stream'):continue
        if r['gain']<=0:absolute_failures.append(r['group']+': weighted new upper bound <=0')
        if r['final_new_gain']<=0:absolute_failures.append(r['group']+': final new upper bound <=0')
        if r['old_gain']<-.0025:absolute_failures.append(r['group']+': old upper bound below noninferiority')
    comparisons=[]
    for control in CONTROLS:
        diff=[dict(u,**{k:u[k]-lookup[u['context'],u['stream'],control][k] for k in METRICS}) for u in units]
        pooled={k:average(diff,k) for k in METRICS}
        sub=[dict(group=f'{key}={v}',reward=average([r for r in diff if r[key]==v],'reward')) for key in ('fold','stream') for v in sorted({r[key] for r in diff})]
        contexts=[dict(context=i,reward=average([r for r in diff if r['context']==i],'reward')) for i in range(16)]
        reasons=[]
        if pooled['reward']<.0005:reasons.append('reward upper bound below .0005')
        if pooled['final_new_gain']<.002 and pooled['old_gain']<.005:reasons.append('both practical-arm upper bounds insufficient')
        if any(r['reward']<=0 for r in sub):reasons.append('nonpositive fold/stream reward upper bound')
        positive=sum(r['reward']>0 for r in contexts)
        if positive<12:reasons.append('fewer than12 potentially positive contexts')
        comparisons.append(dict(control=control,optimistic_upper_bounds=pooled,fold_stream_reward_bounds=sub,context_reward_bounds=contexts,possibly_positive_contexts=positive,necessary_conditions_pass=not reasons,failure_reasons=reasons))
    impossible=bool(absolute_failures) or any(not r['necessary_conditions_pass'] for r in comparisons)
    summaries=[dict(objective=obj,**{k:average([r for r in oracle if r['objective']==obj],k) for k in METRICS},selection_counts=dict(Counter(r['selected_method'] for r in oracle if r['objective']==obj))) for obj in ('reward','final_new_gain','old_gain')]
    return dict(UNIT_UPPER_BOUNDS=units,GROUP_UPPER_BOUNDS=groups,COMPARISON_UPPER_BOUNDS=comparisons,ORACLE_ROWS=oracle,ORACLE_SUMMARY=summaries,
        DECISION=dict(status='NO_PRACTICAL_POLICY_MARGIN_ON_FROZEN_TABLE' if impossible else 'NECESSARY_MARGIN_ONLY_NOT_METHOD_SUCCESS',full_gate_possible_not_ruled_out=not impossible,absolute_upper_bound_failures=absolute_failures,scope='only whole-trajectory choice among these five measured actions on this finite table',independent_confirmation=False,learned_policy_tested=False,RL_benefit_tested=False,campaign_success=False))


def selfcheck():
    def fixture(new,old):
        out=[]
        for i in range(16):
            for s in (3,4):
                for m in METHODS:
                    a=new if m=='IGNORE_C' else .01;b=old if m=='IGNORE_C' else 0.
                    out.append(dict(context=i,stream=s,method=m,fold=i//8,source_step=2000 if i%8<4 else 8000,labeled_images=2 if i%4<2 else 4,condition='brightness' if i%2==0 else 'contrast',factor=.8 if i%2==0 else 1.2,gain=a,final_new_gain=a,old_gain=b,reward=a+b))
        return out
    assert analyze(fixture(.02,.01))['DECISION']['full_gate_possible_not_ruled_out']
    assert not analyze(fixture(.01,0.))['DECISION']['full_gate_possible_not_ruled_out']
    bad=[dict(r,gain=-.01,final_new_gain=-.01,reward=-.01+r['old_gain']) for r in fixture(.02,.01)]
    assert analyze(bad)['DECISION']['absolute_upper_bound_failures']
    out=analyze(fixture(.0101,.0001));assert all('both practical-arm upper bounds insufficient' in r['failure_reasons'] for r in out['COMPARISON_UPPER_BOUNDS'])
    assert len(out['ORACLE_ROWS'])==96 and len(out['UNIT_UPPER_BOUNDS'])==32
    return dict(status='PASS',model_calls=0,image_reads=0,optimizer_updates=0,checks=['complete finite table','optimistic feasible case','tie is no relative margin','negative absolute bound','both practical thresholds','fixed oracle counts'])


def main(root):
    selfcheck();source=Path(os.environ['EXEC_SOURCE']);rows=json.loads((source/'TRAJECTORIES.json').read_text())
    assert json.loads((source/'COMPLETION_AUDIT.json').read_text())['status']=='PASS'
    assert json.loads((source/'FINAL_PUBLICATION.json').read_text())['anonymous_http']=='200'
    result=analyze(rows)
    for name,value in result.items():
        with (root/(name+'.json')).open('x') as out:json.dump(value,out,indent=2)
    cost=dict(native_updates=0,actor_updates=0,linear_solves=0,model_forwards=0,image_reads=0,query_calls=0,new_annotation=0,source_trajectories=160,source_condition_streams=32,oracle_rows=96)
    with (root/'COSTS.json').open('x') as out:json.dump(cost,out,indent=2)
    with (root/'FINAL.json').open('x') as out:json.dump(dict(status='COMPLETE',decision=result['DECISION']['status']),out,indent=2)
    print(json.dumps(result['DECISION']))


if __name__=='__main__':
    if os.environ.get('EXEC_SELFCHECK')=='1':print(json.dumps(selfcheck()))
    else:main(Path(os.environ['EXEC_RUN']))
