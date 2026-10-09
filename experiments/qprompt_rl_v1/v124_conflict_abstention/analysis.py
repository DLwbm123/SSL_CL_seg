"""Frozen paired conflict-abstention training readout."""
import math

METHODS=('BASE','IGNORE_C','RANDOM_MATCHED','UNIFORM_MATCHED','OFF')
STREAMS=(3,4)


def contexts():
    return [dict(context=i,fold=f,source_step=m,labeled_images=n,condition=k,factor=v)
            for i,(f,m,n,k,v) in enumerate((f,m,n,k,v) for f in (0,1) for m in (2000,8000)
                for n in (2,4) for k,v in (('brightness',.8),('contrast',1.2)))]


def average(rows,keys):
    assert rows
    return {k:sum(r[k] for r in rows)/len(rows) for k in keys}


def training(rows):
    lookup={(r['context'],r['stream'],r['method'],r['step']):r for r in rows}
    assert len(rows)==len(lookup)==320 and set(lookup)=={(i,s,m,t) for i in range(16) for s in STREAMS for m in METHODS for t in (150,300)}
    keys=('gain','final_new_gain','old_gain','reward');trajectories=[]
    for ctx in contexts():
        i=ctx['context']
        for s in STREAMS:
            for m in METHODS:
                a,z=lookup[i,s,m,150],lookup[i,s,m,300]
                assert (a['new_entry'],a['old_entry'])==(z['new_entry'],z['old_entry'])
                gain=.25*(a['new']-a['new_entry'])+.75*(z['new']-z['new_entry'])
                trajectories.append(dict(ctx,stream=s,method=m,gain=gain,final_new_gain=z['new']-z['new_entry'],old_gain=z['old']-z['old_entry'],reward=gain+z['old']-z['old_entry']))
    assert all(math.isfinite(r[k]) for r in trajectories for k in keys)
    summaries=[dict(method=m,**average([r for r in trajectories if r['method']==m],keys)) for m in METHODS]
    pooled={r['method']:r for r in summaries};groups=[]
    for key in ('fold','stream','source_step','labeled_images','condition','context'):
        for v in sorted({r[key] for r in trajectories}):
            for m in METHODS:groups.append(dict(group=f'{key}={v}',method=m,**average([r for r in trajectories if r[key]==v and r['method']==m],keys)))
    comparisons=[]
    for m in METHODS:
        if m=='IGNORE_C':continue
        d={k:pooled['IGNORE_C'][k]-pooled[m][k] for k in keys}
        practical=(d['final_new_gain']>=.002 and d['old_gain']>=-.0025) or (d['old_gain']>=.005 and d['final_new_gain']>=-.0025)
        by={(r['group'],r['method']):r for r in groups}
        robust=all(by[f'{k}={v}','IGNORE_C']['reward']>by[f'{k}={v}',m]['reward'] for k in ('fold','stream') for v in sorted({r[k] for r in trajectories}))
        n=sum(by[f'context={i}','IGNORE_C']['reward']>by[f'context={i}',m]['reward'] for i in range(16))
        comparisons.append(dict(control=m,**d,practical_tradeoff=practical,fold_stream_positive=robust,positive_contexts=n,passed=d['reward']>=.0005 and practical and robust and n>=12))
    absolute=all(r['gain']>0 and r['final_new_gain']>0 and r['old_gain']>=-.0025 for r in groups if r['method']=='IGNORE_C' and r['group'].split('=')[0] in ('fold','stream'))
    passed=absolute and all(r['passed'] for r in comparisons)
    return dict(TRAJECTORIES=trajectories,METHOD_SUMMARY=summaries,GROUP_SUMMARY=groups,
                DECISION=dict(status='MODEL_HELD_OUT_ABSTENTION_SIGNAL' if passed else 'NO_RELIABLE_MODEL_HELD_OUT_ABSTENTION_SIGNAL',passed=passed,absolute=absolute,comparisons=comparisons,independent_confirmation=False,campaign_success=False,RL_benefit_tested=False))


def selfcheck():
    rr=[dict(c,stream=s,method=m,step=t,new=.51 if m=='IGNORE_C' else .5,old=.5,new_entry=.5,old_entry=.5) for c in contexts() for s in STREAMS for m in METHODS for t in (150,300)]
    assert training(rr)['DECISION']['passed']
    for control in ('BASE','RANDOM_MATCHED','UNIFORM_MATCHED','OFF'):
        assert not training([dict(r,new=.51 if r['method']==control else r['new']) for r in rr])['DECISION']['passed']
    assert not training([dict(r,old=.49 if r['method']=='IGNORE_C' else r['old']) for r in rr])['DECISION']['passed']
    assert not training([dict(r,new=.49 if r['method']=='IGNORE_C' and r['fold']==0 else r['new']) for r in rr])['DECISION']['passed']
