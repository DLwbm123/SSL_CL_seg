"""Frozen model-held-out gate and conditional paired-training readout."""
import math

METHODS=('BASE','RULE','IGNORE_C','EMA_ONLY','OFF')
STREAMS=(3,4)


def contexts():
    return [dict(context=i,fold=f,source_step=m,labeled_images=n,condition=k,factor=v)
            for i,(f,m,n,k,v) in enumerate((f,m,n,k,v) for f in (0,1) for m in (2000,8000)
                for n in (2,4) for k,v in (('brightness',.8),('contrast',1.2)))]


def average(rows,keys):
    assert rows
    return {k:sum(r[k] for r in rows)/len(rows) for k in keys}


def quality(rows):
    lookup={(r['context'],r['stream']):r for r in rows}
    assert len(rows)==len(lookup)==32 and set(lookup)=={(i,s) for i in range(16) for s in STREAMS}
    keys=('weighted_net_accuracy','weighted_nll_delta')
    assert all(r['weight']>0 and all(math.isfinite(r[k]) for k in keys) for r in rows)
    percontext=[dict(ctx,**average([lookup[ctx['context'],s] for s in STREAMS],keys)) for ctx in contexts()]
    groups=[dict(group=f'{k}={v}',**average([r for r in rows if r[k]==v],keys))
            for k in ('fold','stream','source_step','labeled_images','condition') for v in sorted({r[k] for r in rows})]
    positive=lambda r:r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0
    pooled=average(rows,keys);n=sum(positive(r) for r in percontext)
    passed=positive(pooled) and n>=12 and all(positive(r) for r in groups)
    return dict(status='MODEL_HELD_OUT_DEVELOPMENT_SIGNAL' if passed else 'STOP_NO_MODEL_HELD_OUT_SIGNAL',
                passed=passed,pooled=pooled,joint_positive_contexts=n,contexts=percontext,groups=groups,
                model_level_holdout=True,independent_confirmation=False,campaign_success=False)


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
        if m=='RULE':continue
        d={k:pooled['RULE'][k]-pooled[m][k] for k in keys}
        practical=(d['final_new_gain']>=.002 and d['old_gain']>=-.0025) or (d['old_gain']>=.005 and d['final_new_gain']>=-.0025)
        by={(r['group'],r['method']):r for r in groups}
        robust=all(by[f'{k}={v}','RULE']['reward']>by[f'{k}={v}',m]['reward'] for k in ('fold','stream') for v in sorted({r[k] for r in trajectories}))
        n=sum(by[f'context={i}','RULE']['reward']>by[f'context={i}',m]['reward'] for i in range(16))
        comparisons.append(dict(control=m,**d,practical_tradeoff=practical,fold_stream_positive=robust,positive_contexts=n,passed=d['reward']>=.0005 and practical and robust and n>=12))
    absolute=all(r['gain']>0 and r['final_new_gain']>0 and r['old_gain']>=-.0025 for r in groups if r['method']=='RULE' and r['group'].split('=')[0] in ('fold','stream'))
    passed=absolute and all(r['passed'] for r in comparisons)
    return dict(TRAJECTORIES=trajectories,METHOD_SUMMARY=summaries,GROUP_SUMMARY=groups,
                DECISION=dict(status='MODEL_HELD_OUT_TRAINING_SIGNAL' if passed else 'NO_RELIABLE_MODEL_HELD_OUT_TRAINING_SIGNAL',passed=passed,absolute=absolute,comparisons=comparisons,independent_confirmation=False,campaign_success=False,RL_benefit_tested=False))


def selfcheck():
    rows=[dict(c,stream=s,weight=1,weighted_net_accuracy=.01,weighted_nll_delta=-.01) for c in contexts() for s in STREAMS]
    assert quality(rows)['passed']
    assert not quality([dict(r,weighted_net_accuracy=0) for r in rows])['passed']
    assert not quality([dict(r,weighted_nll_delta=.02 if r['fold']==0 else -.01) for r in rows])['passed']
    rr=[dict(c,stream=s,method=m,step=t,new=.51 if m=='RULE' else .5,old=.5,new_entry=.5,old_entry=.5) for c in contexts() for s in STREAMS for m in METHODS for t in (150,300)]
    assert training(rr)['DECISION']['passed']
    assert not training([dict(r,new=.51 if r['method']=='IGNORE_C' else r['new']) for r in rr])['DECISION']['passed']
    assert not training([dict(r,old=.49 if r['method']=='RULE' else r['old']) for r in rr])['DECISION']['passed']
