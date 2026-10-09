"""Frozen paired, train-only readout; no fitted model or parameter selection."""
import math

METHODS=('BASE','SEMANTIC','ROTATED')
METRICS=('gain','final_new_gain','old_gain','reward')


def mean(rows):
    assert rows
    return {k:sum(r[k] for r in rows)/len(rows) for k in METRICS}


def comparison(a,b):
    d={k:a[k]-b[k] for k in METRICS}
    d['pass']=d['reward']>=.0005 and ((d['final_new_gain']>=.002 and d['old_gain']>=-.0025) or (d['old_gain']>=.005 and d['final_new_gain']>=-.0025))
    return d


def analyze(rows):
    lookup={(r['context'],r['stream'],r['method'],r['step']):r for r in rows}
    assert len(rows)==len(lookup)==192 and set(lookup)=={(i,s,m,t) for i in range(8) for s in (3,4) for m in METHODS for t in (150,200,250,300)}
    assert all(math.isfinite(r[k]) for r in rows for k in ('new','old','new_entry','old_entry','new_gain','old_gain'))
    trajectories=[]
    for i in range(8):
        for s in (3,4):
            for m in METHODS:
                points=[lookup[i,s,m,t] for t in (150,200,250,300)];a,z=points[0],points[-1]
                assert len({(r['new_entry'],r['old_entry']) for r in points})==1
                assert all(abs(r['new_gain']-(r['new']-r['new_entry']))<1e-14 and abs(r['old_gain']-(r['old']-r['old_entry']))<1e-14 for r in points)
                gain=.25*a['new_gain']+.75*z['new_gain']
                trajectories.append(dict(context=i,stream=s,method=m,gain=gain,final_new_gain=z['new_gain'],old_gain=z['old_gain'],reward=gain+z['old_gain']))
    summaries=[dict(method=m,n=16,**mean([r for r in trajectories if r['method']==m])) for m in METHODS]
    contexts=[dict(context=i,method=m,n=2,**mean([r for r in trajectories if r['method']==m and r['context']==i])) for i in range(8) for m in METHODS]
    streams=[dict(stream=s,method=m,n=8,**mean([r for r in trajectories if r['method']==m and r['stream']==s])) for s in (3,4) for m in METHODS]
    pooled={r['method']:r for r in summaries};perstream={(r['stream'],r['method']):r for r in streams};percontext={(r['context'],r['method']):r for r in contexts}
    delta=comparison(pooled['SEMANTIC'],pooled['BASE']);negative_control=comparison(pooled['SEMANTIC'],pooled['ROTATED'])
    absolute=all((r:=perstream[s,'SEMANTIC'])['gain']>0 and r['final_new_gain']>0 and r['old_gain']>=-.0025 for s in (3,4))
    beats=all(perstream[s,'SEMANTIC']['reward']>perstream[s,m]['reward'] for s in (3,4) for m in ('BASE','ROTATED'))
    positive=sum(percontext[i,'SEMANTIC']['reward']>percontext[i,'BASE']['reward'] for i in range(8))
    signal=delta['pass'] and absolute and beats and negative_control['reward']>0 and positive>=6
    decision=dict(status='TRAIN_ONLY_TARGET_REPAIR_SIGNAL' if signal else 'NO_RELIABLE_TRAIN_ONLY_TARGET_REPAIR_SIGNAL',semantic_vs_base=delta,semantic_vs_rotated=negative_control,semantic_absolute=absolute,each_stream_beats_both_controls=beats,positive_contexts=positive,train_only_signal=signal,campaign_success=False,independent_confirmation=False,actor_updates=0)
    return dict(TRAJECTORIES=trajectories,METHOD_SUMMARY=summaries,CONTEXT_SUMMARY=contexts,STREAM_SUMMARY=streams,DECISION=decision)


def selfcheck():
    rows=[dict(context=i,stream=s,method=m,step=t,new=.5,old=.5,new_entry=.5,old_entry=.5,new_gain=0.,old_gain=0.) for i in range(8) for s in (3,4) for m in METHODS for t in (150,200,250,300)]
    assert not analyze(rows)['DECISION']['train_only_signal']
    for r in rows:
        if r['method']=='SEMANTIC':r.update(new=.51,new_gain=.01)
    assert analyze(rows)['DECISION']['train_only_signal']
    for r in rows:
        if r['method']=='ROTATED':r.update(new=.51,new_gain=.01)
    assert not analyze(rows)['DECISION']['train_only_signal']
    rows[0]['new']=float('nan')
    try:analyze(rows)
    except AssertionError:pass
    else:raise AssertionError('nonfinite metric admitted')
