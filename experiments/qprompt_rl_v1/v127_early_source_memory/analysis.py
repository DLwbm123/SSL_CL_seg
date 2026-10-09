"""Absolute-source and matched-prefix readout without checkpoint selection."""
import math

NEW=('EARLY_MEMORY','EARLY_SELECTED_EMA')
METHODS=('BASE',*NEW,'FROZEN_SOURCE','FROZEN_ENTRY')
FIELDS=('gain','final_new_gain','old_gain','reward','post100_new','post100_old')


def average(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in FIELDS}


def analyze(new,diagnostic):
    source={(r['context'],r['stream'],r['mode'],r['step'],r['role']):r for r in diagnostic}
    assert len(source)==len(diagnostic)==576
    lookup={(r['context'],r['stream'],r['method'],r['step']):r for r in new}
    assert len(new)==len(lookup)==192 and set(lookup)=={(i,s,m,t) for i in range(16) for s in (3,4) for m in NEW for t in (100,150,300)}
    rows=[]
    for i in range(16):
        for s in (3,4):
            info={k:lookup[i,s,NEW[0],100][k] for k in ('context','fold','source_step','labeled_images','condition','factor','stream')}
            for m in METHODS:
                for t in (100,150,300):
                    if m in NEW:r=dict(lookup[i,s,m,t],provenance='NEW_V127')
                    else:
                        mode,at=('SOURCE',0) if m=='FROZEN_SOURCE' else ('ENTRY',100) if m=='FROZEN_ENTRY' or t==100 else ('BASE',t)
                        a=source[i,s,mode,at,'held'];b=source[i,s,mode,at,'old']
                        r=dict(info,method=m,step=t,new=a['macro'],old=b['macro'],new_rim=a['rim'],new_cup=a['cup'],old_rim=b['rim'],old_cup=b['cup'],provenance='CACHED_V126')
                    r.update(new_source=source[i,s,'SOURCE',0,'held']['macro'],old_source=source[i,s,'SOURCE',0,'old']['macro']);rows.append(r)
    table={(r['context'],r['stream'],r['method'],r['step']):r for r in rows};trajectories=[]
    for i in range(16):
        for s in (3,4):
            for m in METHODS:
                entry,mid,last=[table[i,s,m,t] for t in (100,150,300)]
                gain=.25*(mid['new']-last['new_source'])+.75*(last['new']-last['new_source']);old=last['old']-last['old_source']
                trajectories.append(dict({k:last[k] for k in ('context','fold','source_step','labeled_images','condition','factor','stream','method')},gain=gain,final_new_gain=last['new']-last['new_source'],old_gain=old,reward=gain+old,post100_new=last['new']-entry['new'],post100_old=last['old']-entry['old']))
    assert all(math.isfinite(r[k]) for r in trajectories for k in FIELDS)
    summary=[dict(method=m,**average([r for r in trajectories if r['method']==m])) for m in METHODS]
    groups=[]
    for key in ('fold','stream','source_step','labeled_images','condition','context'):
        for v in sorted({r[key] for r in trajectories}):
            for m in METHODS:groups.append(dict(group=f'{key}={v}',method=m,**average([r for r in trajectories if r[key]==v and r['method']==m])))
    by={(r['group'],r['method']):r for r in groups};pooled={r['method']:r for r in summary};comparisons=[]
    for control in METHODS:
        if control=='EARLY_MEMORY':continue
        d={k:pooled['EARLY_MEMORY'][k]-pooled[control][k] for k in FIELDS}
        practical=(d['final_new_gain']>=.002 and d['old_gain']>=-.0025) or (d['old_gain']>=.005 and d['final_new_gain']>=-.0025)
        robust=all(by[f'{k}={v}','EARLY_MEMORY']['reward']>by[f'{k}={v}',control]['reward'] for k,values in [('fold',(0,1)),('stream',(3,4))] for v in values)
        n=sum(by[f'context={i}','EARLY_MEMORY']['reward']>by[f'context={i}',control]['reward'] for i in range(16))
        comparisons.append(dict(control=control,**d,practical=practical,fold_stream_positive=robust,positive_contexts=n,passed=d['reward']>=.0005 and practical and robust and n>=12))
    absolute=all(r['gain']>0 and r['final_new_gain']>0 and r['old_gain']>=-.0025 for r in groups if r['method']=='EARLY_MEMORY' and r['group'].split('=')[0] in ('fold','stream'))
    passed=absolute and all(r['passed'] for r in comparisons)
    return dict(RESULTS=rows,TRAJECTORIES=trajectories,METHOD_SUMMARY=summary,GROUP_SUMMARY=groups,DECISION=dict(status='EARLY_SOURCE_MEMORY_DEVELOPMENT_GAIN' if passed else 'NO_BROAD_ABSOLUTE_EARLY_SOURCE_MEMORY_GAIN',passed=passed,absolute_from_source=absolute,comparisons=comparisons,independent_confirmation=False,RL_tested=False))


def selfcheck():
    contexts=[dict(context=i,fold=i//8,source_step=2000 if i%8<4 else 8000,labeled_images=2 if i%4<2 else 4,condition='brightness' if i%2==0 else 'contrast',factor=.8 if i%2==0 else 1.2) for i in range(16)]
    diagnostic=[dict(c,stream=s,mode=m,step=t,role=r,macro=.5,rim=.5,cup=.5) for c in contexts for s in (3,4) for m,t in [('SOURCE',0),('ENTRY',100),('BASE',150),('BASE',300),('OFF',150),('OFF',300)] for r in ('fit','held','old')]
    new=[dict(c,stream=s,method=m,step=t,new=.51 if m=='EARLY_MEMORY' else .5,old=.5,new_rim=.5,new_cup=.5,old_rim=.5,old_cup=.5) for c in contexts for s in (3,4) for m in NEW for t in (100,150,300)]
    assert analyze(new,diagnostic)['DECISION']['passed']
    assert not analyze([dict(r,new=.51) for r in new],diagnostic)['DECISION']['passed']
    assert not analyze([dict(r,new=.49 if r['method']=='EARLY_MEMORY' and r['fold']==1 else r['new']) for r in new],diagnostic)['DECISION']['passed']
    try:analyze(new[:-1],diagnostic)
    except AssertionError:pass
    else:raise AssertionError('partial table accepted')
