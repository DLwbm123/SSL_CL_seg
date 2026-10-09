"""Frozen condition-held-out nearest-neighbor readout; no fitted hyperparameters."""
import numpy as np

METHODS=('RANDOM','CONFIDENCE','COVERAGE','OFF')
METRICS=('gain','final_new_gain','old_gain','reward')


def choose(train_x,train_r,test_x):
    scale=train_x.std(0);scale[scale<1e-6]=1
    distance=(((train_x-test_x)/scale)**2).sum(1)
    nearest=distance==distance.min()
    return int(train_r[nearest].mean(0).argmax()),int(train_r.mean(0).argmax())


def mean(rows):return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}


def comparison(a,b):
    d={k:a[k]-b[k] for k in METRICS}
    d['pass']=bool(d['reward']>=.0005 and ((d['final_new_gain']>=.002 and d['old_gain']>=-.0025) or (d['old_gain']>=.005 and d['final_new_gain']>=-.0025)))
    return d


def absolute(rows):
    return all((m:=mean([r for r in rows if r['seed']==seed]))['gain']>0 and m['final_new_gain']>0 and m['old_gain']>=-.0025 for seed in (601,602))


def analyze(rows,features):
    keys=[(s,g) for s in (601,602) for g in range(32)]
    lookup={(r['seed'],r['group'],r['method']):r for r in rows};fl={(r['seed'],r['group']):r['x'] for r in features}
    assert len(rows)==len(lookup)==256 and len(features)==len(fl)==64
    assert set(lookup)=={(*k,m) for k in keys for m in METHODS} and set(fl)==set(keys)
    x=np.asarray([fl[k] for k in keys],dtype=float);r=np.asarray([[lookup[(*k,m)]['reward'] for m in METHODS[:3]] for k in keys]);contexts=np.array([g%8 for s,g in keys])
    assert x.shape==(64,26) and np.isfinite(x).all() and np.isfinite(r).all()
    fixed=int(r.mean(0).argmax());selected=[]
    for i,key in enumerate(keys):
        mask=contexts!=contexts[i];assert int(mask.sum())==56
        nn,glob=choose(x[mask],r[mask],x[i]);oracle=int(r[i].argmax())
        for policy,action in [('NN_LOCO',nn),('GLOBAL_LOCO',glob),('ORACLE',oracle),('BEST_FIXED_POSTHOC',fixed),*[(m,j) for j,m in enumerate(METHODS)]]:
            selected.append(dict(lookup[(*key,METHODS[action])],policy=policy,selected_method=METHODS[action]))
    names=['NN_LOCO','GLOBAL_LOCO','ORACLE','BEST_FIXED_POSTHOC',*METHODS]
    summaries=[dict(policy=p,n=64,**mean([v for v in selected if v['policy']==p])) for p in names];means={v['policy']:v for v in summaries}
    folds=[dict(context=i,policy=p,n=8,**mean([v for v in selected if v['policy']==p and v['context']==i])) for i in range(8) for p in names]
    seeds=[dict(seed=s,policy=p,n=32,**mean([v for v in selected if v['policy']==p and v['seed']==s])) for s in (601,602) for p in names]
    cmp={p:comparison(means['NN_LOCO'],means[p]) for p in ('GLOBAL_LOCO','RANDOM','CONFIDENCE','COVERAGE','OFF')}
    foldpositive=sum(next(v for v in folds if v['context']==i and v['policy']=='NN_LOCO')['reward']>next(v for v in folds if v['context']==i and v['policy']=='GLOBAL_LOCO')['reward'] for i in range(8))
    oracle_cmp=comparison(means['ORACLE'],means['BEST_FIXED_POSTHOC'])
    oracle_absolute=absolute([v for v in selected if v['policy']=='ORACLE'])
    nn_absolute=absolute([v for v in selected if v['policy']=='NN_LOCO'])
    seed_beats=all(next(v for v in seeds if v['seed']==s and v['policy']=='NN_LOCO')['reward']>next(v for v in seeds if v['seed']==s and v['policy']=='GLOBAL_LOCO')['reward'] for s in (601,602))
    signal=all(v['pass'] for v in cmp.values()) and nn_absolute and seed_beats and foldpositive>=6
    decision=dict(status='TRAIN_ONLY_MACRO_LEARNABILITY_SIGNAL' if signal else 'NO_RELIABLE_TRAIN_ONLY_MACRO_SIGNAL',oracle_absolute=oracle_absolute,oracle_vs_best_fixed=oracle_cmp,NN_absolute=nn_absolute,NN_comparisons=cmp,NN_each_seed_beats_fold_global=seed_beats,NN_positive_context_folds=foldpositive,train_only_signal=signal,campaign_success=False,independent_confirmation=False,actor_updates=0)
    return dict(METHOD_SUMMARY=summaries,CONTEXT_SUMMARY=folds,SEED_SUMMARY=seeds,SELECTED_RESULTS=selected,DECISION=decision)


def selfcheck():
    a,b=choose(np.array([[0.,0.],[0.,0.],[2.,0.]]),np.array([[0.,2.,0.],[0.,0.,4.],[5.,0.,0.]]),np.array([0.,0.]))
    assert (a,b)==(2,0)
    base=dict(gain=0.,final_new_gain=0.,old_gain=0.,reward=0.)
    assert not comparison(base,base)['pass']
    assert comparison(dict(base,final_new_gain=.003,reward=.003),base)['pass']
    keys=[(s,g) for s in (601,602) for g in range(32)]
    features=[dict(seed=s,group=g,x=[float(g%8)]*26) for s,g in keys]
    rows=[dict(seed=s,group=g,context=g%8,cycle=g//8,method=m,**base) for s,g in keys for m in METHODS]
    out=analyze(rows,features);assert not out['DECISION']['train_only_signal'] and not out['DECISION']['campaign_success']
    assert all(v['selected_method']=='RANDOM' for v in out['SELECTED_RESULTS'] if v['policy']=='NN_LOCO')
    changed=[dict(v) for v in rows]
    for v in changed:
        if v['context']==0 and v['method']=='COVERAGE':v['reward']=100
    again=analyze(changed,features)
    assert all(v['selected_method']=='RANDOM' for v in again['SELECTED_RESULTS'] if v['context']==0 and v['policy'] in ('NN_LOCO','GLOBAL_LOCO'))
    features[0]['x'][0]=float('nan')
    try:analyze(rows,features)
    except AssertionError:pass
    else:raise AssertionError('nonfinite feature admitted')
