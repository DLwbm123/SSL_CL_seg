"""Frozen effect-size gates and paired bootstrap, never fitting or threshold search."""
import statistics as st
import numpy as np

DOMAINS=('RIM_ONE_r3','Drishti_GS')
METRICS=('macro_fg_dice','rim_dice','cup_dice')
def summary(v):
    v=list(map(float,v));return dict(mean=st.mean(v),median=st.median(v),sample_SD=st.stdev(v) if len(v)>1 else None,minimum=min(v),maximum=max(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),values=v)
def score(table,s,a,d,metric='macro_fg_dice'):
    rr=[r for r in table if int(r['seed'])==s and r['arm']==a and r['domain']==d]
    if len(rr)!=1:raise ValueError('unique complete final student row required')
    return float(rr[0][metric])
def q(table,s,a):return st.mean(score(table,s,a,d) for d in DOMAINS)
def costs(table,seeds,a,b):
    return [dict(seed=s,domain=d,metric=m,delta=score(table,s,a,d,m)-score(table,s,b,d,m),flag='AMBER_CLASS_COST' if m!='macro_fg_dice' and score(table,s,a,d,m)-score(table,s,b,d,m)<-.02 else '') for s in seeds for d in DOMAINS for m in METRICS]
def risk(rows):
    dm=[st.mean(r['delta'] for r in rows if r['domain']==d and r['metric']=='macro_fg_dice') for d in DOMAINS]
    cls=[st.mean(r['delta'] for r in rows if r['domain']==d and r['metric']==m) for d in DOMAINS for m in METRICS[1:]]
    return dict(domain_mean_guard=min(dm)>=-.005,class_mean_guard=min(cls)>=-.02,severe_class_guard=min(r['delta'] for r in rows if r['metric']!='macro_fg_dice')>=-.05)
def select(table,complete):
    seeds=(31,32,33);out={}
    baseline=max(st.mean(q(table,s,b) for s in seeds) for b in ('SUP_CE','SUP_CED'))
    for a in ('MT_CED','MIX_CED'):
        delta=[q(table,s,a)-q(table,s,'SUP_CED') for s in seeds];rr=costs(table,seeds,a,'SUP_CED')
        checks=dict(mean_gain=st.mean(delta)>=.005,two_positive=sum(x>0 for x in delta)>=2,stronger_baseline=st.mean(q(table,s,a) for s in seeds)-baseline>=.005,**risk(rr),engineering_complete=complete)
        pseudo=st.mean(q(table,s,a)-q(table,s,'MIX_CTX') for s in seeds) if a=='MIX_CED' else st.mean(delta)
        if a=='MIX_CED':checks['pseudo_increment']=pseudo>=.003
        out[a]=dict(eligible=all(checks.values()),checks=checks,paired_gain=summary(delta),costs=rr,pseudo_increment=pseudo)
    mt=out['MT_CED']['eligible'];mix=out['MIX_CED']['eligible'];extra=st.mean(q(table,s,'MIX_CED')-q(table,s,'MT_CED') for s in seeds)
    selected=('MIX_CED' if mix and extra>=.003 else 'MT_CED') if mt else ('MIX_CED' if mix else None)
    foundation=st.mean(q(table,s,'SUP_CED')-q(table,s,'SUP_CE') for s in seeds)
    return dict(status='SELECTED_FOR_P2' if selected else 'NO_SSL_CANDIDATE_ADMITTED',selected=selected,primary_candidate='MIX_CED',simpler_alternative='MT_CED',candidates=out,mix_minus_mt=extra,supervised_foundation_gain=foundation,supervised_foundation_status='SUPERVISED_FOUNDATION_SIGNAL_ONLY' if selected is None and foundation>=.005 else 'DESCRIPTIVE',all_P1_tasks_complete=complete)
def replication(table,selected,complete):
    if selected is None:return dict(status='NOT_ADMITTED',selected=None)
    seeds=(41,42);comparisons={}
    for b in ('SUP_CE','SUP_CED'):
        v=[q(table,s,selected)-q(table,s,b) for s in seeds];rr=costs(table,seeds,selected,b)
        comparisons[b]=dict(gain=summary(v),costs=rr,checks=dict(mean_value=st.mean(v)>=.01,**risk(rr)))
    pseudo=st.mean(q(table,s,selected)-q(table,s,'MIX_CTX') for s in seeds) if selected=='MIX_CED' else None
    passed=complete and all(all(x['checks'].values()) for x in comparisons.values()) and (pseudo is None or pseudo>=.003)
    ced=comparisons['SUP_CED'];direction=ced['gain']['positive']==2 and ced['checks']['class_mean_guard']
    return dict(selected=selected,status='VALUE_REPRODUCED' if passed else 'VALUE_NOT_REPRODUCED',comparisons=comparisons,pseudo_increment=pseudo,pseudo_increment_pass=pseudo is None or pseudo>=.003,engineering_complete=complete,stability='DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS' if direction else ('MEAN_GAIN_ONLY_STABILITY_UNRESOLVED' if passed else 'DIRECTION_NOT_REPRODUCED'))

def intervals(domain_values,seeds,arms,comparisons,phase):
    """Arrays are [seed,arm,patient]; same patient weights across every arm and seed."""
    arrays=[np.asarray(domain_values[d],dtype=float) for d in DOMAINS]
    if any(x.shape[:2]!=(len(seeds),len(arms)) or x.ndim!=3 or not np.isfinite(x).all() for x in arrays):raise ValueError('paired patient tensor')
    g=np.random.default_rng(2026090807);samples={'patient':[],'training_seed':[]}
    for _ in range(2000):
        # Two fixed domains, independently resampled patients; domains never resampled.
        samples['patient'].append(np.mean([x[:,:,g.integers(x.shape[2],size=x.shape[2])].mean(2) for x in arrays],axis=0))
        ix=g.integers(len(seeds),size=len(seeds))
        samples['training_seed'].append(np.mean([x[ix].mean(2) for x in arrays],axis=0))
    out=[]
    for kind,values in samples.items():
        arr=np.asarray(values)
        for a,b in comparisons:
            diff=arr[:,:,arms.index(a)]-arr[:,:,arms.index(b)]
            # Primary patient interval averages fixed trained seeds; also show each fixed model.
            scopes=[('all',diff.mean(1))]+([(str(s),diff[:,i]) for i,s in enumerate(seeds)] if kind=='patient' else [])
            for scope,v in scopes:
                lo,hi=np.quantile(v,[.025,.975]);out.append(dict(phase=phase,resampling=kind,seed_scope=scope,arm=a,baseline=b,replicates=2000,analysis_seed=2026090807,lower_95=float(lo),upper_95=float(hi),contains_zero=bool(lo<=0<=hi),interpretation='fixed-model cohort uncertainty' if kind=='patient' else 'few-seed sensitivity; not precise coverage or independent patients'))
    return out
