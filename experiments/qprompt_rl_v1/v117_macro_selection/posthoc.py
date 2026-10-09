from pathlib import Path
import json,statistics as st,collections
p=Path(__file__).resolve().parent;rows=json.loads((p/'RESULTS.json').read_text());sel=json.loads((p/'SELECTED_RESULTS.json').read_text());L={(v['seed'],v['group'],v['method']):v for v in rows};keys=sorted({(r['seed'],r['group']) for r in rows});M=('RANDOM','CONFIDENCE','COVERAGE','OFF');metric=('gain','final_new_gain','old_gain','reward');mean=lambda a:{m:st.mean(r[m] for r in a) for m in metric};out={}
out['context']=[]
for c in range(8):
 data={m:mean([r for r in rows if r['context']==c and r['method']==m]) for m in M};nn=mean([r for r in sel if r['context']==c and r['policy']=='NN_LOCO']);oracle=mean([r for r in sel if r['context']==c and r['policy']=='ORACLE']);out['context'].append(dict(context=c,rules=data,nn=nn,oracle=oracle,choices=dict(collections.Counter(r['method'] for r in sel if r['context']==c and r['policy']=='NN_LOCO'))))
out['cycle']=[dict(cycle=c,methods={m:mean([r for r in rows if r['cycle']==c and r['method']==m]) for m in M},nn=mean([r for r in sel if r['cycle']==c and r['policy']=='NN_LOCO'])) for c in range(4)]
out['switches']=[]
for m in M[:3]:
 rs=[r for r in sel if r['policy']=='NN_LOCO' and r['method']==m];diff=[{k:r[k]-L[(r['seed'],r['group'],'COVERAGE')][k] for k in metric} for r in rs];out['switches'].append(dict(method=m,count=len(rs),conditional_delta=mean(diff),pooled_contribution={k:sum(d[k] for d in diff)/64 for k in metric},reward_positive=sum(d['reward']>0 for d in diff),reward_negative=sum(d['reward']<0 for d in diff)))
out['negative_states']={m:dict(reward_negative=sum(L[(*k,m)]['reward']<0 for k in keys),new_negative=sum(L[(*k,m)]['final_new_gain']<0 for k in keys),old_negative=sum(L[(*k,m)]['old_gain']<0 for k in keys)) for m in M};out['all_three_negative']=sum(max(L[(*k,m)]['reward'] for m in M[:3])<0 for k in keys)
out['off']=dict(off_better_than_all=sum(L[(*k,'OFF')]['reward']>max(L[(*k,m)]['reward'] for m in M[:3]) for k in keys),oracle4_minus_oracle3=st.mean(max(L[(*k,m)]['reward'] for m in M)-max(L[(*k,m)]['reward'] for m in M[:3]) for k in keys))
out['oracle_minus_fixed_seed']={s:st.mean(max(L[(*k,m)]['reward'] for m in M[:3])-L[(*k,'COVERAGE')]['reward'] for k in keys if k[0]==s) for s in (601,602)}
out['best_new_oracle_increment_over_coverage']=st.mean(max(L[(*k,m)]['final_new_gain'] for m in M[:3])-L[(*k,'COVERAGE')]['final_new_gain'] for k in keys)
out['best_old_oracle_increment_over_coverage']=st.mean(max(L[(*k,m)]['old_gain'] for m in M[:3])-L[(*k,'COVERAGE')]['old_gain'] for k in keys)
out['all_four_negative']=sum(max(L[(*k,m)]['reward'] for m in M)<0 for k in keys)
nn=[r for r in sel if r['policy']=='NN_LOCO'];oracle={(r['seed'],r['group']):r for r in sel if r['policy']=='ORACLE'}
out['NN_oracle_agree']=sum(r['method']==oracle[(r['seed'],r['group'])]['method'] for r in nn)
out['confidence_switch_detail']={}
for sign in ('positive','negative'):
 ds=[r['reward']-L[(r['seed'],r['group'],'COVERAGE')]['reward'] for r in nn if r['method']=='CONFIDENCE']
 ds=[v for v in ds if (v>0 if sign=='positive' else v<0)]
 out['confidence_switch_detail'][sign]=dict(n=len(ds),mean=st.mean(ds) if ds else None,sum=sum(ds))
head=[(v['oracle']['reward']-v['rules']['COVERAGE']['reward'])/8 for v in out['context']]
out['oracle_margin_context_contributions']=head;out['ctx2_3_margin_share']=(head[2]+head[3])/sum(head)
net=sum(r['pooled_contribution']['reward'] for r in out['switches'])
assert abs(net-st.mean(r['reward']-L[(r['seed'],r['group'],'COVERAGE')]['reward'] for r in nn))<1e-14
assert sum(r['count'] for r in out['switches'])==64
out['confidence_share_of_NN_loss']=out['switches'][1]['pooled_contribution']['reward']/net
out['scope']='Posthoc arithmetic on published scalars only; no fit, model/query/image calls or gate changes. OFF inclusion is descriptive, not a new preregistered readout.'
with (p/'POSTHOC_ANALYSIS.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(dict(status='PASS',query_calls=0,model_forwards=0,optimizer_calls=0,linear_solves=0)))
