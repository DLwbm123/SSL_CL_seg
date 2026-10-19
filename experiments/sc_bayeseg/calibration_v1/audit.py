"""Read-only independent ledger/state/metric audit for the authorized restart."""
import json,torch
from pathlib import Path
from sc_bayeseg.run import same
import os
settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
r=Path(settings['output']);old=Path(settings['prior_attempt']);origin=Path(settings['origin'])
load=lambda p:json.loads(p.read_text())
cp=lambda p:torch.load(p,map_location='cpu',weights_only=False)
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
def aggregate(row):
 out=dict(row)
 for k in ['stochastic_dice','direct_dice','stochastic_recall','direct_recall']:
  if k in out:
   a=torch.tensor(out.pop(k),dtype=torch.float64);out[k+'_class_mean']=a.mean(0).tolist();out[k+'_class_min']=a.min(0).values.tolist();out[k+'_macro_mean']=float(a.mean())
 return out
root=load(r/'RESULTS.json');assert root['status']=='COMPLETE' and load(r/'EXIT.json')['exit_code']==0 and not (r/'FAILURE.json').exists()
history=rows(old/'physical_updates.jsonl');ledger=rows(r/'physical_updates.jsonl')
assert len(history)==2 and [v['physical'] for v in history]==[1,2] and all(v['kind']=='engineering' for v in history)
assert len(ledger)==1204 and [v['physical'] for v in ledger]==list(range(1,1205))
assert [v['cumulative_physical'] for v in ledger]==list(range(3,1207))
assert root['costs']==dict(diagnostic=1200,engineering=4) and root['cumulative_actual_updates']==1206 and root['prior_actual_updates']==2
for kind,n in [('diagnostic',1200),('engineering',4)]:assert [v['kind_physical'] for v in ledger if v['kind']==kind]==list(range(1,n+1))
source=cp(origin/'L3'/'final.pt');oldids=set(source['trainer']['optimizer_state']['state'])
assert load(r/'RESTORE.json')['status']=='PASS'
initial=cp(r/'K'/'initial.pt');assert same(initial['model'],source['model']) and same(initial['trainer'],source['trainer']) and same(initial['rng'],source['rng'])
assert initial['population']==dict(labeled=2,unlabeled=0) and initial['cursor']==1800
assert same(cp(r/'K'/'final.pt'),cp(r/'T'/'initial.pt'))
out=dict(status='AUDITED_COMPLETE',training_code='ea060b1',new_actual_updates=1204,prior_actual_updates=2,cumulative_actual_updates=1206,formal_updates=1200,engineering_updates_new=4,engineering_updates_cumulative=6,ledger_continuous=True,exit_code=0,duration_seconds=root['finished']-root['started'],source_restored_exactly=True,K_to_T_complete_state_equal=True,calls=root['calls'],stages={})
for stage in ['K','T']:
 p=r/stage;final=cp(p/'final.pt');latest=cp(p/'latest.pt');assert same(final,latest)
 assert final['cursor']==(2400 if stage=='K' else 3000) and final['population']==dict(labeled=2,unlabeled=0)
 for kind,n in [('diagnostic',600),('engineering',2)]:assert [v['local'] for v in ledger if v['stage']==stage and v['kind']==kind]==list(range(1,n+1))
 assert all(torch.equal(v,final['model'][k]) for k,v in source['model'].items() if 'prior_' in k or 'initial_' in k)
 states=final['trainer']['optimizer_state']['state'];genids=set(cp(r/'K'/'final.pt')['trainer']['optimizer_state']['state'])-oldids
 assert len(oldids)==40 and len(genids)==24
 assert all(int(states[k]['step'])==(1800 if stage=='K' else 2400) for k in oldids)
 assert all(int(states[k]['step'])==(600 if stage=='K' else 1200) for k in genids)
 nuisance=set(states)-oldids-genids
 assert len(nuisance)==(0 if stage=='K' else 16) and all(int(states[k]['step'])==600 for k in nuisance)
 if stage=='K':
  assert all(torch.equal(v,final['model'][k]) for k,v in source['model'].items() if not k.startswith(('anatomy.','structure.')))
  assert all(same(v,states[k]) for k,v in source['trainer']['optimizer_state']['state'].items())
  assert load(p/'FROZEN_AUDIT.json')['status']=='PASS'
 assert final['trainer']['scheduler_state']['last_epoch']==final['cursor']
 assert final['trainer']['scheduler_state']['_last_lr']==[3e-4] and all(g['lr']==3e-4 for g in final['trainer']['optimizer_state']['param_groups'])
 q=load(p/'QUALIFICATION.json');comp=load(p/'GOLDEN_COMPARISON.json');assert q['status']=='PASS' and all(comp[k] for k in ['loss_equal','model_equal','trainer_equal']) and all(comp['rng_equal'].values())
 metrics=rows(p/'metrics.jsonl');assert [v['added_step'] for v in metrics]==[0,25,100,300,600]
 if stage=='K':assert all(all(v[k]==metrics[0][k] for k in metrics[0] if k not in ('added_step','cumulative_step')) for v in metrics)
 res=load(p/'RESULTS.json');assert res==root['results'][stage] and res['main_pass']==all(v>=.8 for row in res['stochastic_dice'] for v in row)
 di=load(p/'DENSITY_INITIAL.json');df=load(p/'DENSITY_FINAL.json')
 if stage=='K':
  assert all(di[k]==df[k] for k in ['code','appearance','image','labels'])
  # logq is recovered via sum(terms)-bound in FP32; generator-dependent cancellation rounds.
  assert abs(di['logq']-df['logq'])<2e-6
 sw=load(p/'POSTERIOR_SWAP.json');raw=load(p/'PRIVATE_SWAP_SCORES.json');assert sw==root['swaps'][stage]
 assert sw['calls']['proposal_images']==4 and sw['calls']['density_images']==384 and sw['calls']['generator_draws']==16
 for group,sl in [('train',slice(0,2)),('heldout',slice(2,4))]:
  for component in ['shape','logits','total']:
   for family,ix in [('spatial',[1,2,3]),('identity',[4,5])]:
    margins={}
    for key in ['q0','qJ']:
     a=torch.tensor(raw[key][component],dtype=torch.float32)[:,sl,:];assert a.shape==(8,2,6) and torch.isfinite(a).all()
     margins[key]=a[:,:,0]-a[:,:,ix].mean(-1)
     assert abs(float(margins[key].mean())-sw['groups'][group][component][family+'_'+key]['mean'])<1e-6
    assert abs(float((margins['qJ']-margins['q0']).mean())-sw['groups'][group][component][family+'_change']['mean'])<1e-6
 out['stages'][stage]=dict(result=aggregate(res),metrics=[aggregate(v) for v in metrics],density_initial=di,density_final=df,posterior_swap=sw,qualification=q,golden=comp,final_cursor=final['cursor'],adam_state_counts=dict(original=40,generator=24,nuisance=len(nuisance)),prior_unchanged=True,latest_final_full_equal=True,telemetry=rows(p/'telemetry.jsonl'))
print(json.dumps(out,indent=2))
