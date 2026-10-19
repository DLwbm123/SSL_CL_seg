"""Read-only independent ledger/state/metric audit for the authorized restart."""
import json,torch
from pathlib import Path
from sc_bayeseg.run import same
import os
settings=json.loads(Path(os.environ['TASK_SETTINGS']).read_text())
r=Path(settings['output']);origin=Path(settings['origin'])
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
ledger=rows(r/'physical_updates.jsonl');history=rows(origin/'physical_updates.jsonl')
assert len(history)==1204 and [v['cumulative_physical'] for v in history]==list(range(3,1207))
assert len(ledger)==1204 and [v['physical'] for v in ledger]==list(range(1,1205))
assert [v['cumulative_physical'] for v in ledger]==list(range(1207,2411))
assert root['costs']==dict(diagnostic=1200,engineering=4) and root['cumulative_actual_updates']==2410 and root['prior_actual_updates']==1206
for kind,n in [('diagnostic',1200),('engineering',4)]:assert [v['kind_physical'] for v in ledger if v['kind']==kind]==list(range(1,n+1))
source=cp(origin/'K'/'final.pt');converted=cp(r/'converted_initial.pt')
assert all(torch.equal(v,converted['model'][k]) for k,v in source['model'].items())
assert same(source['trainer']['optimizer_state']['state'],converted['trainer']['optimizer_state']['state'])
assert same(source['trainer']['scheduler_state'],converted['trainer']['scheduler_state']) and same(source['rng'],converted['rng'])
assert converted['cursor']==2400 and converted['population']==dict(labeled=2,unlabeled=0)
assert all(torch.equal(v,converted['model']['physical_encoder.'+k[len('shape_encoder.'):]]) for k,v in source['model'].items() if k.startswith('shape_encoder.'))
assert same(converted,cp(r/'R'/'initial.pt')) and same(cp(r/'R'/'final.pt'),cp(r/'J'/'initial.pt'))
zero=load(r/'ZERO_AUDIT.json');assert zero['status']=='PASS' and zero['optimizer_updates']==0 and zero['extra_reverse_calls']==2
for key in ['deterministic_logits_equal','all_six_stochastic_dice_pass','original_model_parameters_equal','original_adam_states_equal','new_physical_adam_empty','parameter_order_preserved','parameter_and_storage_independent','four_node_logq_equal','image_semantic_gradient_zero','complete_state_unchanged']:assert zero[key]
from sc_bayeseg.split_v1.model import Model,semantic
model=Model();names=[n for n,_ in model.named_parameters()]
semids={i for i,n in enumerate(names) if semantic(n)};genids={i for i,n in enumerate(names) if n.startswith(('anatomy.','structure.'))};physids={i for i,n in enumerate(names) if n.startswith('physical_encoder.')};auxids=set(range(len(names)))-semids-genids-physids
assert (len(semids),len(genids),len(physids),len(auxids))==(40,24,20,16)
assert not physids.intersection(converted['trainer']['optimizer_state']['state'])
anchor=cp(r/'diagnostic_anchor.pt');Rfinal=cp(r/'R'/'final.pt')
assert all(torch.equal(v,anchor['model'][k]) for k,v in Rfinal['model'].items() if not k.startswith(('anatomy.','structure.')))
assert all(torch.equal(v,anchor['model'][k]) for k,v in source['model'].items() if k.startswith(('anatomy.','structure.')))
zero_public=dict(zero);zero_public['stochastic_baseline']=aggregate(zero_public['stochastic_baseline'])
out=dict(status='AUDITED_COMPLETE',training_code='e911923',new_actual_updates=1204,prior_actual_updates=1206,cumulative_actual_updates=2410,formal_updates=1200,engineering_updates=4,ledger_continuous=True,exit_code=0,duration_seconds=root['finished']-root['started'],K_restored_and_converted_exactly=True,R_to_J_complete_state_equal=True,calls=root['calls'],zero_audit=zero_public,stages={})
base=zero_public['stochastic_baseline']['stochastic_dice_macro_mean']
for stage in ['R','J']:
 p=r/stage;final=cp(p/'final.pt');initial=cp(p/'initial.pt');assert same(final,cp(p/'latest.pt'))
 assert final['cursor']==(3000 if stage=='R' else 3600) and final['population']==dict(labeled=2,unlabeled=0)
 for kind,n in [('diagnostic',600),('engineering',2)]:assert [v['local'] for v in ledger if v['stage']==stage and v['kind']==kind]==list(range(1,n+1))
 assert all(torch.equal(v,final['model'][k]) for k,v in source['model'].items() if 'prior_' in k or 'initial_' in k)
 states=final['trainer']['optimizer_state']['state'];assert len(states)==100
 expected=dict(semantic=1800 if stage=='R' else 2400,generator=1200 if stage=='R' else 1800,physical=600 if stage=='R' else 1200,auxiliary=600 if stage=='R' else 1200)
 for label,ids in [('semantic',semids),('generator',genids),('physical',physids),('auxiliary',auxids)]:assert all(int(states[i]['step'])==expected[label] for i in ids)
 changed=any(not torch.equal(v,final['model'][k]) for k,v in initial['model'].items() if semantic(k))
 if stage=='R':
  assert not changed and all(same(source['trainer']['optimizer_state']['state'][i],states[i]) for i in semids)
  assert load(p/'FROZEN_AUDIT.json')['status']=='PASS'
 else:assert changed
 assert final['trainer']['scheduler_state']['last_epoch']==final['cursor'] and final['trainer']['scheduler_state']['_last_lr']==[3e-4]
 assert all(g['lr']==3e-4 for g in final['trainer']['optimizer_state']['param_groups'])
 q=load(p/'QUALIFICATION.json');comp=load(p/'GOLDEN_COMPARISON.json');assert q['status']=='PASS' and all(comp[k] for k in ['loss_equal','model_equal','trainer_equal']) and all(comp['rng_equal'].values())
 metrics=rows(p/'metrics.jsonl');assert [v['added_step'] for v in metrics]==[0,25,100,300,600]
 if stage=='R':assert all(all(v[k]==metrics[0][k] for k in metrics[0] if k not in ('added_step','cumulative_step')) for v in metrics)
 res=load(p/'RESULTS.json');assert res==root['results'][stage] and res['main_pass']==all(v>=.8 for row in res['stochastic_dice'] for v in row)
 assert res['retention_pass']==(sum(map(sum,res['stochastic_dice']))/6>=base-.02)
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
 out['stages'][stage]=dict(result=aggregate(res),metrics=[aggregate(v) for v in metrics],density_initial=load(p/'DENSITY_INITIAL.json'),density_final=load(p/'DENSITY_FINAL.json'),posterior_swap=sw,qualification=q,golden=comp,final_cursor=final['cursor'],adam_steps=expected,semantic_parameters_changed=changed,prior_unchanged=True,latest_final_full_equal=True,telemetry=rows(p/'telemetry.jsonl'))
print(json.dumps(out,indent=2))
