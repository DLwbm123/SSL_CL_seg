"""Read-only training evidence audit; write anonymous closeout aggregates only."""
import os,json,csv,io,math,collections,time
from pathlib import Path
import torch

def main():
 r=Path(os.environ['EXEC_RUN']);out=r/'reports';plan=json.loads((Path(__file__).with_name('EXECUTION_PLAN.json')).read_text());specs={s['id']:s for s in plan['R3b_jobs']};read=lambda p:json.loads(p.read_text());events=lambda p:[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
 def write(name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
 def table(name,rows):
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=sorted({k for x in rows for k in x}),lineterminator='\n');w.writeheader();w.writerows(rows)
 def finite(x):
  if torch.is_tensor(x):return bool(torch.isfinite(x).all())
  if isinstance(x,dict):return all(finite(v) for v in x.values())
  if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
  return True
 torch.set_num_threads(2);audits=[];costs=[];grants=events(r/'BUDGET_EVENTS.jsonl');counts=collections.Counter(x['category'] for x in grants);assert dict(counts)=={k:v for k,v in read(r/'BUDGET.json').items() if v}
 for task,spec in specs.items():
  f=r/'tasks'/task;s=torch.load(f/'final.pt',map_location='cpu',weights_only=False);e=read(f/'EVALUATION.json');ev=events(f/'TRANSACTIONS.jsonl');commits=[x for x in ev if x['event']=='student_commit'];retained=[x for x in commits if not x['temporary']];controller=sum(x['event']=='optimizer_success' and x['kind']=='controller' for x in ev)
  assert s['t']==s['scheduler']['t']==1200 and s['retained']==spec['retained_new_student_updates'] and e['complete'];assert len({x['transaction'] for x in retained})==s['retained'];assert all(int(v['step'])==2000+s['retained'] for v in s['optimizer']['state'].values());assert all(g['lr']==0 for g in s['optimizer']['param_groups']);assert finite({k:s[k] for k in ('student','optimizer','teacher','policy','policy_reference','policy_optimizer')});assert s['rng']['cpu'].dtype==torch.uint8 and len(s['rng']['cuda'])==1
  assert all(e[k]==s[k] for k in ('code_commit','config_sha','prefix_sha','schedule_sha','controller_version'));binding=read(r/'PREFIX_BINDINGS.private.json')[spec['dependencies'][0]];assert binding['sha256']==s['prefix_sha'] and binding['source_commit']==s['prefix_source_commit']
  if s['policy_optimizer'] is not None:assert all(int(v['step'])==controller for v in s['policy_optimizer']['state'].values())
  else:assert controller==0
  audits.append(dict(task=task,passed=True,ordinary=1200,retained=s['retained'],optimizer_step=2000+s['retained'],controller_calls=controller,training_commit=s['code_commit'],recovery_parent_commit=s.get('recovery_parent_commit'),config_sha=s['config_sha'],prefix_sha=s['prefix_sha'],schedule_sha=s['schedule_sha'],checkpoint_sha=e['checkpoint_sha'],finite=True,RNG_present=True,student_only_evaluation=True));del s
 write('FINAL_STATE_AUDIT.json',audits)
 folders=[(p,'valid' if p.name in specs or p.name.startswith('R3A') else 'qualification') for p in (r/'tasks').iterdir() if p.is_dir()]+[(p,'invalidated_P1') for p in (r/'invalidated/P1').iterdir() if p.is_dir() and p.name.startswith(('R3A','R3B'))]
 for f,role in folders:
  ev=events(f/'TRANSACTIONS.jsonl');cost=events(f/'COST_EVENTS.jsonl');attempt=[x for x in ev if x['event']=='attempt'];success=[x for x in ev if x['event']=='optimizer_success'];commits=[x for x in ev if x['event']=='student_commit'];label=collections.Counter()
  for x in cost:label[x['purpose']]+=x['images']
  costs.append(dict(task=f.name,evidence_role=role,attempts=len(attempt),successes=len(success),failed=sum(x['event']=='optimizer_failed' for x in ev),student_calls=sum(x['kind']!='controller' and x['kind']!='synthetic_controller' for x in success),controller_calls=sum(x['kind'] in ('controller','synthetic_controller') for x in success),retained_physical=sum(not x['temporary'] for x in commits),disposable_physical=sum(x['temporary'] for x in commits),student_forwards=sum(x['student_forwards'] for x in cost),teacher_forwards=sum(x['teacher_forwards'] for x in cost),**{'label_images_'+k:v for k,v in label.items()}))
 assert sum(x['attempts'] for x in costs)==len(grants)==sum(x['successes'] for x in costs)
 table('ALL_PHYSICAL_COSTS.csv',costs);write('TRANSACTION_AUDIT.json',dict(categories=dict(counts),caps=dict(r3a_student=320,r3b_student=144000,controller=8640,student_replay=14432,controller_replay=864,synthetic_student=64,synthetic_controller=128,smoke=16),grants=len(grants),optimizer_successes=sum(x['successes'] for x in costs),optimizer_failures=sum(x['failed'] for x in costs),unique_valid_retained=sum(x['retained'] for x in audits),unique_valid_disposable=320+10800,valid_controller_calls=sum(x['controller_calls'] for x in audits),invalidated_student_calls=sum(x['student_calls'] for x in costs if x['evidence_role']=='invalidated_P1'),invalidated_controller_calls=sum(x['controller_calls'] for x in costs if x['evidence_role']=='invalidated_P1'),all_student_calls=sum(x['student_calls'] for x in costs),all_controller_calls=sum(x['controller_calls'] for x in costs),note='All current, qualification and invalidated P1 transaction streams included. Physical grants are not unique valid updates. No optimizer failures; the two pre-repair evaluator assertions are non-optimizer failures. Per-step costs include repeated forwards; qualification images mix synthetic and real L smoke.'))
 rows=[]
 for task,spec in specs.items():
  f=r/'tasks'/task;ds=[read(p) for p in sorted((f/'decisions').glob('*.private.json'))];a=events(f/'ACTION_LOG.jsonl');ds=[d for d in ds if d.get('feedback')];values=collections.defaultdict(list)
  for d in ds:
   b=d['feedback'];values['raw_reward']+=b['raw_rewards'];values['online_audit_sign_agreement'] += [float((p['online']-b['anchor']['online']>0)==(p['audit']-b['anchor']['audit']>0)) for p in b['candidates']]
   values['entropy'].append(-sum(p*math.log(p) for p in b['behavior']));values['skip_probability'].append(b['behavior'][0]);values['coarse_probability'].append(b['behavior'][1]);values['fine_probability'].append(b['behavior'][2]);values['clip_fraction'] += [x['clip_fraction'] for x in b['controller']];values['KL'] += [x['kl'] for x in b['controller']]
   for p in b['candidates']:
    for k,v in p['info']['coverage'].items():values['coverage_action_'+k].append(v)
    if p['info']['U_grad_norm'] is not None:values['U_gradient_norm'].append(p['info']['U_grad_norm'])
   if d['state']:values['fine_teacher_coverage'].append(d['state'][11]);values['coarse_teacher_coverage'].append(d['state'][12])
  if ds:
   rows.append(dict(task=task,seed=spec['seed'],backbone=spec['backbone'],domain=spec['domain'],arm=spec['arm'],decisions=len(ds),**{k:sum(v)/len(v) for k,v in values.items()},early_reward=sum(x for d in ds[:10] for x in d['feedback']['raw_rewards'])/40,late_reward=sum(x for d in ds[-10:] for x in d['feedback']['raw_rewards'])/40))
 table('POLICY_AND_FEEDBACK_SUMMARY.csv',rows)
 results=list(csv.DictReader((out/'RESULTS.csv').open()));deltas=list(csv.DictReader((out/'PAIRED_DELTAS.csv').open()));summ=[]
 for seed in ('261','262','263','fresh262263'):
  for contrast in sorted({x['contrast'] for x in deltas}):
   for metric in ('macro','rim','cup'):
    v=[float(x['delta']) for x in deltas if x['contrast']==contrast and x['metric']==metric and (x['seed'] in ('262','263') if seed=='fresh262263' else x['seed']==seed)]
    summ.append(dict(seed_group=seed,contrast=contrast,metric=metric,cells=len(v),mean_delta=sum(v)/len(v),mean_pp=100*sum(v)/len(v),positive_cells=sum(x>0 for x in v),worst_delta=min(v)))
 table('CONTRAST_SUMMARY.csv',summ)
 stats={};seen=set()
 for p in r.rglob('*'):
  if p.is_file() and not p.is_symlink():
   s=p.stat();key=(s.st_dev,s.st_ino)
   if key not in seen:seen.add(key);stats['unique_bytes']=stats.get('unique_bytes',0)+s.st_size
 session=read(r/'SESSION.json');final=read(r/'FINAL.json');write('RESOURCE_SUMMARY.json',dict(wall_seconds=final['finished']-session['start'],**stats,peak_qualification_reserved_bytes={p.parent.name:read(p)['peak_reserved'] for p in (r/'qualification').glob('*/PASSED.json')},limits='No continuous GPU power or utilization sampling was enabled. Worker residence time is not GPU busy time. Storage counts include code, raw logs, invalidated states and current checkpoints.'))
 print(json.dumps(dict(state_audits=len(audits),budget=read(out/'TRANSACTION_AUDIT.json'),contrasts=[x for x in summ if x['seed_group']=='fresh262263' and x['metric']=='macro'],resources=read(out/'RESOURCE_SUMMARY.json')),indent=2))
if __name__=='__main__':main()
