"""Zero-training fixture includes native-source and native-baseline receipt schemas."""
import tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.lcrseg.dpr_sign_replication_v0_1 import contract as ct,core as c,results as r,execute as ex

def run():
 p=ct.protocol();docs={};tables={};source='FIXTURE';students={};sb={}
 def read(path):return docs[str(path)] if str(path) in docs else json.loads(Path(path).read_text())
 def put(path,obj):docs[str(path)]=obj
 with tempfile.TemporaryDirectory() as temp:
  b=Path(temp)
  for t in p['tasks']:
   root=b/'tasks'/t['task_id'];root.mkdir(parents=True);is_source=t['arm']=='SRC_CE';is_sign=t['arm']==c.MAIN
   counts=dict(optimizer_steps=t['updates'],backward=t['updates'],ema_updates=t['updates'],response_VJP=int(1.6*t['updates']) if is_sign else 0)
   bound=dict(parent_student_hash=None) if is_source else dict(source_student_hash='sourcehash')
   rr=dict(source=source,status='TRAINING_COMPLETE',task=t,updates=t['updates'],student_hash='sourcehash' if is_source else 'targethash',counts=counts,boundary=bound,U_opens=80*c.COUNTS[t['domain']][1] if is_sign else 0,L_opens=2*t['updates'],models=2,memory={'full_models':2},fixed_subset_unchanged=True,EMA_updates=t['updates'],seconds=0,label_order_hash='same')
   put(root/'receipt.json',rr);put(root/'evaluation/receipt.json',dict(source=source,student_hash=rr['student_hash'],models=1));put(root/'warmup.json',dict(same=True));put(root/f"diagnostics_pass_{t['updates']}.json",dict(status='PASS'))
   (root/f"checkpoint_{t['updates']}.pt").write_bytes(b'fixture');(root/'deploy_student.pt').write_bytes(b'fixture')
   if is_source:sb[t['task_id']]=dict(student_hash='sourcehash')
   else:
    students[t['task_id']]=dict(student_hash='targethash')
    with (root/'steps.jsonl').open('w') as f:
     for pos in range(1,t['updates']+1):
      active=is_sign and pos>t['steps_per_epoch']*20
      z=dict(position=pos,pseudo_labels=0,checks={'fixture':True},lambda_response=1 if active else 0,candidate_accepted=1,guard_rejected=0,d0_norm=1.,eta_norm=.01,correction_ratio=.01,b_norm=.1,raw_response_norm=1.,delta_norm=.1,raw_non_descent=0,pooled_raw_RMS=1.,pooled_candidate_RMS=.9,pooled_applied_RMS=.9)
      f.write(json.dumps(z)+'\n')
   for phase in ('train','eval'):
    put(b/'logs'/(t['task_id']+'_'+phase+'_exit.json'),dict(exit_code=0))
    oc=dict(optimizer_steps=t['updates'],backward=t['updates'],ema_updates=t['updates'],autograd_grad=counts['response_VJP'],sample_train_labeled=2*t['updates'],sample_train_unlabeled=rr['U_opens']) if phase=='train' else dict(sample_val=c.COUNTS[t['domain']][2] if is_source else 65)
    put(b/'tasks'/(t['task_id']+('_operations' if phase=='train' else '_eval_operations'))/'operation_counts.json',dict(status='PASS',counts=oc))
  put(b/'TARGET_WEIGHT_SEAL.json',dict(source=source,students=students));put(b/'SOURCE_BINDING.json',dict(sources=sb));put(b/'qualification_ledger.json',dict(synthetic=True))
  # Four frozen score columns: disc,rim,cup,macro.
  def score(path,domains):
   tid=Path(path).parent.parent.name;t=next(t for t in p['tasks'] if t['task_id']==tid)
   from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import METRICS
   return {d:np.full((c.COUNTS[d][2],len(METRICS)),.7 if t['arm']=='SRC_CE' else .61 if t['arm']==c.MAIN else .6) for d in domains}
  with patch.object(ct,'verify',return_value=source),patch.object(ct,'read',side_effect=read),patch.object(r,'scores',side_effect=score):
   terminal=r.finish(b);assert terminal['scientific_state']=='POSITIVE_REPLICATION_ESTIMATE' and abs(terminal['primary']['mean_delta']['Final']-.01)<1e-12
  for n in ('qualification_local/qualification.json','qualification_server/qualification.json','smoke/receipt.json'):put(b/n,dict(status='PASS',source=source,admission_mib=1))
  put(b/'qualification_ledger.json',dict(real_smoke_updates=8));dispatched=[]
  with patch.object(ct,'nas',side_effect=Path),patch.object(ct,'verify',return_value=source),patch.object(ct,'read',side_effect=read),patch.object(ct,'sha256',return_value='fixture'),patch.object(ct,'neutral_subprocess_paths'),patch.object(ex.subprocess,'check_output',return_value='99999'),patch.object(ex,'child',side_effect=lambda b,tid,phase,*args:dispatched.append((tid,phase))),patch.object(r,'finish',return_value={'fixture':True}):ex.execute(b,'fixture','fixture')
  assert len(dispatched)==36 and len(set(dispatched))==36
 for v,label in [(-.01,'NO_POSITIVE_REPLICATION_ESTIMATE'),(0,'NO_POSITIVE_REPLICATION_ESTIMATE'),(.001,'POSITIVE_REPLICATION_ESTIMATE')]:assert r.decision(np.full((2,3,4),v),dict(crosses_zero=True))['scientific_state']==label
 return dict(status='PASS',optimizer_updates=0,model_forwards=0,dispatches=36,report_schemas='source plus native baseline plus SIGN',primary_thresholds=True)

if __name__=='__main__':print(run())
