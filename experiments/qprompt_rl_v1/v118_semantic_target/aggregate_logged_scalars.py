"""Read existing anonymous scalar ledgers; no model, image or query calls."""
from pathlib import Path
import json,statistics,os
root=Path(os.environ['EXEC_RUN'])
rows=[]
for i in range(8):
 for s in (3,4):
  p=root/f'jobs/train{i}_{s}'
  selection={ (r['key'],r['step']):r for r in (json.loads(x) for x in (p/'SELECTION_LEDGER.jsonl').read_text().splitlines()) }
  for line in (p/'PROTOTYPE_LEDGER.jsonl').read_text().splitlines():
   r=json.loads(line)
   if r['event']!='success':continue
   q=selection[r['key'],r['step']]
   rows.append(dict(context=i,stream=s,method=r['mode'],step=r['step'],selected=r['selected'],changed=r['changed'],support=r['support'],present=r['present'],target_class_counts=r['target_class_counts'],original_EMA_selected_class_counts=q['classes']))
def agg(a):
 sel=sum(r['selected'] for r in a)
 return dict(updates=len(a),selected_pixels=sel,changed_pixels=sum(r['changed'] for r in a),changed_fraction=sum(r['changed'] for r in a)/sel,missing_support_updates=[sum(r['support'][k]==0 for r in a) for k in range(3)],no_prototype_updates=sum(not r['present'] for r in a),support_min=[min(r['support'][k] for r in a) for k in range(3)],support_median=[statistics.median(r['support'][k] for r in a) for k in range(3)],target_class_fractions=[sum(r['target_class_counts'][k] for r in a)/sel for k in range(3)],EMA_selection_class_fractions=[sum(r['original_EMA_selected_class_counts'][k] for r in a)/sel for k in range(3)])
out=dict(scope='existing scalar ledgers only; no model/image/query/fit',methods=[dict(method=m,**agg([r for r in rows if r['method']==m])) for m in ('BASE','SEMANTIC','ROTATED')],contexts=[dict(context=i,method=m,**agg([r for r in rows if r['method']==m and r['context']==i])) for i in range(8) for m in ('BASE','SEMANTIC','ROTATED')],windows=[dict(method=m,start=t,**agg([r for r in rows if r['method']==m and t<=r['step']<t+50])) for m in ('BASE','SEMANTIC','ROTATED') for t in (100,150,200,250)],first=[r for r in rows if r['step']==100])
assert len(rows)==9600 and len(out['first'])==48
print(json.dumps(out))
