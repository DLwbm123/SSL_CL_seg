"""Read-only scoring of saved training decision states; no new image/model training."""
from pathlib import Path
import importlib.util,json,os,time
import torch

r=Path(os.environ['EXEC_RUN']);torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('selector',r/'selector.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
rows=[];forwards=0;start=time.time()
for seed in (601,602):
 job=r/f'jobs/learn{seed}';saved=torch.load(job/'ACTORS.private.pt',map_location='cpu',weights_only=False);assert saved['groups']==32
 actors={}
 for arm in ('rl','ce'):
  actor=S.Actor(seed);actor.load_state_dict(saved[arm]);actor.eval();actors[arm]=actor
 stats={arm:[] for arm in actors};total=0
 for group in range(32):
  branches=torch.load(job/f'GROUP_{group:02d}_TRAJECTORIES.private.pt',map_location='cpu',weights_only=False)['branches'];assert len(branches)==4
  for branch in branches:
   x,legal=branch['x'],branch['legal'];total+=len(x)
   assert x.shape[1:]==(16,18) and legal.shape==x.shape[:2] and torch.isfinite(x).all()
   for k in range(0,len(x),1024):
    xx,ll=x[k:k+1024],legal[k:k+1024];n=ll.sum(-1);use=n>1
    if not use.any():continue
    xx,ll,n=xx[use],ll[use],n[use]
    uniform=ll.double()/n[:,None]
    for arm,actor in actors.items():
     with torch.no_grad():logits=actor(xx).double().masked_fill(~ll,-torch.inf);log=logits.log_softmax(-1);p=log.exp()
     forwards+=1;entropy=-(p*log.masked_fill(~ll,0.)).sum(-1);tv=(p-uniform).abs().sum(-1)/2;kl=n.double().log()-entropy
     assert torch.isfinite(entropy).all() and kl.min()>-1e-10
     stats[arm].append(torch.stack((entropy/n.double().log(),tv,kl,(p-uniform).abs().max(-1).values,p.max(-1).values*n),-1))
 for arm,values in stats.items():
  v=torch.cat(values);names=['normalized_entropy','total_variation_from_uniform','KL_from_uniform','max_absolute_probability_change','max_probability_over_uniform']
  row=dict(seed=seed,actor=arm,all_saved_training_decisions=total,multiple_legal_decisions=len(v),group_count=32,branch_count=128)
  for j,name in enumerate(names):row[name]=dict(mean=float(v[:,j].mean()),median=float(v[:,j].median()),p95=float(v[:,j].quantile(.95)),maximum=float(v[:,j].max()),minimum=float(v[:,j].min()))
  rows.append(row)
out=dict(status='COMPLETE',scope='Final actors scored on every saved same-seed Q_train collection decision state, all 32 groups x 4 branches; post-hoc training-state diagnostic, not deployment policy audit or independent test',image_reads=0,segmentation_forwards=0,optimizer_updates=0,new_query_calls=0,actor_forward_calls=forwards,elapsed_seconds=time.time()-start,rows=rows)
print(json.dumps(out,allow_nan=False))
