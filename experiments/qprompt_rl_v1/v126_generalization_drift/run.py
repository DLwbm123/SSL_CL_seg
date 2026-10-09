"""Read-only source/entry/continuation fit-versus-held diagnostics."""
import fcntl
import importlib.util
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def configure(cfg):
    global H,N,c,b,A
    spec=importlib.util.spec_from_file_location('prior',str(Path(cfg['v124'])/'run.py'));H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H);H.configure(cfg)
    N=H.N;c=H.c;b=H.b;A=H.A;N.STAGE='V126'


def measure(p,y):
    valid=y!=255
    assert valid.any() and torch.isfinite(p).all()
    dice=[float((2*(p[k][valid]*(y[valid]==k)).sum()+1)/(p[k][valid].sum()+(y[valid]==k).sum()+1)) for k in (1,2)]
    ce=float(-p.permute(1,2,0)[valid].gather(1,y[valid][:,None].long()).clamp_min(1e-8).log().mean())
    return dict(macro=float(np.mean(dice)),rim=dice[0],cup=dice[1],ce=ce)


def selfcheck():
    y=torch.tensor([[0,1],[2,255]]);p=torch.nn.functional.one_hot(y.clamp_max(2),3).permute(2,0,1).float()
    assert measure(p,y)==dict(macro=1.,rim=1.,cup=1.,ce=0.)
    wrong=measure(p.roll(1,0),y);assert wrong['macro']<1 and wrong['ce']>1
    return dict(status='PASS',checks=['perfect and wrong prediction','void excluded'],model_forwards=0)


def worker(root,cfg):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    roles=c.split_roles(cfg['data']);assert roles==N.D.read(Path(cfg['action_root'])/'ROLES.private.json')
    ctx=A.contexts()[cfg['context_index']];stream=cfg['stream'];r=H.fold_roles(roles,ctx['fold'],ctx['labeled_images']);condition=(ctx['condition'],ctx['factor'])
    ledger=b.JobLedger(root,{});counts=Counter();allowed=set()
    sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role=='train_labeled' and ds.rows[i]['case_id'] in allowed
        counts['image_reads']+=1;return sample(ds,i)
    c.e.primitive.CurrentData.__getitem__=guarded
    t=b.make(cfg,r,ledger,(ctx['source_step'],ctx['labeled_images'],condition,'heldout'),300)
    t.provider.seed=10168+10000*stream
    source=c.snapshot(t)
    snapshots=[('SOURCE',0,None),('ENTRY',100,Path(cfg['v123'])/f"jobs/entry{ctx['context']}_{stream}/ENTRY100.private.pt")]
    snapshots += [(mode,at,Path(cfg['v124'])/f"jobs/train{ctx['context']}_{stream}/{mode}_{at}.private.pt") for mode in ('BASE','OFF') for at in (150,300)]
    rows=[];reference=N.D.read(Path(cfg['v124'])/'RESULTS.json')
    ref=[x for x in reference if x['context']==ctx['context'] and x['stream']==stream]
    for mode,at,path in snapshots:
        state=source if path is None else torch.load(path,map_location='cpu',weights_only=False);c.restore(t,state)
        for role,ids,cond in [('fit',r['A_fit'],condition),('held',r['A_hold'],condition),('old',r['Q_train_old'],('identity',1.))]:
            allowed=set(ids);ds=c.subset(c.e.primitive.CurrentData(cfg['data'],0,'train_labeled'),ids);values=[]
            with t.readonly(),torch.no_grad():
                for i in range(len(ds)):
                    item=ds[i];x=c.photo(item['image'][None].cuda(),cond);y=item['label'].cuda()
                    counts['forward_attempts']+=1;c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='attempt',ordinal=counts['forward_attempts'],mode=mode,step=at,role=role,image_index=i))
                    p=t.clean(x)[0];v=measure(p,y);v['supervised_loss']=float(t.model.parent.supervised(p[None].clamp_min(1e-8).log(),y[None]));values.append(v)
                    counts['forward_success']+=1;c.e.append(root/'FORWARD_LEDGER.jsonl',dict(event='success',ordinal=counts['forward_success']))
            assert c.e.same(state,c.snapshot(t)), 'readout mutated state'
            row=dict(ctx,stream=stream,mode=mode,step=at,role=role,images=len(ids),**{k:float(np.mean([v[k] for v in values])) for k in values[0]})
            if mode!='SOURCE' and role!='fit':
                metric='new' if role=='held' else 'old'
                expected=ref[0][metric+'_entry'] if mode=='ENTRY' else next(v for v in ref if v['method']==mode and v['step']==at)[metric]
                assert abs(row['macro']-expected)<1e-7, (mode,role,row['macro'],expected)
            rows.append(row)
    assert not ledger.count and counts['image_reads']==counts['forward_attempts']==counts['forward_success']==6*(ctx['labeled_images']+8)
    N.D.write(root/'RESULTS.json',rows);N.D.write(root/'COUNTS.json',dict(counts));N.D.write(root/'FINAL.json',dict(status='COMPLETE',physical={},readout_state_unchanged=True,prior_scores_match=True,time=time.time()))


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert N.D.read(Path(cfg['v124'])/'COMPLETION_AUDIT.json')['status']=='PASS'
    (root/'jobs').mkdir();(root/'PHYSICAL_LEDGER.jsonl').touch()
    jobs=[dict(id=f'diagnose{i}_{s}',job='diagnose',context_index=i,stream=s,caps={}) for i in range(16) for s in (3,4)]
    account=Accounting(root,caps={},jobs=jobs);N.schedule(root,cfg,jobs,account)
    rows=sum((N.D.read(root/'jobs'/j['id']/'RESULTS.json') for j in jobs),[]);counts=Counter()
    for j in jobs:counts.update(N.D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert len(rows)==576 and counts['forward_success']==counts['forward_attempts']==counts['image_reads']==2112 and not account.count
    N.D.write(root/'RESULTS.json',rows);N.D.table(root/'RESULTS.csv',rows)
    N.D.write(root/'COSTS.json',dict(native_updates=0,actor_updates=0,linear_solves=0,new_annotation=0,Q_dev=0,**dict(counts)))
    N.D.write(root/'FINAL.json',dict(status='COMPLETE',rows=576,readout_state_unchanged=True,prior_scores_match=True,time=time.time()));N.B.write(root/'STATUS.json',N.D.read(root/'FINAL.json'))


if __name__=='__main__':
    root=Path(os.environ['EXEC_RUN']);cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());configure(cfg)
    if os.environ.get('EXEC_SELFCHECK')=='1':print(json.dumps(selfcheck()))
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':worker(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
