"""Compare pooled versus per-state training advantage scale with matched CE and RL."""
import csv
import fcntl
import importlib.util
import json
import math
import os
import random
import runpy
import statistics as st
import subprocess
import time
import traceback
from pathlib import Path

ARMS = ('POOLED_CE', 'POOLED_RL')
NEW = tuple(f'{arm}_{seed}' for arm in ARMS for seed in (601, 602))
OLD = ('NATIVE','FIXED_BEST','UNIFORM_ACTION','TIME_FIXED')+tuple(
    f'{arm}_{seed}' for arm in ('Z_1024','LOCAL_CE','EPISODIC_CE','EPISODIC_RL','DENSE_CE','DENSE_RL','CONTINUE_CE','OLD_ANCHOR_RL','ALIGNED_ANCHOR_RL','FRESH_CE','FRESH_RL','HEAD_CE','HEAD_RL') for seed in (601,602))
CAPS = dict(qualification=8, actor_fit=4096, development=3200)


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def dense(gain, old, entry): return gain+old-entry


def summarize(rows):
    keys=('new','old','utility','gain','forget_penalty','absolute_forgetting')
    means={m:{k:st.mean(r[k]['macro'] if k in ('new','old') else r[k] for r in rows if r['method']==m)
              for k in keys} for m in OLD+NEW}
    for arm in ('Z_1024','LOCAL_CE','EPISODIC_CE','EPISODIC_RL','DENSE_CE','DENSE_RL','CONTINUE_CE','OLD_ANCHOR_RL','ALIGNED_ANCHOR_RL','FRESH_CE','FRESH_RL','HEAD_CE','HEAD_RL')+ARMS:
        means[arm]={k:st.mean(means[f'{arm}_{s}'][k] for s in (601,602)) for k in keys}
    controls=OLD[:4]+('Z_1024','LOCAL_CE','EPISODIC_CE','EPISODIC_RL','DENSE_CE','DENSE_RL','CONTINUE_CE','OLD_ANCHOR_RL','ALIGNED_ANCHOR_RL','FRESH_CE','FRESH_RL','HEAD_CE','HEAD_RL','POOLED_CE')
    delta={m:{k:means['POOLED_RL'][k]-means[m][k] for k in ('new','old','utility')} for m in controls}
    paired={f'{m}_{s}':means[f'POOLED_RL_{s}']['utility']-means[f'{m}_{s}']['utility']
            for m in ('POOLED_CE','FRESH_RL') for s in (601,602)}
    trade={m:(delta[m]['new']>=.002 and delta[m]['old']>=-.0025) or (delta[m]['old']>=.005 and delta[m]['new']>=-.0025)
           for m in ('UNIFORM_ACTION','CONTINUE_CE','FRESH_CE','POOLED_CE')}
    passed=all(v['utility']>=.0005 for v in delta.values()) and all(v>0 for v in paired.values()) and all(trade.values())
    return dict(status='PASS_V94_POOLED_ADVANTAGE_D1_ONLY' if passed else 'STOP_V94_NO_POOLED_ADVANTAGE_INCREMENTAL_GAIN',
                means=means,primary_deltas=delta,paired_seed_utility=paired,tradeoff=trade,V84='NOT_RUN',independent_confirmation=False)


def selfcheck():
    assert abs(dense(.01,.78,.80)+.01)<1e-12
    assert abs(dense(.01,.81,.80)-.02)<1e-12
    rows=[dict(method=m,new={'macro':.8+(.003 if m.startswith('POOLED_RL') else 0)},old={'macro':.8},
               utility=.003 if m.startswith('POOLED_RL') else 0,gain=0,forget_penalty=0,absolute_forgetting=0) for m in OLD+NEW]
    assert summarize(rows)['status'].startswith('PASS')
    assert summarize([dict(r,utility=0) for r in rows])['status'].startswith('STOP')
    assert summarize([dict(r,new={'macro':.8}) for r in rows])['status'].startswith('STOP')
    # Exact entropy/KL optimum: loss gap must equal temperature times KL.
    adv=[-1.,0.,1.];prior=[.6,.3,.1];q=[.2,.5,.3]
    raw=[math.exp((a+.25*math.log(p))/.26) for a,p in zip(adv,prior)];opt=[v/sum(raw) for v in raw]
    def loss(prob):return sum(p*(-a+.25*math.log(p/r)+.01*math.log(p)) for p,a,r in zip(prob,adv,prior))
    gap=.26*sum(p*math.log(p/r) for p,r in zip(q,opt))
    assert abs(loss(q)-loss(opt)-gap)<1e-12 and gap>=0


def gpu_job(root,cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    roles=c.split_roles(cfg['data']);assert roles==read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger=b.JobLedger(root,cfg['caps']);queries=0;original=Path(cfg['original']);campaign=Path(cfg['campaign'])
    def query(t,role,condition):
        nonlocal queries
        queries+=len(roles[role]);event=dict(role=role,step=t.step,image_calls_upper_bound=queries)
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**event))
        try:value=scores(t,roles,role,condition)
        except BaseException:
            c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**event));raise
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**event));return value
    def restore(t,path,step):
        value=torch.load(path,map_location='cpu',weights_only=False)
        assert value['step']==step and value['options']['total_steps']==300
        c.restore(t,value);return value
    i=cfg['context_index']
    m,n,kind,factor=read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
    ctx=(m,n,(kind,factor),f'dev{i}');t=b.make(cfg,roles,ledger,ctx,300)
    entry=restore(t,original/f'jobs/development/{ctx[3]}/ENTRY.private.pt',100)
    if cfg['job']=='train':
        assert read(campaign/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
        for method in NEW:
            models=[B.load_model(campaign/f'jobs/fit/{method}.private.pt')];c.restore(t,entry)
            t.category='development';t.key=ctx[3]+'/'+method;decisions=[];generator=torch.Generator().manual_seed(860301+i)
            for _ in range(200):
                if t.step in (100,200):t.action,row=B.select(t,models,generator);decisions.append(row)
                t.update()
                if t.step==150:c.e.atomic_save(c.snapshot(t),root/(method+'_MID.private.pt'))
                if t.step%50==0:write(root/'STATUS.json',dict(status='RUNNING',method=method,step=t.step,physical=dict(ledger.count)))
            c.e.atomic_save(c.snapshot(t),root/(method+'_FINAL.private.pt'))
            write(root/(method+'_DECISIONS.private.json'),decisions)
            write(root/(method+'_TRAINING.json'),dict(status='SEALED',context=ctx[3],method=method,updates=200,step=300,actions=[r['action'] for r in decisions]))
        assert dict(ledger.count)=={'development':800} and queries==0
    elif cfg['job']=='evaluate':
        assert read(campaign/'ENDPOINT_LOCK.json')['endpoints']==136
        previous=[r for r in read(campaign/'REUSED_RESULTS.json') if r['context']==ctx[3]]
        assert len({r['new_entry'] for r in previous})==len({r['old_memory_reference'] for r in previous})==1
        entrynew,reference=previous[0]['new_entry'],previous[0]['old_memory_reference'];rows=[]
        for method in NEW:
            src=campaign/f'jobs/train{i}';restore(t,src/(method+'_MID.private.pt'),150);mid=query(t,'Q_dev_new',ctx[2])
            restore(t,src/(method+'_FINAL.private.pt'),300);new=query(t,'Q_dev_new',ctx[2]);old=query(t,'Q_dev_old',('identity',1.))
            value=B.reward(mid['macro'],new['macro'],entrynew,old['macro'],reference)
            rows.append(dict(context=ctx[3],method=method,source='NEW',new50=mid,new=new,old=old,new_entry=entrynew,
                old_memory_reference=reference,gain=value['gain'],forget_penalty=value['forget_penalty'],utility=value['reward'],
                absolute_forgetting=reference-old['macro'],actions=read(src/(method+'_TRAINING.json'))['actions']))
        assert not ledger.count and queries==48;write(root/'RESULTS.json',rows)
    else:raise ValueError('unknown GPU job')
    write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),query_image_evaluations=queries,time=time.time()))


def pooled_advantage(returns, floor):
    centered=returns-returns.mean(-1,keepdim=True)
    scale=centered.square().mean().sqrt().clamp_min(floor)
    return (centered/scale).clamp(-3,3),scale


def scalecheck():
    import torch
    x=torch.tensor([[-1.,0.,1.],[-.1,0.,.1]])
    a,scale=pooled_advantage(x,1e-6)
    assert torch.allclose(a[0],10*a[1])
    assert torch.allclose(pooled_advantage(x+torch.tensor([[2.],[-3.]]),1e-6)[0],a,atol=1e-6)
    assert torch.allclose(pooled_advantage(2*x,1e-6)[0],a)
    assert torch.isfinite(pooled_advantage(torch.ones_like(x),1e-6)[0]).all()
    assert scale>0


def fit_job(root,cfg):
    import torch
    torch.set_num_threads(2);scalecheck()
    assert read(Path(cfg['v89'])/'jobs/actor_qualification/QUALIFICATION.json')['status']=='PASS'
    rows=[r for i in range(8) for stream in (1,2) for r in read(Path(cfg['v92'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json')]
    keys=sorted({(r['context'],r['stream'],r['step']) for r in rows});groups=[]
    for key in keys:
        g=sorted((r for r in rows if (r['context'],r['stream'],r['step'])==key),key=lambda r:r['action'])
        assert [r['action'] for r in g]==list(range(9)) and all(r['state']==g[0]['state'] for r in g);groups.append(g)
    assert len(groups)==32 and len(rows)==288 and all(math.isfinite(r['dense_reward']) for r in rows)
    x=torch.tensor([g[0]['state'] for g in groups]);returns=torch.tensor([[r['dense_reward'] for r in g] for g in groups])
    adv,pooled_scale=pooled_advantage(returns,B.FLOOR)
    rawstd=returns.std(-1,unbiased=False);centered=returns-returns.mean(-1,keepdim=True)
    B.csvfile(root/'SCALE_DIAGNOSTIC.csv',[dict(context=k[0],stream=k[1],step=k[2],raw_std=float(rawstd[i]),old_scale=float(rawstd[i].clamp_min(B.FLOOR)),pooled_scale=float(pooled_scale),old_to_pooled_scale_ratio=float(rawstd[i].clamp_min(B.FLOOR)/pooled_scale),clipped_actions=int(((centered[i]/pooled_scale).abs()>3).sum())) for i,k in enumerate(keys)])
    write(root/'SCALE_QUALIFICATION.json',dict(status='PASS',formula='center per state; global RMS across all centered training action returns; floor then clamp',pooled_scale=float(pooled_scale),floor=B.FLOOR,training_states=32,actions=288,query_accesses=0,optimizer_updates=0,time=time.time()))
    budget=runpy.run_path(cfg['budget_helper'])['Budget'](root,dict(actor_fit=4096));logs=[];summary=[]
    for arm in ARMS:
        for seed in (601,602):
            actor,center,scale=B.load_model(Path(cfg['v90'])/f'jobs/fit/DENSE_CE_{seed}.private.pt');z=(x-center)/scale
            prior=actor(z).detach().log_softmax(-1)
            opt=torch.optim.Adam(actor.parameters(),lr=.001)
            for update in range(1024):
                loss=B.actor_loss(actor,z,adv,prior,arm=='POOLED_RL');assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.))
                budget.step('actor_fit',f'{arm}/{seed}/{update}',opt)
                if update+1 in (1,64,1024):logs.append(dict(arm=arm,seed=seed,update=update+1,objective_before_update=float(loss.detach())))
            with (root/f'{arm}_{seed}.private.pt').open('xb') as f:torch.save(dict(actor=actor.state_dict(),mean=center,scale=scale,seed=seed,arm=arm,updates=1024,source='V94'),f)
            with torch.no_grad():
                log=actor(z).log_softmax(-1);prob=log.exp()
                summary.append(dict(arm=arm,seed=seed,normalized_return=float((prob*adv).sum(-1).mean()),raw_dense_return=float((prob*returns).sum(-1).mean()),KL_to_reference=float((prob*(log-prior)).sum(-1).mean()),entropy=float(-(prob*log).sum(-1).mean())))
    assert dict(budget.count)=={'actor_fit':4096}
    B.csvfile(root/'FIT_LOG.csv',logs);B.csvfile(root/'FIT_SUMMARY.csv',summary)
    write(root/'FIT_QUALIFICATION.json',dict(status='PASS',trainable_parameters=1097,pooled_scale_selfcheck=True,finite_loss_and_gradients=True,extra_optimizer_updates=0))
    write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(budget.count),time=time.time()))


def schedule(root,cfg,jobs,account,cpu=False):
    pending=list(jobs);active={};failed=[];done=[]
    while pending or active:
        for slot,(p,spec) in list(active.items()):
            if p.poll() is None:continue
            dest=root/'jobs'/spec['id'];write(dest/'PROCESS_EXIT.json',dict(exit_code=p.returncode,time=time.time()))
            (failed if p.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in ((-1,) if cpu else (5,6,7)):
                if not pending:break
                if slot in active:continue
                if not cpu and int(subprocess.check_output(['nvidia-smi','--id='+str(slot),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12000:continue
                spec=pending.pop(0);dest=root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=slot))
                env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='' if cpu else str(slot),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    p=subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(dest/'LAUNCH.json',dict(pid=p.pid,gpu=slot,time=time.time(),commit=cfg['commit']));active[slot]=p,spec
        account.refresh()
        write(root/'COSTS.json',dict(scope='V94_NEW_ONLY',attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_cap=3208,actor_cap=4096))
        write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',active=[s['id'] for _,s in active.values()],pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=Path(cfg['v93']);assert read(previous/'FINAL.json')['status']=='COMPLETE'
    assert read(previous/'REUSE_AUDIT.json')['status']=='PASS'
    assert read(previous/'ENDPOINT_LOCK.json')['endpoints']==120
    rows=read(previous/'DEVELOPMENT_RESULTS.json')['rows'];assert len(rows)==len({(r['context'],r['method']) for r in rows})==120
    assert {r['method'] for r in rows}==set(OLD)
    for i in range(4):assert read(previous/f'jobs/eval{i}/PROCESS_EXIT.json')['exit_code']==0
    write(root/'REUSED_RESULTS.json',[dict(r,source='REUSED_V93') for r in rows])
    write(root/'REUSE_AUDIT.json',dict(status='PASS',endpoints=120,source='V93 complete sealed and evaluated endpoints, including audited V83/V87/V89/V90/V91/V92 controls',new_updates=0))
    fits=[dict(id='fit',job='fit',caps={})];quals=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    trains=[dict(id=f'train{i}',job='train',context_index=i,caps=dict(development=800)) for i in range(4)]
    evals=[dict(id=f'eval{i}',job='evaluate',context_index=i,caps={}) for i in range(4)]
    jobs=fits+quals+trains+evals;(root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account=Accounting(root,caps=CAPS,jobs=jobs);schedule(root,cfg,fits,account,cpu=True)
    qualified=dict(cfg,behavior={str(s):str(root/f'jobs/fit/POOLED_RL_{s}.private.pt') for s in (601,602)})
    schedule(root,qualified,quals,account);schedule(root,cfg,trains,account);assert dict(account.count)==CAPS
    sealed=[]
    for i in range(4):
        for method in NEW:
            src=root/f'jobs/train{i}';receipt=read(src/(method+'_TRAINING.json'))
            assert receipt['status']=='SEALED' and receipt['step']==300
            assert all((src/(method+'_'+s+'.private.pt')).is_file() for s in ('MID','FINAL'))
            sealed.append(dict(context=f'dev{i}',method=method))
    write(root/'ENDPOINT_LOCK.json',dict(endpoints=136,new=sealed,reused=120,time=time.time(),all_before_new_evaluation=True))
    schedule(root,cfg,evals,account);rows=read(root/'REUSED_RESULTS.json')
    for i in range(4):rows+=read(root/f'jobs/eval{i}/RESULTS.json')
    assert len(rows)==len({(r['context'],r['method']) for r in rows})==136
    result=summarize(rows);write(root/'DECISION.json',result);write(root/'DEVELOPMENT_RESULTS.json',dict(rows=rows,**result))
    write(root/'CONTEXT_COMPARISONS.json',[dict(context=f'dev{i}',**summarize([r for r in rows if r['context']==f'dev{i}'])) for i in range(4)])
    B.csvfile(root/'ENDPOINT_RESULTS.csv',[dict(context=r['context'],method=r['method'],source=r['source'],actions=str(r['actions']),**{k:r[k] for k in ('gain','forget_penalty','utility','absolute_forgetting')},**{f'{phase}_{k}':v for phase in ('new50','new','old') for k,v in r[phase].items()}) for r in rows])
    qd=sum(read(root/'jobs'/s['id']/'FINAL.json')['query_image_evaluations'] for s in evals);assert qd==192
    write(root/'COSTS.json',dict(attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_total=3208,actor_total=4096,training_query_image_evaluations=0,development_query_image_evaluations=qd,new_unique_images=0,reused_development_endpoints=120))
    report=['# V94 pooled advantage scale','',result['status'],'','| Method | New | Old | Utility |','|---|---:|---:|---:|']
    for m,v in result['means'].items():report.append('| '+m+' | '+' | '.join(f'{v[k]:.9f}' for k in ('new','old','utility'))+' |')
    report+=['','Fixed original deployment utility. Same denseCE initialization/objective/fit budget as V92; pooled versus per-state advantage scale. All136 endpoints sealed before new evaluation. CurrentD1 repeated development only. Prior STOPs preserved;V84 NOT_RUN.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    write(root/'FINAL.json',dict(status='COMPLETE',decision=result['status'],physical=dict(account.count),publication='PENDING',time=time.time()));write(root/'STATUS.json',read(root/'FINAL.json'))


if __name__=='__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK')=='1':print('PASS pooled-scale primary gate and reward algebra')
    else:
        root=Path(os.environ['EXEC_RUN']);cfg=read(os.environ['EXEC_CONFIG'])
        spec=importlib.util.spec_from_file_location('previous_worker',cfg['base_entry']);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']),f)
        try:
            if os.environ.get('EXEC_MODE')!='job':coordinator(root,cfg)
            elif cfg['job']=='fit':fit_job(root,cfg)
            elif cfg['job']=='qualification':B.gpu_job(root,cfg)
            else:gpu_job(root,cfg)
        except BaseException as exc:
            write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
