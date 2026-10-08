"""Matched twelve-action CE/RL using current expanded-policy continuation returns."""
import fcntl
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

CAPS = dict(qualification=8, audit=56000, actor_fit=4096)
ARMS = ('REFRESH_CE', 'REFRESH_RL')
ACTORS = tuple(f'{a}_{s}' for a in ('REFRESH_INIT',)+ARMS for s in (601,602))
NEW = ACTORS+('REFRESH_UNIFORM','REFRESH_GLOBAL','REFRESH_TIME')


def current_advantages(returns, scale):
    assert math.isfinite(float(scale)) and float(scale)>0
    return ((returns-returns[:,:9].mean(-1,keepdim=True))/scale).clamp(-3,3)


def current_gain(basegain, final, baseline):
    return basegain+.75*(final-baseline)


def selfcheck():
    import torch
    H.selfcheck();V.selfcheck()
    x=torch.arange(24,dtype=torch.float32).reshape(2,12)/100
    a=current_advantages(x,.01);assert torch.equal(a,((x-x[:,:9].mean(-1,keepdim=True))/.01).clamp(-3,3))
    assert current_gain(.01,.7,.7)==.01 and abs(current_gain(.01,.72,.7)-.025)<1e-12
    assert 16*(12*200+11*100)==CAPS['audit'] and 16*(12*12+11*8)==3712
    rows=[dict(context=f'dev{i}',method=m,mode=mode,**{k:(.803 if k=='new' else .8) if k in ('new','old') else (.003 if k=='utility' else 0.) for k in H.KEYS}) for i in range(4) for m in NEW for mode in ('ORIGINAL_T1_SAMPLE','EXACT_T1_EXPECTATION','FIXED_ARGMAX')]
    for r in rows:
        if not r['method'].startswith('REFRESH_RL'):r.update(new=.8,utility=0.)
    history={m:{k:.8 if k in ('new','old') else 0. for k in H.KEYS} for m in ('CONTINUE_CE','POOLED_RL','EXPANDED_CE','EXPANDED_RL','EXPANDED_INIT')}
    assert summarize(rows,history)['status'].startswith('PASS')
    assert summarize([dict(r,utility=0.) for r in rows],history)['status'].startswith('NO_')


def cached_qualification(root,cfg):
    import torch
    torch.set_num_threads(2);source=Path(cfg['v100']);prior=Path(cfg['v99']);vectors=0;pairs=0
    distributions=G.read(source/'DISTRIBUTION_LOCK.json')['distributions']
    for i in range(4):
        states=G.read(source/f'jobs/extract{i}/STATES.private.json')
        for seed in (601,602):
            method=f'EXPANDED_CE_{seed}';model=V.load12(prior/f'jobs/fit/{method}.private.pt')
            d=next(r for r in distributions if r['context']==f'dev{i}' and r['method']==method)
            for key in [-1]+list(range(12)):
                state=torch.tensor(states[str(key)],dtype=torch.float32);before=torch.get_rng_state();p=B.probabilities(state,[model]).tolist()
                assert p==(d['first'] if key==-1 else d['second'][key]) and torch.equal(before,torch.get_rng_state());vectors+=1
            generator=torch.Generator().manual_seed(860301+i);a=int(torch.multinomial(torch.tensor(d['first']),1,generator=generator));b=int(torch.multinomial(torch.tensor(d['second'][a]),1,generator=generator));assert [a,b]==d['sampled_replay'];pairs+=1
    assert vectors==104 and pairs==8
    G.write(root/'CACHED_STATE_QUALIFICATION.json',dict(status='PASS',exact_probability_vectors=vectors,exact_sampled_pairs=pairs,optimizer_updates=0,query_accesses=0))
    G.write(root/'FINAL.json',dict(status='COMPLETE',physical={},query_image_evaluations=0,time=time.time()))


def gpu_job(root,cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    if cfg['job']=='qualification':E.G=G;E.B=B;E.gpu_job(root,cfg);return
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168);E.install_actions(c)
    roles=c.split_roles(cfg['data']);assert roles==G.read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger=b.JobLedger(root,cfg['caps']);i=cfg['context_index'];stream=cfg['stream'];ctx=b.contexts()[i];t=b.make(cfg,roles,ledger,ctx,300)
    t.provider.seed=168+10000*stream;src=Path(cfg['v92'])/f'jobs/collect{i}_{stream}'
    entry=torch.load(src/'ENTRY100_STREAM.private.pt',map_location='cpu',weights_only=False);assert entry['step']==100;c.restore(t,entry)
    models=[V.load12(Path(cfg['v99'])/f'jobs/fit/{a}_{s}.private.pt') for a in ('EXPANDED_CE','EXPANDED_RL') for s in (601,602)]
    generator=torch.Generator().manual_seed(860901+100*stream+i);first,trace=B.select(t,models,generator);afterfirst=generator.get_state()
    oldrows=G.read(src/'TRAINING_TABLE.private.json');assert all(r['state']==trace['state'] for r in oldrows if r['step']==100)
    refs=[json.loads(x) for x in (Path(cfg['original'])/'jobs/audit_1/ACTION_ROWS.jsonl').read_text().splitlines()]
    refs=[r for r in refs if r['context']==ctx[3] and r['entry_step']==100];assert len(refs)==9 and len({r['new_entry'] for r in refs})==1
    newentry=refs[0]['new_entry'];oldentry=oldrows[0]['old_entry'];rows=[];queries=0;t.category='audit'
    def query(role,condition):
        nonlocal queries
        assert role in ('Q_train_new','Q_train_old');queries+=len(roles[role]);record=dict(role=role,step=t.step,image_calls_upper_bound=queries)
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value=scores(t,roles,role,condition)
        except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value['macro']
    def row(step,action,state,gain,oldfinal,continuation,reused=False):
        return dict(context=ctx[3],stream=stream,step=step,action=action,state=state,gain=gain,old_entry=oldentry,old_final=oldfinal,dense_reward=gain+oldfinal-oldentry,continuation_action=continuation,reused=reused)
    for action in range(12):
        c.restore(t,entry);t.action=action;t.key=f'{ctx[3]}/{stream}/first{action}';paired=torch.Generator();paired.set_state(afterfirst);values={}
        for _ in range(200):
            if t.step==200:
                prefix=c.snapshot(t);second,secondtrace=B.select(t,models,paired);t.action=second
                if action==first:
                    retained=prefix;selectedsecond=second;retainedtrace=secondtrace;c.e.atomic_save(prefix,root/'ON_POLICY_ENTRY200.private.pt')
            t.update()
            if t.step in (150,300):values[t.step]=query('Q_train_new',ctx[2])
            if t.step==300:oldfinal=query('Q_train_old',('identity',1.))
        gain=.25*(values[150]-newentry)+.75*(values[300]-newentry);rows.append(row(100,action,trace['state'],gain,oldfinal,second))
        c.e.atomic_save(c.snapshot(t),root/f'FIRST_{action}_FINAL.private.pt')
        if action==first:baseline=values[300];basegain=gain;baseold=oldfinal
        G.write(root/'STATUS.json',dict(status='RUNNING',rows=len(rows),physical=dict(ledger.count),query_image_evaluations=queries))
    c.restore(t,retained);state=t.extract().tolist();assert state==retainedtrace['state']
    for action in range(12):
        if action==selectedsecond:gain=basegain;oldfinal=baseold
        else:
            c.restore(t,retained);t.action=action;t.key=f'{ctx[3]}/{stream}/second{action}'
            for _ in range(100):t.update()
            gain=current_gain(basegain,query('Q_train_new',ctx[2]),baseline);oldfinal=query('Q_train_old',('identity',1.))
            c.e.atomic_save(c.snapshot(t),root/f'SECOND_{action}_FINAL.private.pt')
        rows.append(row(200,action,state,gain,oldfinal,-1,action==selectedsecond))
        G.write(root/'STATUS.json',dict(status='RUNNING',rows=len(rows),physical=dict(ledger.count),query_image_evaluations=queries))
    assert len(rows)==24 and dict(ledger.count)=={'audit':3500} and queries==232
    G.write(root/'TRAINING_TABLE.private.json',rows);G.write(root/'BEHAVIOR.private.json',dict(first=trace,second=retainedtrace))
    G.write(root/'COLLECTION_RECEIPT.json',dict(status='PASS',rows=24,original_state100_exact=True,new_state200_matches_current_behavior=True,first_action=first,second_action=selectedsecond,reused_current_second_returns=1,reused_old_rewards=0,native_updates=3500,query_image_evaluations=232))
    G.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),query_image_evaluations=queries,time=time.time()))


def fit_job(root,cfg):
    import torch
    torch.set_num_threads(2);rows=sum((G.read(Path(cfg['campaign'])/f'jobs/collect{i}_{s}/TRAINING_TABLE.private.json') for i in range(8) for s in (1,2)),[])
    keys=sorted({(r['context'],r['stream'],r['step']) for r in rows});groups=[]
    for key in keys:
        g=sorted((r for r in rows if (r['context'],r['stream'],r['step'])==key),key=lambda r:r['action']);assert [r['action'] for r in g]==list(range(12)) and all(r['state']==g[0]['state'] for r in g);groups.append(g)
    assert len(groups)==32 and len(rows)==384
    x=torch.tensor([g[0]['state'] for g in groups]);returns=torch.tensor([[r['dense_reward'] for r in g] for g in groups]);assert torch.isfinite(returns).all()
    reward_scale=G.read(Path(cfg['v94'])/'jobs/fit/SCALE_QUALIFICATION.json')['pooled_scale'];assert reward_scale==.0026099849492311478
    adv=current_advantages(returns,reward_scale);global_action=int(returns.mean(0).argmax())
    controls={'REFRESH_GLOBAL':{str(step):global_action for step in (100,200)},'REFRESH_TIME':{str(step):int(returns[[k[2]==step for k in keys]].mean(0).argmax()) for step in (100,200)}};G.write(root/'TRAINING_CONTROLS.json',controls)
    budget=runpy.run_path(cfg['budget_helper'])['Budget'](root,dict(actor_fit=4096));logs=[];summary=[]
    for seed in (601,602):
        src=Path(cfg['v99'])/f'jobs/fit/EXPANDED_CE_{seed}.private.pt';initial,center,scale=V.load12(src);z=(x-center)/scale;prior=initial(z).detach().log_softmax(-1)
        with (root/f'REFRESH_INIT_{seed}.private.pt').open('xb') as f:torch.save(dict(actor=initial.state_dict(),mean=center,scale=scale,seed=seed,updates=0),f)
        for arm in ARMS:
            actor,_,_=V.load12(src);assert all(torch.equal(v,initial.state_dict()[k]) for k,v in actor.state_dict().items());opt=torch.optim.Adam(actor.parameters(),lr=.001)
            for update in range(1024):
                loss=B.actor_loss(actor,z,adv,prior,arm=='REFRESH_RL');assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.));budget.step('actor_fit',f'{arm}/{seed}/{update}',opt)
                if update+1 in (1,64,1024):logs.append(dict(arm=arm,seed=seed,update=update+1,objective_before_update=float(loss.detach())))
            with (root/f'{arm}_{seed}.private.pt').open('xb') as f:torch.save(dict(actor=actor.state_dict(),mean=center,scale=scale,seed=seed,arm=arm,updates=1024),f)
            with torch.no_grad():
                log=actor(z).log_softmax(-1);prob=log.exp();summary.append(dict(arm=arm,seed=seed,normalized_return=float((prob*adv).sum(-1).mean()),raw_dense_return=float((prob*returns).sum(-1).mean()),KL_to_reference=float((prob*(log-prior)).sum(-1).mean()),entropy=float(-(prob*log).sum(-1).mean()),new_action_mass=float(prob[:,9:].sum(-1).mean())))
    assert dict(budget.count)=={'actor_fit':4096}
    B.csvfile(root/'FIT_LOG.csv',logs);B.csvfile(root/'FIT_SUMMARY.csv',summary);B.csvfile(root/'REWARD_SUMMARY.csv',[{k:v for k,v in r.items() if k!='state'} for r in rows])
    G.write(root/'FIT_QUALIFICATION.json',dict(status='PASS',states=32,actions=384,parameters=1196,fixed_reward_scale=reward_scale,matched_initialization=True,clipped_values=int((adv.abs()==3).sum())))
    G.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(budget.count),time=time.time()))


def select_job(root,cfg):
    import torch
    torch.set_num_threads(2);campaign=Path(cfg['campaign']);assert G.read(campaign/'FIT_LOCK.json')['actors']==6
    models={m:V.load12(campaign/f'jobs/fit/{m}.private.pt') for m in ACTORS};controls=G.read(campaign/'jobs/fit/TRAINING_CONTROLS.json');distributions=[]
    for i in range(4):
        states=G.read(Path(cfg['v100'])/f'jobs/extract{i}/STATES.private.json')
        for method in NEW:
            if method in models:
                before=torch.get_rng_state();probs={key:B.probabilities(torch.tensor(state,dtype=torch.float32),[models[method]]).tolist() for key,state in states.items()};assert torch.equal(before,torch.get_rng_state())
                first=probs['-1'];second=[probs[str(a)] for a in range(12)]
            elif method=='REFRESH_UNIFORM':first=[1/12]*12;second=[first[:] for _ in range(12)]
            else:first=[float(a==controls[method]['100']) for a in range(12)];second=[[float(a==controls[method]['200']) for a in range(12)] for _ in range(12)]
            generator=torch.Generator().manual_seed(860301+i)
            if method in models or method=='REFRESH_UNIFORM':a=int(torch.multinomial(torch.tensor(first),1,generator=generator));b=int(torch.multinomial(torch.tensor(second[a]),1,generator=generator));sampled=[a,b]
            else:sampled=[controls[method]['100'],controls[method]['200']]
            distributions.append(dict(context=f'dev{i}',method=method,first=first,second=second,sampled=sampled,argmax=H.greedy(first,second)))
    G.write(root/'DISTRIBUTIONS.json',distributions);G.write(root/'FINAL.json',dict(status='COMPLETE',physical={},query_image_evaluations=0,time=time.time()))


def summarize(rows,historical):
    means={}
    for mode in sorted({r['mode'] for r in rows}):
        group=[r for r in rows if r['mode']==mode];means[mode]={m:{k:st.mean(r[k] for r in group if r['method']==m) for k in H.KEYS} for m in NEW}
        for arm in ('REFRESH_INIT',)+ARMS:means[mode][arm]={k:st.mean(means[mode][f'{arm}_{s}'][k] for s in (601,602)) for k in H.KEYS}
    primary=means['ORIGINAL_T1_SAMPLE'];rl=primary['REFRESH_RL'];controls={m:primary[m] for m in ('REFRESH_CE','REFRESH_INIT','REFRESH_UNIFORM','REFRESH_GLOBAL','REFRESH_TIME')};controls.update({m:historical[m] for m in ('CONTINUE_CE','POOLED_RL','EXPANDED_CE','EXPANDED_RL','EXPANDED_INIT')})
    delta={m:{k:rl[k]-v[k] for k in ('new','old','utility')} for m,v in controls.items()}
    paired={f'{arm}_{s}':primary[f'REFRESH_RL_{s}']['utility']-primary[f'{arm}_{s}']['utility'] for arm in ('REFRESH_CE','REFRESH_INIT') for s in (601,602)}
    trade={m:(delta[m]['new']>=.002 and delta[m]['old']>=-.0025) or (delta[m]['old']>=.005 and delta[m]['new']>=-.0025) for m in ('REFRESH_UNIFORM','REFRESH_CE','REFRESH_INIT','CONTINUE_CE')}
    passed=all(v['utility']>=.0005 for v in delta.values()) and all(v>0 for v in paired.values()) and all(trade.values())
    return dict(status='PASS_V101_CURRENT_EXPANDED_CACHED_D1_ONLY' if passed else 'NO_V101_CURRENT_EXPANDED_RL_INCREMENTAL_GAIN',means=means,primary_deltas=delta,paired_seed_utility=paired,practical=trade,independent_confirmation=False,automatic_research_stop=False)


def schedule(root,cfg,jobs,account,cpu=False):
    pending=list(jobs);active={};failed=[];done=[]
    while pending or active:
        for slot,(process,spec) in list(active.items()):
            if process.poll() is None:continue
            dest=root/'jobs'/spec['id'];G.write(dest/'PROCESS_EXIT.json',dict(exit_code=process.returncode,time=time.time()))
            (failed if process.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in ((-1,) if cpu else (5,6,7)):
                if not pending:break
                if slot in active:continue
                if not cpu and int(subprocess.check_output(['nvidia-smi','--id='+str(slot),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12000:continue
                spec=pending.pop(0);dest=root/'jobs'/spec['id'];dest.mkdir(exist_ok=False);G.write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=slot))
                env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='' if cpu else str(slot),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:process=subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                G.write(dest/'LAUNCH.json',dict(pid=process.pid,gpu=slot,time=time.time(),commit=cfg['commit']));active[slot]=process,spec
        account.refresh();G.write(root/'COSTS.json',dict(scope='V101_NEW_ONLY',attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_cap=56008,actor_cap=4096))
        G.write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',active=[s['id'] for _,s in active.values()],pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert G.read(Path(cfg['v99'])/'FINAL.json')['status']==G.read(Path(cfg['v100'])/'FINAL.json')['status']=='COMPLETE'
    cached=[dict(id='cached_qualification',job='cached_qualification',caps={})];qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    collect=[dict(id=f'collect{i}_{s}',job='collect',context_index=i,stream=s,caps=dict(audit=3500)) for i in range(8) for s in (1,2)]
    fit=[dict(id='fit',job='fit',caps={})];select=[dict(id='select',job='select',caps={})];jobs=cached+qual+collect+fit+select;assert len(jobs)==20
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False);account=Accounting(root,caps=CAPS,jobs=jobs)
    schedule(root,cfg,cached,account,cpu=True);schedule(root,cfg,qual,account);schedule(root,cfg,collect,account);assert dict(account.count)==dict(qualification=8,audit=56000)
    schedule(root,cfg,fit,account,cpu=True);assert dict(account.count)==CAPS
    G.write(root/'FIT_LOCK.json',dict(actors=6,fitted=4,initial=2,training_controls_sealed=True,time=time.time()))
    schedule(root,cfg,select,account,cpu=True);distributions=G.read(root/'jobs/select/DISTRIBUTIONS.json');assert len(distributions)==36
    G.write(root/'DISTRIBUTION_LOCK.json',dict(status='SEALED',before_metric_lookup=True,distributions=distributions,time=time.time()))
    grid=G.read(Path(cfg['v98'])/'GRID_RESULTS.json')['rows'];lookup={(r['context'],r['first'],r['second']):r for r in grid};assert len(lookup)==576
    history=G.read(Path(cfg['v99'])/'DEVELOPMENT_RESULTS.json');allrows=list(history['rows']);assert len(allrows)==204;rows=[]
    def flat(r):return {k:r[k]['macro'] if k in ('new','old') else r[k] for k in H.KEYS}
    for d in distributions:
        context,method=d['context'],d['method'];outcomes=[flat(lookup[(context,a,b)]) for a in range(12) for b in range(12)]
        sampled=lookup[(context,*d['sampled'])];argmax=lookup[(context,*d['argmax'])];expected=H.mean_outcome(H.weights(d['first'],d['second']),outcomes)
        allrows.append(dict(sampled,method=method,actions=d['sampled'],source='REUSED_GRID_OUTCOME'))
        for mode,v in [('ORIGINAL_T1_SAMPLE',flat(sampled)),('EXACT_T1_EXPECTATION',expected),('FIXED_ARGMAX',flat(argmax))]:rows.append(dict(context=context,method=method,mode=mode,**v))
    assert len(allrows)==len({(r['context'],r['method']) for r in allrows})==240 and len(rows)==108
    decision=summarize(rows,history['means']);G.write(root/'DECISION.json',decision);G.write(root/'DEVELOPMENT_RESULTS.json',dict(rows=allrows,**decision));G.write(root/'READOUT_RESULTS.json',dict(rows=rows,**decision));B.csvfile(root/'READOUT_RESULTS.csv',rows)
    qt=sum(G.read(root/'jobs'/s['id']/'FINAL.json')['query_image_evaluations'] for s in collect);assert qt==3712
    G.write(root/'COSTS.json',dict(attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_total=56008,actor_total=4096,training_image_evaluations=qt,development_image_evaluations=0,new_unique_images=0,new_training_rewards=384,reused_current_second_returns=16,reused_old_training_rewards=0,cached_sampled_policy_rows=36,historical_sampled_rows=204,all_mode_rows=108,unique_cached_grid_rows=576))
    G.write(root/'FINAL.json',dict(status='COMPLETE',decision=decision['status'],physical=dict(account.count),publication='PENDING',time=time.time()));G.write(root/'STATUS.json',G.read(root/'FINAL.json'))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    G=load('grid',cfg['grid_entry']);B=load('reward',cfg['base_entry']);E=load('actions',cfg['expanded_entry']);V=load('expanded_policy',cfg['v99_entry']);H=load('readout',cfg['v100_entry']);V.B=B
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS12actor andloss,conditionalreadout,fixedscale,currentprefixgain,budget;zerooptimizer/query')
    else:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']),f)
        try:
            if os.environ.get('EXEC_MODE')!='job':coordinator(root,cfg)
            elif cfg['job']=='cached_qualification':cached_qualification(root,cfg)
            elif cfg['job']=='fit':fit_job(root,cfg)
            elif cfg['job']=='select':select_job(root,cfg)
            else:gpu_job(root,cfg)
        except BaseException as exc:G.write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
