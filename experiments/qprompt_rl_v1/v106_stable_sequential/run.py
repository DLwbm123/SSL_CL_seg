"""Stable-state matched policies evaluated through real native trajectories."""
import copy
import fcntl
import importlib.util
import json
import os
import random
import runpy
import statistics as st
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ARMS = ('WARM', 'CE', 'RL', 'DISTILL')
SEEDS = (601, 602)
STREAMS = (3, 4, 5)
ACTORS = tuple(f'{arm}_{seed}' for arm in ARMS for seed in SEEDS)
METHODS = tuple(f'{mode}_{m}' for mode in ('SAMPLE', 'ARGMAX') for m in ACTORS) + ('UNIFORM', 'GLOBAL', 'RIDGE', 'NN')
CAPS = dict(qualification=8, development=48000, actor_qualification=1, warmup=1024, CE=1024, RL=1024, distillation=1024)
METRICS = ('new', 'old', 'utility', 'gain', 'forget_penalty', 'absolute_forgetting')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fit(root, cfg):
    torch.set_num_threads(1)
    budget = runpy.run_path(cfg['budget_helper'])['Budget'](root, {k:CAPS[k] for k in ('actor_qualification','warmup','CE','RL','distillation')})
    # V105 uses its own qualification category; map only that single synthetic call.
    class Qualification:
        def step(self, category, key, optimizer):
            assert category == 'qualification'
            budget.step('actor_qualification', key, optimizer)
    M.selfcheck(H,V,D,Qualification())
    keys, original, rewards, _, _ = D.dataset(cfg,'V101')
    saved = np.load(Path(cfg['v103'])/'FEATURES.private.npz')
    assert np.array_equal(saved['original'],original)
    values = saved['probes'].mean(1); x=torch.tensor(values,dtype=torch.float32);returns=torch.tensor(rewards,dtype=torch.float32)
    global_action = int(rewards.mean(0).argmax())
    assert all(int(rewards[[i for i,k in enumerate(keys) if k[2]==step]].mean(0).argmax())==global_action for step in (100,200)), 'time control admission changed'
    fitrows=[]; logs=[]; distrows=[]
    for seed in SEEDS:
        rows,part = H.train_pair(x,returns,seed,f'FULL/{seed}',V,B,D,budget,root)
        fitrows += rows; logs += part
        cp=torch.load(root/f'FULL_{seed}_WARM.private.pt',map_location='cpu',weights_only=False)
        actor=H.fresh(V,seed);actor.load_state_dict(cp['actor'])
        mean,std,adv,scale=H.prepare(x,returns);assert all(torch.equal(cp[k],v) for k,v in [('mean',mean),('std',std),('scale',scale)])
        z=(x-mean)/std
        with torch.no_grad():prior=actor(z).log_softmax(-1);star=M.target(z,adv,prior,D)
        opt=torch.optim.Adam(actor.parameters(),lr=.001)
        for step in range(512):
            value=M.update(actor,opt,z,star,budget,'distillation',f'FULL/{seed}/{step}')
            if step in (0,127,511):logs.append(dict(key=f'FULL/{seed}',arm='distillation',step=step+1,loss_before_update=value))
        with (root/f'FULL_{seed}_DISTILL.private.pt').open('xb') as f:torch.save(dict(actor=actor.state_dict(),mean=mean,std=std,scale=scale,seed=seed),f)
        with torch.no_grad():
            log=actor(z).log_softmax(-1).double().log_softmax(-1)
            distrows.append(dict(seed=seed,forward_KL=float((star.exp()*(star-log)).sum(-1).mean()),reverse_KL=float((log.exp()*(log-star)).sum(-1).mean())))
    # Preserve the V102 train-only ridge/nearest-neighbor normalization and lambda.
    mean=values.mean(0);scale=values.std(0);scale[scale<1e-6]=1;z=(values-mean)/scale
    centered=rewards-rewards.mean(1,keepdims=True);intercept=centered.mean(0)
    matrix=z.T@z/len(z)+np.eye(24);rhs=z.T@(centered-intercept)/len(z)
    record=dict(category='ridge',ordinal=1)
    c.e.append(root/'LINEAR_SOLVE_LEDGER.jsonl',dict(event='attempt',**record))
    try:coef=np.linalg.solve(matrix,rhs)
    except BaseException:c.e.append(root/'LINEAR_SOLVE_LEDGER.jsonl',dict(event='failure',**record));raise
    c.e.append(root/'LINEAR_SOLVE_LEDGER.jsonl',dict(event='success',**record))
    assert np.allclose(matrix@coef,rhs,atol=1e-12,rtol=1e-10) and np.isfinite(coef).all()
    np.savez(root/'PREDICTORS.private.npz',mean=mean,scale=scale,z=z,returns=rewards,coef=coef,intercept=intercept)
    D.write(root/'CONTROLS.json',dict(global_action=global_action,time_equals_global=True,ridge_lambda=1,training_states=32))
    assert dict(budget.count)=={k:CAPS[k] for k in ('actor_qualification','warmup','CE','RL','distillation')}
    for name,rows in [('FIT_SUMMARY',fitrows),('FIT_LOG',logs),('DISTILL_SUMMARY',distrows)]:D.write(root/(name+'.json'),rows);D.table(root/(name+'.csv'),rows)
    D.write(root/'QUALIFICATION.json',dict(status='PASS',synthetic_actor_updates=1,grouped_target_and_KL_checks=True,ridge_residual_pass=True,global_time_identity=True))
    D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(budget.count),models=8,linear_solves=1,time=time.time()))


def models(campaign):
    result={}
    for seed in SEEDS:
        for arm in ARMS:
            cp=torch.load(campaign/f'jobs/fit/FULL_{seed}_{arm}.private.pt',map_location='cpu',weights_only=False)
            actor=H.fresh(V,seed);actor.load_state_dict(cp['actor']);actor.eval()
            result[f'{arm}_{seed}']=(actor,cp['mean'],cp['std'])
    return result


def distribution(state, method, fitted, predictor, global_action):
    if method.startswith(('SAMPLE_','ARGMAX_')):
        actor,mean,std=fitted[method.split('_',1)[1]]
        with torch.no_grad():prob=actor((state.float()-mean)/std).softmax(-1)
    elif method=='UNIFORM':prob=torch.ones(12)/12
    else:
        z=(state.double().numpy()-predictor['mean'])/predictor['scale']
        if method=='GLOBAL':action=global_action
        elif method=='RIDGE':action=int((predictor['intercept']+z@predictor['coef']).argmax())
        elif method=='NN':action=int(predictor['returns'][int(np.argmin(((predictor['z']-z)**2).sum(-1)))].argmax())
        else:raise ValueError('unknown policy')
        prob=torch.zeros(12);prob[action]=1.
    assert prob.shape==(12,) and torch.isfinite(prob).all() and (prob>=0).all() and abs(float(prob.sum())-1)<1e-6
    return prob


def select(state,method,fitted,predictor,global_action,generator):
    prob=distribution(state,method,fitted,predictor,global_action)
    action=int(torch.multinomial(prob,1,generator=generator)) if method.startswith('SAMPLE_') or method=='UNIFORM' else int(prob.argmax())
    return action,prob.tolist()


def gpu_job(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    original=E.install_actions(c);E.action_check(c,original)
    roles=c.split_roles(cfg['data']);assert roles==D.read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger=b.JobLedger(root,cfg['caps']);campaign=Path(cfg['campaign']);counts=Counter()
    fitted=models(campaign) if cfg['job']!='evaluate' else {}
    predictor=np.load(campaign/'jobs/fit/PREDICTORS.private.npz') if cfg['job']!='evaluate' else None
    global_action=D.read(campaign/'jobs/fit/CONTROLS.json')['global_action']
    allowed=set(roles['Q_dev_old']+roles['Q_dev_new']) if cfg['job']=='evaluate' else set(roles['A_fit'])
    sample=c.e.primitive.CurrentData.__getitem__
    def guarded(ds,i):
        assert ds.role in ('train_labeled','train_unlabeled')
        assert ds.rows[i]['case_id'] in (allowed if ds.role=='train_labeled' else set(roles['U_adapt']))
        counts['item_attempts_'+ds.role]+=1;value=sample(ds,i);counts['item_success_'+ds.role]+=1
        return value
    c.e.primitive.CurrentData.__getitem__=guarded
    def create(ctx):
        t=b.make(cfg,roles,ledger,ctx,300)
        for name,model in [('student',t.model),('ema',t.ema),('memory',t.memory)]:
            def hook(module,args,name=name):counts[name+'_image_forwards']+=len(args[0])
            model.register_forward_pre_hook(hook)
        return t
    def state(t,key):
        parts=[]
        for j in range(4):
            counts['probe_extractions']+=1;record=dict(key=key,probe=j,ordinal=counts['probe_extractions'])
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='attempt',**record))
            try:value=P.probe(t,j,c)
            except BaseException:c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='failure',**record));raise
            c.e.append(root/'EXTRACTION_LEDGER.jsonl',dict(event='success',**record));parts.append(value.numpy().astype(np.float64))
        return torch.tensor(np.mean(parts,axis=0),dtype=torch.float64)
    if cfg['job']=='qualification':
        keys,_,_,_,_=D.dataset(cfg,'V101');saved=np.load(Path(cfg['v103'])/'FEATURES.private.npz')['probes'].mean(1)
        for i,action in ((0,9),(7,11)):
            ctx=b.contexts()[i];t=create(ctx);t.provider.seed=10168
            entry=torch.load(Path(cfg['v92'])/f'jobs/collect{i}_1/ENTRY100_STREAM.private.pt',map_location='cpu',weights_only=False)
            c.restore(t,entry);before=c.snapshot(t);s=state(t,f'{i}/first')
            assert torch.equal(s,torch.tensor(saved[keys.index((ctx[3],1,100))],dtype=torch.float64))
            chosen,prob=select(s,'SAMPLE_RL_601',fitted,predictor,global_action,torch.Generator().manual_seed(862600+i))
            assert c.e.same(before,c.snapshot(t));t.category='qualification';t.key=f'{i}';t.action=action
            for _ in range(2):t.update()
            expected=c.snapshot(t);c.restore(t,entry);s2=state(t,f'{i}/repeat')
            again,prob2=select(s2,'SAMPLE_RL_601',fitted,predictor,global_action,torch.Generator().manual_seed(862600+i))
            assert torch.equal(s,s2) and chosen==again and prob==prob2 and c.e.same(before,c.snapshot(t))
            t.action=action
            for _ in range(2):t.update()
            assert c.e.same(expected,c.snapshot(t))
            del t;torch.cuda.empty_cache()
        assert dict(ledger.count)=={'qualification':8} and counts['probe_extractions']==16
        D.write(root/'QUALIFICATION.json',dict(status='PASS',native_updates=8,paired_continuations=2,old_stable_states_exact=2,policy_student_RNG_isolation=True,probe_extractions=16))
    else:
        i,stream=cfg['context_index'],cfg['stream']
        m,n,kind,factor=D.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
        ctx=(m,n,(kind,factor),f'dev{i}');t=create(ctx)
        entry=torch.load(Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt',map_location='cpu',weights_only=False)
        c.restore(t,entry);assert t.step==100
        t.provider.seed=168+10000*stream
        random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream)
        paired=c.snapshot(t)
        if cfg['job']=='train':
            c.e.atomic_save(paired,root/'ENTRY.private.pt')
            for method in METHODS:
                c.restore(t,paired);t.category='development';t.key=f'dev{i}/stream{stream}/{method}';trace=[]
                generator=torch.Generator().manual_seed(862600+1000*stream+i)
                for _ in range(200):
                    if t.step in (100,200):
                        before=c.snapshot(t);s=state(t,f'{method}/{t.step}')
                        action,prob=select(s,method,fitted,predictor,global_action,generator)
                        assert c.e.same(before,c.snapshot(t));t.action=action
                        trace.append(dict(step=t.step,state=s.tolist(),probabilities=prob,action=action))
                    t.update()
                    if t.step==150:c.e.atomic_save(c.snapshot(t),root/(method+'_MID.private.pt'))
                    if t.step%50==0:B.write(root/'STATUS.json',dict(status='RUNNING',method=method,step=t.step,physical=dict(ledger.count)))
                c.e.atomic_save(c.snapshot(t),root/(method+'_FINAL.private.pt'))
                D.write(root/(method+'_DECISIONS.private.json'),trace)
                D.write(root/(method+'_TRAINING.json'),dict(status='SEALED',method=method,context=ctx[3],stream=stream,updates=200,actions=[r['action'] for r in trace]))
            assert dict(ledger.count)=={'development':4000} and counts['probe_extractions']==160
        elif cfg['job']=='evaluate':
            assert D.read(campaign/'ENDPOINT_LOCK.json')['trajectories']==240
            references=[r for r in D.read(Path(cfg['v87'])/'DEVELOPMENT_RESULTS.json')['rows'] if r['context']==ctx[3]]
            assert len({r['new_entry'] for r in references})==len({r['old_memory_reference'] for r in references})==1
            entrynew,reference=references[0]['new_entry'],references[0]['old_memory_reference'];rows=[]
            def query(role,condition):
                assert role in ('Q_dev_new','Q_dev_old');counts['query_images']+=len(roles[role]);record=dict(role=role,step=t.step,ordinal=counts['query_images']//4)
                c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
                try:value=scores(t,roles,role,condition)
                except BaseException:c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
                c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record));return value
            src=campaign/f'jobs/train{i}_{stream}'
            for method in METHODS:
                c.restore(t,torch.load(src/(method+'_MID.private.pt'),map_location='cpu',weights_only=False));mid=query('Q_dev_new',ctx[2])
                c.restore(t,torch.load(src/(method+'_FINAL.private.pt'),map_location='cpu',weights_only=False));new=query('Q_dev_new',ctx[2]);old=query('Q_dev_old',('identity',1.))
                value=B.reward(mid['macro'],new['macro'],entrynew,old['macro'],reference)
                rows.append(dict(context=ctx[3],stream=stream,method=method,mid_new=mid['macro'],new=new['macro'],old=old['macro'],utility=value['reward'],gain=value['gain'],forget_penalty=value['forget_penalty'],absolute_forgetting=reference-old['macro'],new_entry=entrynew,old_memory_reference=reference,actions=D.read(src/(method+'_TRAINING.json'))['actions']))
            assert not ledger.count and counts['query_images']==240
            D.write(root/'RESULTS.json',rows)
        else:raise ValueError('unknown GPU job')
    D.write(root/'COUNTS.json',dict(counts))
    D.write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),counts=dict(counts),time=time.time()))


def summary(rows):
    means={m:{k:st.mean(r[k] for r in rows if r['method']==m) for k in METRICS} for m in METHODS}
    for mode in ('SAMPLE','ARGMAX'):
        for arm in ARMS:means[f'{mode}_{arm}']={k:st.mean(means[f'{mode}_{arm}_{s}'][k] for s in SEEDS) for k in METRICS}
    control_names=('SAMPLE_WARM','SAMPLE_CE','SAMPLE_DISTILL','UNIFORM','GLOBAL','RIDGE','NN')
    delta={m:{k:means['SAMPLE_RL'][k]-means[m][k] for k in ('new','old','utility')} for m in control_names}
    paired={f'{arm}_{s}':means[f'SAMPLE_RL_{s}']['utility']-means[f'SAMPLE_{arm}_{s}']['utility'] for arm in ('WARM','CE','DISTILL') for s in SEEDS}
    trade={m:(delta[m]['new']>=.002 and delta[m]['old']>=-.0025) or (delta[m]['old']>=.005 and delta[m]['new']>=-.0025) for m in ('SAMPLE_WARM','SAMPLE_CE','SAMPLE_DISTILL','UNIFORM')}
    passed=all(v>0 for v in paired.values()) and all(v['utility']>=.0005 for v in delta.values()) and all(trade.values())
    return dict(status='V106_POSITIVE_CANDIDATE_REQUIRES_CONFIRMATION' if passed else 'V106_NO_PRACTICAL_SEQUENTIAL_RL_GAIN',means=means,primary_deltas=delta,paired_seed_utility=paired,practical_tradeoff=trade,independent_patient_or_source_confirmation=False)


def selfcheck():
    rows=[]
    for i in range(4):
        for stream in STREAMS:
            for method in METHODS:
                good=method.startswith('SAMPLE_RL_')
                rows.append(dict(context=f'dev{i}',stream=stream,method=method,**{k:(.803 if good else .8) if k in ('new','old') else (.003 if good else 0.) for k in METRICS}))
    assert summary(rows)['status'].startswith('V106_POSITIVE')
    for r in rows:r['utility']=0.
    assert summary(rows)['status'].startswith('V106_NO_')
    assert len(METHODS)==20 and len(rows)==240 and 240*200==CAPS['development']
    state=torch.zeros(24);pred=dict(mean=np.zeros(24),scale=np.ones(24),z=np.zeros((2,24)),returns=np.array([[0.,1.]+[0.]*10,[1.,0.]+[0.]*10]),coef=np.zeros((24,12)),intercept=np.zeros(12))
    assert select(state,'RIDGE',{},pred,0,torch.Generator())[0]==0
    assert select(state,'NN',{},pred,0,torch.Generator())[0]==1
    assert select(state,'GLOBAL',{},pred,3,torch.Generator())[0]==3


def schedule(root,cfg,jobs,account,cpu=False):
    pending=list(jobs);active={};done=[];failed=[]
    while pending or active:
        for gpu,(proc,spec) in list(active.items()):
            if proc.poll() is None:continue
            dest=root/'jobs'/spec['id'];D.write(dest/'PROCESS_EXIT.json',dict(exit_code=proc.returncode,time=time.time()))
            (failed if proc.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[gpu]
        if not failed:
            for gpu in ((-1,) if cpu else (4,5,6,7)):
                if not pending:break
                if gpu in active:continue
                if not cpu and int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12000:continue
                spec=pending.pop(0);dest=root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                D.write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=gpu))
                env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    proc=subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                D.write(dest/'LAUNCH.json',dict(pid=proc.pid,gpu=gpu,time=time.time(),commit=cfg['commit']));active[gpu]=(proc,spec)
        account.refresh()
        B.write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',phase=jobs[0]['job'],active=[s['id'] for _,s in active.values()],pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selfcheck();D.write(root/'SELF_CHECK.json',dict(status='PASS',optimizer_updates=0,checks=['positive_and_negative_decision','full_matrix_counts','simple_action_ties']))
    fitjobs=[dict(id='fit',job='fit',caps={})];qual=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    train=[dict(id=f'train{i}_{s}',job='train',context_index=i,stream=s,caps=dict(development=4000)) for i in range(4) for s in STREAMS]
    evaluate=[dict(id=f'evaluate{i}_{s}',job='evaluate',context_index=i,stream=s,caps={}) for i in range(4) for s in STREAMS]
    jobs=fitjobs+qual+train+evaluate
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account=Accounting(root,caps=CAPS,jobs=jobs)
    schedule(root,cfg,fitjobs,account,cpu=True)
    D.write(root/'FIT_LOCK.json',dict(status='SEALED_BEFORE_DEPLOYMENT',actors=8,controls=4,time=time.time()))
    schedule(root,cfg,qual,account);schedule(root,cfg,train,account)
    assert dict(account.count)==CAPS and not account.failure
    assert sum(len(list((root/'jobs'/j['id']).glob('*_TRAINING.json'))) for j in train)==240
    D.write(root/'ENDPOINT_LOCK.json',dict(status='SEALED_BEFORE_QUERY_READOUT',trajectories=240,snapshots=480,time=time.time(),physical=dict(account.count)))
    schedule(root,cfg,evaluate,account)
    rows=sum((D.read(root/'jobs'/j['id']/'RESULTS.json') for j in evaluate),[])
    assert len(rows)==len({(r['context'],r['stream'],r['method']) for r in rows})==240
    counts=Counter()
    for j in qual+train+evaluate:counts.update(D.read(root/'jobs'/j['id']/'COUNTS.json'))
    assert counts['query_images']==2880 and counts['probe_extractions']==1936
    decision=summary(rows)
    D.write(root/'DEVELOPMENT_RESULTS.json',dict(rows=rows,**decision));D.table(root/'DEVELOPMENT_RESULTS.csv',rows)
    D.write(root/'DECISION.json',decision)
    B.write(root/'COSTS.json',dict(native_updates=48008,actor_optimizer_updates=4097,linear_data_solves=1,physical=dict(account.count),success=dict(account.success),failures=dict(account.failure),counts=dict(counts),new_annotation_cases=0))
    D.write(root/'FINAL.json',dict(status='COMPLETE',decision=decision['status'],time=time.time(),physical=dict(account.count),publication='PENDING'))
    B.write(root/'STATUS.json',D.read(root/'FINAL.json'))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    B=load('base_loss',cfg['base_entry']);D=load('diagnostic',cfg['diagnostic_entry']);V=load('actor12',cfg['v99_entry'])
    H=load('fold_local',cfg['v104_entry']);M=load('matched_target',cfg['v105_entry']);P=load('probe',cfg['probe_entry']);E=load('actions12',cfg['expanded_entry'])
    D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
    try:
        if os.environ.get('EXEC_MODE')=='job':
            if cfg['job']=='fit':fit(root,cfg)
            else:gpu_job(root,cfg)
        else:coordinator(root,cfg)
    except BaseException as exc:
        D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
