"""One paired full-episode policy-improvement round, using the native engine."""
import csv
import fcntl
import json
import os
import random
import runpy
import statistics as st
import subprocess
import time
import traceback
from pathlib import Path

ARMS = ('LOCAL_CE', 'EPISODIC_CE', 'EPISODIC_RL')
NEW = tuple(f'{arm}_{seed}' for arm in ARMS for seed in (601, 602))
REUSED = ('NATIVE', 'FIXED_BEST', 'UNIFORM_ACTION', 'TIME_FIXED', 'Z_1024_601', 'Z_1024_602')
FLOOR = .0007379373167760999


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');tmp.replace(path)


def csvfile(path, rows):
    with Path(path).open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def reward(short, end, entry, old, reference):
    gain = .25*(short-entry)+.75*(end-entry);penalty = max(0., reference-old-.005)
    return dict(reward=gain-penalty, gain=gain, forget_penalty=penalty)


def actor_loss(actor, z, adv, prior, rl):
    log = actor(z).log_softmax(-1);prob = log.exp();entropy_term = .01*(prob*log).sum(-1).mean()
    if rl:return -(prob*adv).sum(-1).mean()+.25*(prob*(log-prior)).sum(-1).mean()+entropy_term
    return -(adv.div(.25).softmax(-1)*log).sum(-1).mean()+entropy_term


def summarize(rows):
    keys = ('new', 'old', 'utility', 'gain', 'forget_penalty', 'absolute_forgetting')
    means = {m:{k:st.mean(r[k]['macro'] if k in ('new', 'old') else r[k]
                        for r in rows if r['method'] == m) for k in keys} for m in REUSED+NEW}
    for arm in ('Z_1024',)+ARMS:
        means[arm] = {k:st.mean(means[f'{arm}_{s}'][k] for s in (601, 602)) for k in keys}
    primary = means['EPISODIC_RL']
    comparators = REUSED[:4]+('Z_1024', 'LOCAL_CE', 'EPISODIC_CE')
    delta = {m:{k:primary[k]-means[m][k] for k in ('new', 'old', 'utility')} for m in comparators}
    paired = {str(s):means[f'EPISODIC_RL_{s}']['utility']-means[f'EPISODIC_CE_{s}']['utility'] for s in (601, 602)}
    tradeoff = {m:(delta[m]['new'] >= .002 and delta[m]['old'] >= -.0025)
                 or (delta[m]['old'] >= .005 and delta[m]['new'] >= -.0025)
                for m in ('UNIFORM_ACTION', 'EPISODIC_CE')}
    passed = all(v['utility'] >= .0005 for v in delta.values()) and all(v > 0 for v in paired.values()) and all(tradeoff.values())
    return dict(status='PASS_V89_EPISODIC_RL_CURRENT_D1_ONLY' if passed else 'STOP_V89_NO_RL_INCREMENTAL_GAIN',
                means=means, primary_deltas=delta, paired_seed_utility=paired, tradeoff=tradeoff,
                secondary_credit_utility=means['EPISODIC_CE']['utility']-means['LOCAL_CE']['utility'],
                V84='NOT_RUN', independent_confirmation=False, GRPO=False)


def selfcheck():
    assert abs(reward(.72, .74, .70, .79, .80)['reward']-.03) < 1e-12
    assert abs(reward(.72, .74, .70, .79, .80)['forget_penalty']-.005) < 1e-12
    rows = [dict(method=m, context=f'dev{i}', new={'macro':.8+(.003 if m.startswith('EPISODIC_RL') else 0.)},
                 old={'macro':.8}, utility=.003 if m.startswith('EPISODIC_RL') else 0.,
                 gain=0., forget_penalty=0., absolute_forgetting=.01) for i in range(4) for m in REUSED+NEW]
    assert summarize(rows)['status'].startswith('PASS')
    assert summarize([dict(r, utility=0.) for r in rows])['status'].startswith('STOP')
    assert summarize([dict(r, new={'macro':.8}) for r in rows])['status'].startswith('STOP')


def load_model(path):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor
    value = torch.load(path, map_location='cpu', weights_only=False)
    actor = Actor(value['seed']);actor.load_state_dict(value['actor']);actor.eval()
    assert value['mean'].shape == value['scale'].shape == (24,) and (value['scale'] > 0).all()
    return actor, value['mean'], value['scale']


def probabilities(state, models):
    import torch
    with torch.no_grad():
        prob = sum(actor((state-center)/scale).softmax(-1) for actor,center,scale in models)/len(models)
    assert torch.isfinite(prob).all() and abs(float(prob.sum())-1.) < 1e-6
    return prob


def select(t, models, generator):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c
    before = c.snapshot(t);state = t.extract();prob = probabilities(state, models)
    action = int(torch.multinomial(prob, 1, generator=generator))
    assert c.e.same(before, c.snapshot(t)), 'policy inference changed student/global RNG'
    return action, dict(step=t.step, state=state.tolist(), probabilities=prob.tolist(), action=action)


def gpu_job(root, cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    e = c.e;torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    ledger = b.JobLedger(root, cfg['caps']);queries = 0
    roles = c.split_roles(cfg['data']);assert roles == read(Path(cfg['action_root'])/'ROLES.private.json')
    behavior = [load_model(cfg['behavior'][str(s)]) for s in (601, 602)]
    original = Path(cfg['original']);campaign = Path(cfg['campaign'])

    def query(t, role, condition, full=False):
        nonlocal queries
        queries += len(roles[role]);record = dict(role=role,step=t.step,image_calls_upper_bound=queries)
        e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**record))
        try:value = scores(t, roles, role, condition)
        except BaseException:
            e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**record));raise
        e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**record))
        return value if full else value['macro']

    def training_entry(index):
        ctx = b.contexts()[index];t = b.make(cfg, roles, ledger, ctx, 300)
        value = torch.load(original/f'jobs/decision_entries/{ctx[3]}/ENTRY100.private.pt', map_location='cpu', weights_only=False)
        c.restore(t, value);assert t.step == 100 and t.options['total_steps'] == 300
        return ctx, t, value

    if cfg['job'] == 'qualification':
        for index in (0, 7):
            ctx,t,entry = training_entry(index);t.category = 'qualification';t.key = ctx[3]
            action,_ = select(t, behavior, torch.Generator().manual_seed(861000+index));t.action = action
            for _ in range(2):t.update()
            expected = c.snapshot(t);c.restore(t, entry)
            again,_ = select(t, behavior, torch.Generator().manual_seed(861000+index));assert again == action
            t.action = again
            for _ in range(2):t.update()
            assert e.same(expected, c.snapshot(t)), 'native paired continuation mismatch'
            del t;torch.cuda.empty_cache()
        assert dict(ledger.count) == {'qualification':8} and queries == 0
        write(root/'QUALIFICATION.json', dict(status='PASS', native_updates=8, exact_pairs=2,
              policy_student_rng_isolation=True, query_accesses=0, native_engine_checks='REUSED_V83'))
    elif cfg['job'] == 'collect':
        assert read(campaign/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
        index,stream = cfg['context_index'],cfg['stream'];ctx,t,base = training_entry(index)
        source_rows = [json.loads(s) for s in (original/'jobs/audit_1/ACTION_ROWS.jsonl').read_text().splitlines()]
        references = [r for r in source_rows if r['context'] == ctx[3] and r['entry_step'] == 100]
        assert len(references) == 9
        entrynew = references[0]['new_entry'];reference = references[0]['old_frozen_memory_reference']
        assert all(r['new_entry'] == entrynew and r['old_frozen_memory_reference'] == reference for r in references)
        t.provider.seed = 168+stream*10000
        random.seed(860100+stream);np.random.seed(860100+stream);torch.manual_seed(860100+stream);torch.cuda.manual_seed_all(860100+stream)
        entry = c.snapshot(t);e.atomic_save(entry, root/'ENTRY100_STREAM.private.pt')
        generator = torch.Generator().manual_seed(860901+100*stream+index)
        first_action,first_decision = select(t, behavior, generator);after_first = generator.get_state()
        rows = [];retained = None;selected = None;t.category = 'audit'
        for action in range(9):
            c.restore(t, entry);t.action = action;t.key = f'{ctx[3]}/stream{stream}/first{action}'
            values = {};second_action = None
            paired = torch.Generator();paired.set_state(after_first)
            for _ in range(200):
                if t.step == 200:
                    second_action, _ = select(t, behavior, paired);t.action = second_action
                t.update()
                if t.step in (125, 150, 200, 300) or (t.step == 225 and action == first_action):
                    values[f'new{t.step}'] = query(t, 'Q_train_new', ctx[2])
                if t.step in (200, 300):values[f'old{t.step}'] = query(t, 'Q_train_old', ('identity', 1.))
                if t.step == 200 and action == first_action:
                    retained = c.snapshot(t);e.atomic_save(retained, root/'ON_POLICY_ENTRY200.private.pt')
            e.atomic_save(c.snapshot(t), root/f'FIRST_{action}_FINAL.private.pt')
            rows.append(dict(context=ctx[3], stream=stream, step=100, action=action,
                state=first_decision['state'], local=reward(values['new125'], values['new200'], entrynew, values['old200'], reference),
                episodic=reward(values['new150'], values['new300'], entrynew, values['old300'], reference),
                continuation_action=second_action, reused=False))
            if action == first_action:selected = dict(values=values, second_action=second_action)
            write(root/'STATUS.json', dict(status='RUNNING', phase='FIRST_ACTION_ROLLOUTS', completed=action+1,
                                          physical=dict(ledger.count), query_image_evaluations=queries))
        assert retained is not None and selected is not None
        c.restore(t, retained);second_state = t.extract().tolist();prefix = selected['values']
        write(root/'BEHAVIOR.private.json', dict(first=first_decision, second_action=selected['second_action']))
        for action in range(9):
            reused = action == selected['second_action']
            if reused:values = prefix
            else:
                c.restore(t, retained);t.action = action;t.key = f'{ctx[3]}/stream{stream}/second{action}';values = {}
                for _ in range(100):
                    t.update()
                    if t.step in (225, 300):values[f'new{t.step}'] = query(t, 'Q_train_new', ctx[2])
                    if t.step == 300:values['old300'] = query(t, 'Q_train_old', ('identity', 1.))
                e.atomic_save(c.snapshot(t), root/f'SECOND_{action}_FINAL.private.pt')
            rows.append(dict(context=ctx[3], stream=stream, step=200, action=action, state=second_state,
                local=reward(values['new225'], values['new300'], prefix['new200'], values['old300'], reference),
                episodic=reward(prefix['new150'], values['new300'], entrynew, values['old300'], reference),
                continuation_action=-1, reused=reused))
            write(root/'STATUS.json', dict(status='RUNNING', phase='SECOND_ACTION_ROLLOUTS', completed=action+1,
                                          physical=dict(ledger.count), query_image_evaluations=queries))
        assert dict(ledger.count) == {'audit':2600} and queries == 316 and len(rows) == 18
        assert sum(r['reused'] for r in rows) == 1
        write(root/'TRAINING_TABLE.private.json', rows)
        write(root/'REUSE_RECEIPT.json', dict(status='PASS', context=ctx[3], stream=stream,
              first_action=first_action, reused_second_action=selected['second_action'], reused_updates=100,
              source='same first-stage on-policy branch, same entry200/RNG/action, saved225/300 readouts', new_updates=2600))
    else:
        registered = read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts']
        index = cfg['context_index'];m,n,kind,factor = registered[index];ctx = (m,n,(kind,factor),f'dev{index}')
        t = b.make(cfg, roles, ledger, ctx, 300)
        entry = torch.load(original/f'jobs/development/{ctx[3]}/ENTRY.private.pt', map_location='cpu', weights_only=False)
        c.restore(t, entry);assert t.step == 100
        if cfg['job'] == 'train':
            assert read(campaign/'jobs/fit/FINAL.json')['status'] == 'COMPLETE'
            for method in NEW:
                model = [load_model(campaign/f'jobs/fit/{method}.private.pt')]
                c.restore(t, entry);t.category = 'development';t.key = ctx[3]+'/'+method;decisions = []
                generator = torch.Generator().manual_seed(860301+index)
                for _ in range(200):
                    if t.step in (100, 200):
                        t.action,row = select(t, model, generator);decisions.append(row)
                    t.update()
                    if t.step == 150:e.atomic_save(c.snapshot(t), root/(method+'_MID.private.pt'))
                    if t.step % 50 == 0:write(root/'STATUS.json', dict(status='RUNNING', method=method, step=t.step, physical=dict(ledger.count)))
                e.atomic_save(c.snapshot(t), root/(method+'_FINAL.private.pt'))
                write(root/(method+'_DECISIONS.private.json'), decisions)
                write(root/(method+'_TRAINING.json'), dict(status='SEALED', context=ctx[3], method=method,
                      updates=200, step=300, actions=[r['action'] for r in decisions]))
            assert dict(ledger.count) == {'development':1200} and queries == 0
        elif cfg['job'] == 'evaluate':
            assert read(campaign/'ENDPOINT_LOCK.json')['endpoints'] == 48
            old = [r for r in read(Path(cfg['v87'])/'DEVELOPMENT_RESULTS.json')['rows'] if r['context'] == ctx[3]]
            assert len({r['new_entry'] for r in old}) == len({r['old_memory_reference'] for r in old}) == 1
            entrynew,reference = old[0]['new_entry'],old[0]['old_memory_reference'];rows = []
            for method in NEW:
                src = campaign/f'jobs/train{index}'
                c.restore(t, torch.load(src/(method+'_MID.private.pt'), map_location='cpu', weights_only=False))
                mid = query(t, 'Q_dev_new', ctx[2], full=True)
                c.restore(t, torch.load(src/(method+'_FINAL.private.pt'), map_location='cpu', weights_only=False))
                new = query(t, 'Q_dev_new', ctx[2], full=True);old = query(t, 'Q_dev_old', ('identity', 1.), full=True)
                value = reward(mid['macro'], new['macro'], entrynew, old['macro'], reference)
                rows.append(dict(context=ctx[3], method=method, source='NEW', new50=mid, new=new, old=old,
                    new_entry=entrynew, old_memory_reference=reference, gain=value['gain'], forget_penalty=value['forget_penalty'],
                    utility=value['reward'], absolute_forgetting=reference-old['macro'], actions=read(src/(method+'_TRAINING.json'))['actions']))
            assert not ledger.count and queries == 72;write(root/'RESULTS.json', rows)
        else:raise ValueError('unknown GPU job')
    write(root/'FINAL.json', dict(status='COMPLETE', physical=dict(ledger.count), query_image_evaluations=queries, time=time.time()))


def actor_qualification(root, cfg):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.core import Actor
    torch.set_num_threads(2);helper = runpy.run_path(cfg['budget_helper'])
    budget = helper['Budget'](root, dict(actor_qualification=2))
    for rl in (False, True):
        actor = Actor(601);opt = torch.optim.Adam(actor.parameters(), lr=.001)
        z = torch.stack((torch.ones(24),-torch.ones(24)));adv = torch.tensor([[3.]+[0.]*8,[0.]*8+[3.]])
        prior = actor(z).detach().log_softmax(-1);before = float(actor_loss(actor,z,adv,prior,rl).detach())
        opt.zero_grad(set_to_none=True);actor_loss(actor,z,adv,prior,rl).backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.))
        budget.step('actor_qualification', 'RL' if rl else 'CE', opt)
        assert float(actor_loss(actor,z,adv,prior,rl).detach()) < before
    write(root/'QUALIFICATION.json',dict(status='PASS',objectives=['CE','KL_regularized_expected_return'],synthetic_updates=2))
    write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(budget.count),time=time.time()))


def fit_job(root, cfg):
    import torch
    torch.set_num_threads(2);helper = runpy.run_path(cfg['budget_helper'])
    assert read(Path(cfg['campaign'])/'jobs/actor_qualification/QUALIFICATION.json')['status'] == 'PASS'
    budget = helper['Budget'](root, dict(actor_fit=6144))
    collected = [r for index in range(8) for stream in (1, 2) for r in
                 read(Path(cfg['campaign'])/f'jobs/collect{index}_{stream}/TRAINING_TABLE.private.json')]
    assert len(collected) == 288
    keys = sorted({(r['context'],r['stream'],r['step']) for r in collected});assert len(keys) == 32
    groups = []
    for key in keys:
        group = sorted((r for r in collected if (r['context'],r['stream'],r['step']) == key), key=lambda r:r['action'])
        assert [r['action'] for r in group] == list(range(9)) and all(r['state'] == group[0]['state'] for r in group)
        groups.append(group)
    x = torch.tensor([g[0]['state'] for g in groups]);advantages = {}
    for kind in ('local', 'episodic'):
        returns = torch.tensor([[r[kind]['reward'] for r in g] for g in groups])
        advantages[kind] = ((returns-returns.mean(-1, keepdim=True))/returns.std(-1, unbiased=False, keepdim=True).clamp_min(FLOOR)).clamp(-3, 3)

    logs = [];summary = []
    for arm in ARMS:
        adv = advantages['local' if arm == 'LOCAL_CE' else 'episodic']
        for seed in (601, 602):
            actor,center,scale = load_model(cfg['behavior'][str(seed)]);z = (x-center)/scale
            prior = actor(z).detach().log_softmax(-1);opt = torch.optim.Adam(actor.parameters(), lr=.001)
            for update in range(1024):
                value = actor_loss(actor,z,adv,prior,arm == 'EPISODIC_RL');assert torch.isfinite(value)
                opt.zero_grad(set_to_none=True);value.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.))
                budget.step('actor_fit', f'{arm}/{seed}/{update}', opt)
                if update+1 in (1,64,1024):logs.append(dict(arm=arm,seed=seed,update=update+1,objective_before_update=float(value.detach())))
            with (root/f'{arm}_{seed}.private.pt').open('xb') as f:
                torch.save(dict(actor=actor.state_dict(),mean=center,scale=scale,seed=seed,arm=arm,updates=1024,source='V89'),f)
            with torch.no_grad():
                log = actor(z).log_softmax(-1);prob = log.exp()
                summary.append(dict(arm=arm,seed=seed,normalized_return=float((prob*adv).sum(-1).mean()),
                    KL_to_initial=float((prob*(log-prior)).sum(-1).mean()),entropy=float(-(prob*log).sum(-1).mean())))
    assert dict(budget.count) == {'actor_fit':6144}
    csvfile(root/'FIT_LOG.csv', logs);csvfile(root/'FIT_SUMMARY.csv', summary)
    flat = [dict(context=r['context'],stream=r['stream'],step=r['step'],action=r['action'],
                 continuation_action=r['continuation_action'],reused=r['reused'],
                 **{f'{kind}_{k}':v for kind in ('local','episodic') for k,v in r[kind].items()}) for r in collected]
    csvfile(root/'REWARD_SUMMARY.csv',flat)
    write(root/'QUALIFICATION.json',dict(status='PASS',synthetic_check='REUSED_EARLIER_ACTOR_QUALIFICATION',states=32,complete_actions=288))
    write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(budget.count),time=time.time()))


def audit_reuse(root, cfg):
    import torch
    previous = Path(cfg['v87']);assert read(previous/'FINAL.json')['status'] == 'COMPLETE'
    assert read(previous/'REUSE_AUDIT.json')['status'] == 'PASS'
    assert read(previous/'jobs/qualification/QUALIFICATION.json')['status'] == 'PASS'
    rows = [r for r in read(previous/'DEVELOPMENT_RESULTS.json')['rows'] if r['method'] in REUSED]
    assert len(rows) == 24
    for row in rows:
        context,method = row['context'],row['method'];index = int(context[-1])
        src = Path(cfg['original'])/f'jobs/development/{context}' if method in REUSED[:3] else previous/f'jobs/train{index}'
        receipt = read(src/(method+'_TRAINING.json'));assert receipt['status'] == 'SEALED'
        assert receipt['actions'] == row['actions']
        for suffix,step in (('MID',150),('FINAL',300)):
            value = torch.load(src/(method+'_'+suffix+'.private.pt'),map_location='cpu',weights_only=False)
            assert value['step'] == step and value['options']['total_steps'] == 300
            del value
    write(root/'REUSED_RESULTS.json',[dict(r,source='REUSED_V87') for r in rows])
    write(root/'REUSE_AUDIT.json',dict(status='PASS',endpoints=24,source='sealed V83/V87 checked receipts and150/300 states',new_updates=0))


def schedule(root, cfg, jobs, account, cpu=False):
    pending = list(jobs);active = {};failed = [];done = []
    while pending or active:
        for slot,(process,spec) in list(active.items()):
            if process.poll() is None:continue
            dest = root/'jobs'/spec['id'];write(dest/'PROCESS_EXIT.json',dict(exit_code=process.returncode,time=time.time()))
            (failed if process.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in ((-1,) if cpu else (5,6,7)):
                if not pending:break
                if slot in active:continue
                if not cpu:
                    free = int(subprocess.check_output(['nvidia-smi','--id='+str(slot),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free < 12000:continue
                spec = pending.pop(0);dest = root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=slot))
                env = dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],
                           EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='' if cpu else str(slot),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    process = subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c',
                        'import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],
                        cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(dest/'LAUNCH.json',dict(pid=process.pid,gpu=slot,time=time.time(),commit=cfg['commit']));active[slot]=process,spec
        account.refresh()
        write(root/'COSTS.json',dict(scope='V89_NEW_ONLY',attempts=dict(account.count),success=dict(account.success),
              failures=dict(account.failure),student_cap=46408,actor_cap=6146))
        write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',active=[s['id'] for _,s in active.values()],
              pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root, cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock = (root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit_reuse(root,cfg)
    actor_check = [dict(id='actor_qualification',job='actor_qualification',caps={})]
    qualification = [dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    collect = [dict(id=f'collect{i}_{s}',job='collect',context_index=i,stream=s,caps=dict(audit=2600)) for i in range(8) for s in (1,2)]
    fit = [dict(id='fit',job='fit',caps={})]
    train = [dict(id=f'train{i}',job='train',context_index=i,caps=dict(development=1200)) for i in range(4)]
    evaluate = [dict(id=f'eval{i}',job='evaluate',context_index=i,caps={}) for i in range(4)]
    jobs = actor_check+qualification+collect+fit+train+evaluate
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    caps = dict(qualification=8,audit=41600,actor_qualification=2,actor_fit=6144,development=4800)
    account = Accounting(root,caps=caps,jobs=jobs)
    schedule(root,cfg,actor_check,account,cpu=True)
    schedule(root,cfg,qualification,account);schedule(root,cfg,collect,account)
    assert account.count['audit'] == 41600
    schedule(root,cfg,fit,account,cpu=True);schedule(root,cfg,train,account)
    assert dict(account.count) == caps
    sealed = []
    for i in range(4):
        for method in NEW:
            src = root/f'jobs/train{i}';receipt = read(src/(method+'_TRAINING.json'))
            assert receipt['status'] == 'SEALED' and receipt['step'] == 300
            assert all((src/(method+'_'+s+'.private.pt')).is_file() for s in ('MID','FINAL'))
            sealed.append(dict(context=f'dev{i}',method=method))
    write(root/'ENDPOINT_LOCK.json',dict(endpoints=48,new=sealed,reused=24,time=time.time(),all_before_new_evaluation=True))
    schedule(root,cfg,evaluate,account)
    rows = read(root/'REUSED_RESULTS.json')
    for i in range(4):rows += read(root/f'jobs/eval{i}/RESULTS.json')
    assert len(rows) == len({(r['context'],r['method']) for r in rows}) == 48
    result = summarize(rows);write(root/'DECISION.json',result);write(root/'DEVELOPMENT_RESULTS.json',dict(rows=rows,**result))
    contexts = [dict(context=f'dev{i}',**summarize([r for r in rows if r['context'] == f'dev{i}'])) for i in range(4)]
    write(root/'CONTEXT_COMPARISONS.json',contexts)
    flat = [dict(context=r['context'],method=r['method'],source=r['source'],actions=str(r['actions']),
                 **{k:r[k] for k in ('gain','forget_penalty','utility','absolute_forgetting')},
                 **{f'{phase}_{k}':v for phase in ('new50','new','old') for k,v in r[phase].items()}) for r in rows]
    csvfile(root/'ENDPOINT_RESULTS.csv',flat)
    qtrain = sum(read(root/'jobs'/s['id']/'FINAL.json')['query_image_evaluations'] for s in collect)
    qdev = sum(read(root/'jobs'/s['id']/'FINAL.json')['query_image_evaluations'] for s in evaluate)
    assert qtrain == 5056 and qdev == 288
    write(root/'COSTS.json',dict(attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),
          student_total=46408,actor_total=6146,training_query_image_evaluations=qtrain,development_query_image_evaluations=qdev,
          new_unique_images=0,reused_development_endpoints=24,reused_training_branches=16))
    report = ['# V8.9 episodic policy improvement','','**'+result['status']+'**','',
              '| Method | New | Old | Utility | Absolute forgetting |','|---|---:|---:|---:|---:|']
    for m,v in result['means'].items():report.append('| '+m+' | '+' | '.join(f'{v[k]:.9f}' for k in ('new','old','utility','absolute_forgetting'))+' |')
    report += ['','The primary tests incremental RL gain over identical-data episodic distillation and all frozen controls.',
        'Secondary distillation gains do not replace a failed primary. All48 endpoints were sealed before new evaluation.',
        'This is one full-action policy-improvement step on repeatedly observed D1 development, not GRPO or independent generalization.',
        'Private states/weights remain on NAS; V84 remains NOT_RUN. All costs and query counts are in COSTS.json.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    write(root/'FINAL.json',dict(status='COMPLETE',decision=result['status'],physical=dict(account.count),publication='PENDING',time=time.time()))
    write(root/'STATUS.json',read(root/'FINAL.json'))


if __name__ == '__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK') == '1':print('PASS: return and primary-gate self-check')
    else:
        root = Path(os.environ['EXEC_RUN']);cfg = read(os.environ['EXEC_CONFIG'])
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']),f)
        try:
            if os.environ.get('EXEC_MODE') != 'job':coordinator(root,cfg)
            elif cfg['job'] == 'actor_qualification':actor_qualification(root,cfg)
            elif cfg['job'] == 'fit':fit_job(root,cfg)
            else:gpu_job(root,cfg)
        except BaseException as exc:
            write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
