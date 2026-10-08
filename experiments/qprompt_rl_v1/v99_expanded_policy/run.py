"""Training-only expanded-action policy fit and qualified cached trajectory readout."""
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

CAPS = dict(qualification=8, audit=14400, actor_fit=4096)
ARMS = ('EXPANDED_CE', 'EXPANDED_RL')
NEW = tuple(f'{a}_{s}' for a in ('EXPANDED_INIT',)+ARMS for s in (601, 602))+('UNIFORM_12', 'GLOBAL_12', 'TIME_12')


def model12(seed):
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c
    class Expanded(c.Actor):
        def forward(self, z, memory=True):
            logits = self.net(z)
            if not memory:logits = logits.masked_fill(torch.arange(12, device=z.device) >= 3, -torch.inf)
            return logits
    actor = Expanded(seed)
    with torch.random.fork_rng(devices=[]):actor.net[-1] = torch.nn.Linear(32, 12)
    assert sum(p.numel() for p in actor.parameters()) == 1196
    return actor


def expand(actor, seed):
    import torch
    state = actor.state_dict();state['net.2.weight'] = torch.cat((state['net.2.weight'], state['net.2.weight'][6:]), 0)
    state['net.2.bias'] = torch.cat((state['net.2.bias'], state['net.2.bias'][6:]), 0)
    state['net.2.bias'][6:] -= math.log(2)
    result = model12(seed);result.load_state_dict(state);result.eval();return result


def load12(path):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    actor = model12(value['seed']);actor.load_state_dict(value['actor']);actor.eval()
    assert value['mean'].shape == value['scale'].shape == (24,) and (value['scale'] > 0).all()
    return actor, value['mean'], value['scale']


def advantages(returns):
    original = returns[:, :9];center = original.mean(-1, keepdim=True)
    scale = (original-center).square().mean().sqrt().clamp_min(B.FLOOR)
    return ((returns-center)/scale).clamp(-3, 3), scale


def summarize(rows):
    keys = ('new', 'old', 'utility', 'gain', 'forget_penalty', 'absolute_forgetting')
    methods = sorted({r['method'] for r in rows})
    means = {m:{k:st.mean(r[k]['macro'] if k in ('new', 'old') else r[k] for r in rows if r['method'] == m) for k in keys} for m in methods}
    for arm in sorted({m.rsplit('_', 1)[0] for m in methods if m.endswith(('_601', '_602'))}):
        means[arm] = {k:st.mean(means[f'{arm}_{s}'][k] for s in (601, 602)) for k in keys}
    primary = means['EXPANDED_RL'];controls = [m for m in means if not m.endswith(('_601', '_602')) and m != 'EXPANDED_RL']
    delta = {m:{k:primary[k]-means[m][k] for k in ('new', 'old', 'utility')} for m in controls}
    paired = {f'{m}_{s}':means[f'EXPANDED_RL_{s}']['utility']-means[f'{m}_{s}']['utility'] for m in ('EXPANDED_CE', 'EXPANDED_INIT') for s in (601, 602)}
    trade = {m:(delta[m]['new'] >= .002 and delta[m]['old'] >= -.0025) or (delta[m]['old'] >= .005 and delta[m]['new'] >= -.0025) for m in ('UNIFORM_12', 'CONTINUE_CE', 'POOLED_RL', 'EXPANDED_CE')}
    passed = all(v['utility'] >= .0005 for v in delta.values()) and all(v > 0 for v in paired.values()) and all(trade.values())
    return dict(status='PASS_V99_EXPANDED_RL_CACHED_D1_ONLY' if passed else 'NO_V99_EXPANDED_RL_INCREMENTAL_GAIN', means=means, primary_deltas=delta, paired_seed_utility=paired, tradeoff=trade, automatic_research_stop=False, independent_confirmation=False, readout='REUSED_GRID_OUTCOMES', V84='NOT_RUN')


def selfcheck():
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c
    old = c.Actor(601)
    with torch.no_grad():old.net[-1].bias.copy_(torch.linspace(-1, 1, 9))
    before = torch.get_rng_state();actor = expand(old, 601);assert torch.equal(before, torch.get_rng_state())
    x = torch.linspace(-1, 1, 120).reshape(5, 24);p = actor(x).softmax(-1);p9 = old(x).softmax(-1)
    assert torch.allclose(p[:, :6], p9[:, :6], atol=1e-7) and torch.allclose(p[:, 6:9]+p[:, 9:], p9[:, 6:], atol=1e-7)
    assert torch.equal(actor(x, memory=False).softmax(-1)[:, 3:], torch.zeros(5, 9))
    returns = torch.linspace(-1, 2, 60).reshape(5, 12);adv, scale = advantages(returns)
    centered = returns[:, :9]-returns[:, :9].mean(-1, keepdim=True)
    assert torch.equal(adv[:, :9], (centered/centered.square().mean().sqrt().clamp_min(B.FLOOR)).clamp(-3, 3))
    prior = actor(x).detach().log_softmax(-1)
    for rl in (False, True):
        actor.zero_grad(set_to_none=True);loss = B.actor_loss(actor, x, adv, prior, rl);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in actor.parameters())
    methods = NEW+tuple(f'{a}_{s}' for a in ('CONTINUE_CE', 'POOLED_RL') for s in (601, 602))
    rows = [dict(method=m, new={'macro':.803 if m.startswith('EXPANDED_RL') else .8}, old={'macro':.8}, utility=.003 if m.startswith('EXPANDED_RL') else 0., gain=0., forget_penalty=0., absolute_forgetting=0.) for m in methods]
    assert summarize(rows)['status'].startswith('PASS')
    assert summarize([dict(r, utility=0.) for r in rows])['status'].startswith('NO_')
    assert summarize([dict(r, new={'macro':.8}) for r in rows])['status'].startswith('NO_')


def gpu_job(root, cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    if cfg['job'] == 'qualification':
        E.G = G;E.B = B;E.gpu_job(root, cfg);return
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    roles = c.split_roles(cfg['data']);assert roles == G.read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger = b.JobLedger(root, cfg['caps']);queries = 0;i = cfg['context_index'];campaign = Path(cfg['campaign'])
    def restore(t, path, step):
        value = torch.load(path, map_location='cpu', weights_only=False)
        assert value['step'] == step and value['options']['total_steps'] == 300
        c.restore(t, value);return value
    def query(t, role, condition):
        nonlocal queries
        assert role in ('Q_train_new', 'Q_train_old')
        queries += len(roles[role]);record = dict(role=role, step=t.step, image_calls_upper_bound=queries)
        c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='attempt', **record))
        try:value = scores(t, roles, role, condition)
        except BaseException:
            c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='failure', **record));raise
        c.e.append(root/'QUERY_LEDGER.jsonl', dict(event='success', **record));return value['macro']
    if cfg['job'] == 'collect':
        E.install_actions(c);ctx = b.contexts()[i];stream = cfg['stream'];t = b.make(cfg, roles, ledger, ctx, 300)
        t.provider.seed = 168+10000*stream;src = Path(cfg['v92'])/f'jobs/collect{i}_{stream}'
        entry = restore(t, src/'ENTRY100_STREAM.private.pt', 100);oldrows = G.read(src/'TRAINING_TABLE.private.json');behavior = G.read(src/'BEHAVIOR.private.json')
        models = [B.load_model(cfg['behavior'][str(s)]) for s in (601, 602)]
        generator = torch.Generator().manual_seed(860901+100*stream+i)
        first, firstrow = B.select(t, models, generator);afterfirst = generator.get_state()
        assert firstrow == behavior['first'] and first == behavior['first']['action']
        old100 = [r for r in oldrows if r['step'] == 100];old200 = [r for r in oldrows if r['step'] == 200]
        assert len(old100) == len(old200) == 9 and all(r['state'] == firstrow['state'] for r in old100)
        originals = [json.loads(x) for x in (Path(cfg['original'])/'jobs/audit_1/ACTION_ROWS.jsonl').read_text().splitlines()]
        refs = [r for r in originals if r['context'] == ctx[3] and r['entry_step'] == 100]
        assert len(refs) == 9 and len({r['new_entry'] for r in refs}) == 1
        newentry = refs[0]['new_entry'];oldentry = oldrows[0]['old_entry'];assert all(r['old_entry'] == oldentry for r in oldrows)
        t.category = 'audit';rows = []
        def row(step, action, state, gain, oldfinal, continuation):
            return dict(context=ctx[3], stream=stream, step=step, action=action, state=state, gain=gain, old_entry=oldentry, old_final=oldfinal, dense_reward=gain+oldfinal-oldentry, continuation_action=continuation, reused=False)
        for action in range(9, 12):
            c.restore(t, entry);t.action = action;t.key = f'{ctx[3]}/{stream}/first{action}';paired = torch.Generator();paired.set_state(afterfirst)
            values = {};continuation = None
            for _ in range(200):
                if t.step == 200:continuation, _ = B.select(t, models, paired);t.action = continuation
                t.update()
                if t.step in (150, 300):values[t.step] = query(t, 'Q_train_new', ctx[2])
                if t.step == 300:oldfinal = query(t, 'Q_train_old', ('identity', 1.))
            gain = .25*(values[150]-newentry)+.75*(values[300]-newentry)
            rows.append(row(100, action, firstrow['state'], gain, oldfinal, continuation))
            c.e.atomic_save(c.snapshot(t), root/f'FIRST_{action}_FINAL.private.pt')
            G.write(root/'STATUS.json', dict(status='RUNNING', rows=len(rows), physical=dict(ledger.count), query_image_evaluations=queries))
        restore(t, src/f'FIRST_{first}_FINAL.private.pt', 300);baseline = query(t, 'Q_train_new', ctx[2])
        basegain = next(r['gain'] for r in old100 if r['action'] == first)
        assert next(r['gain'] for r in old200 if r['action'] == behavior['second_action']) == basegain
        retained = restore(t, src/'ON_POLICY_ENTRY200.private.pt', 200);state200 = t.extract().tolist()
        assert all(r['state'] == state200 for r in old200)
        for action in range(9, 12):
            c.restore(t, retained);t.action = action;t.key = f'{ctx[3]}/{stream}/second{action}'
            for _ in range(100):t.update()
            newfinal = query(t, 'Q_train_new', ctx[2]);oldfinal = query(t, 'Q_train_old', ('identity', 1.))
            rows.append(row(200, action, state200, basegain+.75*(newfinal-baseline), oldfinal, -1))
            c.e.atomic_save(c.snapshot(t), root/f'SECOND_{action}_FINAL.private.pt')
            G.write(root/'STATUS.json', dict(status='RUNNING', rows=len(rows), physical=dict(ledger.count), query_image_evaluations=queries))
        assert len(rows) == 6 and dict(ledger.count) == {'audit':900} and queries == 64
        G.write(root/'TRAINING_TABLE.private.json', rows)
        G.write(root/'REUSE_RECEIPT.json', dict(status='PASS', old_rows=18, new_rows=6, old_state100_and200_exact=True, behavior_exact=True, common_prefix_return='oldgain+.75*(newfinal-oldselectedfinal)', baseline_training_new_final=baseline, new_updates=900, query_image_evaluations=64))
    else:
        m, n, kind, factor = G.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
        t = b.make(cfg, roles, ledger, (m, n, (kind, factor), f'dev{i}'), 300)
        entry = restore(t, Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt', 100)
        def prefix(first):
            src = cfg['v97'] if first < 9 else cfg['v98']
            restore(t, Path(src)/f'jobs/train{i}/FIRST_{first}_PREFIX.private.pt', 200)
        def actor_choices(models):
            c.restore(t, entry);generator = torch.Generator().manual_seed(860301+i)
            first, a = B.select(t, models, generator);prefix(first);second, z = B.select(t, models, generator)
            return [first, second], [a, z]
        if cfg['job'] == 'replay':
            audits = []
            for arm in ('POOLED_CE', 'POOLED_RL'):
                for seed in (601, 602):
                    method = f'{arm}_{seed}';actions, trace = actor_choices([B.load_model(Path(cfg['v94'])/f'jobs/fit/{method}.private.pt')])
                    expected = G.read(Path(cfg['v94'])/f'jobs/train{i}/{method}_DECISIONS.private.json')
                    assert trace == expected, 'cached state/probability/action replay mismatch'
                    audits.append(dict(context=f'dev{i}', method=method, actions=actions, state_probability_action_exact=True))
            G.write(root/'REPLAY_AUDIT.json', audits)
        elif cfg['job'] == 'select':
            assert G.read(campaign/'FIT_LOCK.json')['actors'] == 6
            controls = G.read(campaign/'jobs/fit/TRAINING_CONTROLS.json');choices = [];traces = []
            for method in NEW:
                if method.startswith('EXPANDED_'):actions, trace = actor_choices([load12(campaign/f'jobs/fit/{method}.private.pt')])
                else:
                    c.restore(t, entry);generator = torch.Generator().manual_seed(860301+i);actions = [];trace = []
                    for step in (100, 200):
                        before = c.snapshot(t)
                        action = int(torch.multinomial(torch.ones(12)/12, 1, generator=generator)) if method == 'UNIFORM_12' else controls[method][str(step)]
                        assert c.e.same(before, c.snapshot(t));actions.append(action)
                        if step == 100:prefix(action)
                choices.append(dict(context=f'dev{i}', method=method, actions=actions));traces.append(dict(method=method, trace=trace))
            assert len(choices) == 9;G.write(root/'CHOICES.json', choices);G.write(root/'TRACES.private.json', traces)
        else:raise ValueError('unknown GPU job')
        assert not ledger.count and queries == 0
    G.write(root/'FINAL.json', dict(status='COMPLETE', physical=dict(ledger.count), query_image_evaluations=queries, time=time.time()))


def fit_job(root, cfg):
    import torch
    torch.set_num_threads(2);rows = []
    for i in range(8):
        for stream in (1, 2):
            old = G.read(Path(cfg['v92'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json')
            new = G.read(Path(cfg['campaign'])/f'jobs/collect{i}_{stream}/TRAINING_TABLE.private.json');assert len(old) == 18 and len(new) == 6
            rows += old+new
    keys = sorted({(r['context'], r['stream'], r['step']) for r in rows});groups = []
    for key in keys:
        g = sorted((r for r in rows if (r['context'], r['stream'], r['step']) == key), key=lambda r:r['action'])
        assert [r['action'] for r in g] == list(range(12)) and all(r['state'] == g[0]['state'] for r in g);groups.append(g)
    assert len(groups) == 32 and len(rows) == 384 and all(math.isfinite(r['dense_reward']) for r in rows)
    x = torch.tensor([g[0]['state'] for g in groups]);returns = torch.tensor([[r['dense_reward'] for r in g] for g in groups]);adv, reward_scale = advantages(returns)
    assert float(reward_scale) == G.read(Path(cfg['v94'])/'jobs/fit/SCALE_QUALIFICATION.json')['pooled_scale']
    means = returns.mean(0);global_action = int(means.argmax())
    controls = {'GLOBAL_12':{str(step):global_action for step in (100, 200)}, 'TIME_12':{str(step):int(returns[[k[2] == step for k in keys]].mean(0).argmax()) for step in (100, 200)}}
    G.write(root/'TRAINING_CONTROLS.json', controls)
    budget = runpy.run_path(cfg['budget_helper'])['Budget'](root, dict(actor_fit=4096));logs = [];summary = []
    for seed in (601, 602):
        original, center, scale = B.load_model(Path(cfg['v90'])/f'jobs/fit/DENSE_CE_{seed}.private.pt');initial = expand(original, seed);z = (x-center)/scale
        prior = initial(z).detach().log_softmax(-1)
        with (root/f'EXPANDED_INIT_{seed}.private.pt').open('xb') as f:torch.save(dict(actor=initial.state_dict(), mean=center, scale=scale, seed=seed, updates=0), f)
        for arm in ARMS:
            actor = expand(original, seed);assert all(torch.equal(v, initial.state_dict()[k]) for k,v in actor.state_dict().items())
            opt = torch.optim.Adam(actor.parameters(), lr=.001)
            for update in range(1024):
                loss = B.actor_loss(actor, z, adv, prior, arm == 'EXPANDED_RL');assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
                budget.step('actor_fit', f'{arm}/{seed}/{update}', opt)
                if update+1 in (1, 64, 1024):logs.append(dict(arm=arm, seed=seed, update=update+1, objective_before_update=float(loss.detach())))
            with (root/f'{arm}_{seed}.private.pt').open('xb') as f:torch.save(dict(actor=actor.state_dict(), mean=center, scale=scale, seed=seed, arm=arm, updates=1024), f)
            with torch.no_grad():
                log = actor(z).log_softmax(-1);prob = log.exp()
                summary.append(dict(arm=arm, seed=seed, normalized_return=float((prob*adv).sum(-1).mean()), raw_dense_return=float((prob*returns).sum(-1).mean()), KL_to_reference=float((prob*(log-prior)).sum(-1).mean()), entropy=float(-(prob*log).sum(-1).mean()), new_action_mass=float(prob[:, 9:].sum(-1).mean())))
    assert dict(budget.count) == {'actor_fit':4096}
    B.csvfile(root/'FIT_LOG.csv', logs);B.csvfile(root/'FIT_SUMMARY.csv', summary)
    B.csvfile(root/'REWARD_SUMMARY.csv', [{k:v for k,v in r.items() if k != 'state'} for r in rows])
    G.write(root/'FIT_QUALIFICATION.json', dict(status='PASS', states=32, actions=384, new_actions=96, reused_actions=288, parameters=1196, reward_scale=float(reward_scale), old_advantages_unchanged=True, matched_initialization=True, extra_optimizer_updates=0, clipped_new_values=int((adv[:, 9:].abs() == 3).sum())))
    G.write(root/'FINAL.json', dict(status='COMPLETE', physical=dict(budget.count), time=time.time()))


def schedule(root,cfg,jobs,account,cpu=False):
    pending=list(jobs);active={};failed=[];done=[]
    while pending or active:
        for slot,(p,spec) in list(active.items()):
            if p.poll() is None:continue
            dest=root/'jobs'/spec['id'];G.write(dest/'PROCESS_EXIT.json',dict(exit_code=p.returncode,time=time.time()))
            (failed if p.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in ((-1,) if cpu else (5,6,7)):
                if not pending:break
                if slot in active:continue
                if not cpu and int(subprocess.check_output(['nvidia-smi','--id='+str(slot),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12000:continue
                spec=pending.pop(0);dest=root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                G.write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=slot))
                env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='' if cpu else str(slot),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    p=subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                G.write(dest/'LAUNCH.json',dict(pid=p.pid,gpu=slot,time=time.time(),commit=cfg['commit']));active[slot]=p,spec
        account.refresh()
        G.write(root/'COSTS.json',dict(scope='V99_NEW_ONLY',attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_cap=14408,actor_cap=4096))
        G.write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',active=[s['id'] for _,s in active.values()],pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root, cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock = (root/'COORDINATOR.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert G.read(Path(cfg['v98'])/'FINAL.json')['status'] == 'COMPLETE'
    assert G.read(Path(cfg['v98'])/'ENDPOINT_LOCK.json')['endpoints'] == 576
    previous = G.read(Path(cfg['v96'])/'DEVELOPMENT_RESULTS.json')['rows']
    assert len(previous) == len({(r['context'], r['method']) for r in previous}) == 168
    for i in range(8):
        for s in (1, 2):
            src = Path(cfg['v92'])/f'jobs/collect{i}_{s}'
            assert G.read(src/'PROCESS_EXIT.json')['exit_code'] == 0 and G.read(src/'FINAL.json')['status'] == 'COMPLETE'
            assert len(G.read(src/'TRAINING_TABLE.private.json')) == 18
    G.write(root/'REUSED_RESULTS.json', [dict(r, source='REUSED_V96') for r in previous])
    G.write(root/'REUSE_AUDIT.json', dict(status='PASS', historical_policy_rows=168, reused_training_rewards=288, available_grid_outcomes=576, new_updates=0))
    replays = [dict(id=f'replay{i}', job='replay', context_index=i, caps={}) for i in range(4)]
    quals = [dict(id='qualification', job='qualification', caps=dict(qualification=8))]
    collects = [dict(id=f'collect{i}_{s}', job='collect', context_index=i, stream=s, caps=dict(audit=900)) for i in range(8) for s in (1, 2)]
    fits = [dict(id='fit', job='fit', caps={})]
    selects = [dict(id=f'select{i}', job='select', context_index=i, caps={}) for i in range(4)]
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account = Accounting(root, caps=CAPS, jobs=replays+quals+collects+fits+selects)
    schedule(root, cfg, replays, account)
    audits = sum((G.read(root/f'jobs/replay{i}/REPLAY_AUDIT.json') for i in range(4)), [])
    assert len(audits) == 16 and all(r['state_probability_action_exact'] for r in audits)
    G.write(root/'CACHED_READOUT_QUALIFICATION.json', dict(status='PASS', policy_replays=16, exact_state_probability_action=True, optimizer_updates=0, query_accesses=0, rows=audits))
    schedule(root, cfg, quals, account);schedule(root, cfg, collects, account)
    assert dict(account.count) == dict(qualification=8, audit=14400)
    schedule(root, cfg, fits, account, cpu=True);assert dict(account.count) == CAPS
    assert all((root/f'jobs/fit/{m}.private.pt').is_file() for m in NEW if m.startswith('EXPANDED_'))
    G.write(root/'FIT_LOCK.json', dict(actors=6, fitted=4, initial=2, training_controls_sealed=True, time=time.time()))
    schedule(root, cfg, selects, account)
    choices = sum((G.read(root/f'jobs/select{i}/CHOICES.json') for i in range(4)), [])
    assert len(choices) == len({(r['context'], r['method']) for r in choices}) == 36
    G.write(root/'CHOICE_LOCK.json', dict(status='SEALED', choices=choices, time=time.time(), before_outcome_lookup=True))
    grid = G.read(Path(cfg['v98'])/'GRID_RESULTS.json')['rows'];lookup = {(r['context'], r['first'], r['second']):r for r in grid}
    assert len(lookup) == 576;rows = G.read(root/'REUSED_RESULTS.json')
    for choice in choices:
        first, second = choice['actions'];result = lookup[(choice['context'], first, second)]
        rows.append(dict(result, method=choice['method'], actions=choice['actions'], source='REUSED_GRID_OUTCOME'))
    assert len(rows) == len({(r['context'], r['method']) for r in rows}) == 204
    result = summarize(rows);G.write(root/'DECISION.json', result);G.write(root/'DEVELOPMENT_RESULTS.json', dict(rows=rows, **result))
    G.write(root/'CONTEXT_COMPARISONS.json', [dict(context=f'dev{i}', **summarize([r for r in rows if r['context'] == f'dev{i}'])) for i in range(4)])
    B.csvfile(root/'ENDPOINT_RESULTS.csv', [dict(context=r['context'], method=r['method'], source=r['source'], actions=str(r['actions']), **{k:r[k] for k in ('gain', 'forget_penalty', 'utility', 'absolute_forgetting')}, **{f'{phase}_{k}':v for phase in ('new50', 'new', 'old') for k,v in r[phase].items()}) for r in rows])
    qt = sum(G.read(root/'jobs'/s['id']/'FINAL.json')['query_image_evaluations'] for s in collects);assert qt == 1024
    assert all(G.read(root/'jobs'/s['id']/'FINAL.json').get('query_image_evaluations', 0) == 0 for s in replays+quals+selects)
    G.write(root/'COSTS.json', dict(attempts=dict(account.count), success=dict(account.success), failures=dict(account.failure), student_total=14408, actor_total=4096, training_query_image_evaluations=qt, development_query_image_evaluations=0, cached_policy_rows=36, cached_image_evaluation_equivalents=432, historical_policy_rows=168, reused_training_rewards=288, new_training_rewards=96, new_unique_images=0))
    report = ['# V99 expanded-action policy learning', '', result['status'], '', '| Method | New | Old | Utility |', '|---|---:|---:|---:|']
    for m, v in result['means'].items():report.append('| '+m+' | '+' | '.join(f'{v[k]:.9f}' for k in ('new', 'old', 'utility'))+' |')
    report += ['', 'Training-only expanded rewards, matched CE/RL. All36choices sealed before cached metric lookup. Zero new development optimizer updates or image queries. The complete finite grid was already observed; this is repeated-D1 cached development, not independent confirmation. Old gates remain diagnostics and do not automatically stop future research.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    G.write(root/'FINAL.json', dict(status='COMPLETE', decision=result['status'], physical=dict(account.count), publication='PENDING', time=time.time()))
    G.write(root/'STATUS.json', G.read(root/'FINAL.json'))


def load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path);module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root = Path(os.environ['EXEC_RUN'])
    G = load('grid_base', cfg['grid_entry']);B = load('reward_base', cfg['base_entry']);E = load('expanded_actions', cfg['expanded_entry'])
    if os.environ.get('EXEC_SELFCHECK') == '1':
        import torch
        torch.set_num_threads(2);G.selfcheck();selfcheck()
        print('PASS12actor mass/RNG/memorymask/gradients,oldadvantage identity,primary positive/negative; zero optimizer/query')
    else:
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(), time=time.time(), commit=cfg['commit']), f)
        try:
            if os.environ.get('EXEC_MODE') != 'job':coordinator(root, cfg)
            elif cfg['job'] == 'fit':fit_job(root, cfg)
            else:gpu_job(root, cfg)
        except BaseException as exc:
            G.write(root/'STATUS.json', dict(status='ENGINEERING_STOP', error=repr(exc), traceback=traceback.format_exc(), time=time.time()));raise
