"""Frozen policy distributions on cached native trajectories; zero training and queries."""
import fcntl
import json
import math
import os
import random
import statistics as st
import subprocess
import time
import traceback
from pathlib import Path

KEYS = ('new', 'old', 'utility', 'gain', 'forget_penalty', 'absolute_forgetting')


def normalized(values):
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    total = sum(values);assert abs(total-1) < 1e-6
    return [v/total for v in values]


def weights(first, second):
    n = len(first);assert len(second) == n and all(len(row) == n for row in second)
    a = normalized(first);b = [normalized(row) for row in second]
    result = [a[i]*b[i][j] for i in range(n) for j in range(n)]
    assert abs(sum(result)-1) < 1e-12
    return result


def greedy(first, second):
    a = max(range(len(first)), key=lambda i:first[i])
    return [a, max(range(len(second[a])), key=lambda j:second[a][j])]


def mean_outcome(distribution, outcomes):
    assert len(distribution) == len(outcomes)
    return {k:sum(p*r[k] for p,r in zip(distribution, outcomes)) for k in KEYS}


def selfcheck():
    assert weights([.25, .75], [[1., 0.], [0., 1.]]) == [.25, 0., 0., .75]
    assert greedy([.5, .5], [[.5, .5], [0., 1.]]) == [0, 0]
    rows = [{k:float(i) for k in KEYS} for i in range(4)]
    assert mean_outcome(weights([.5, .5], [[.5, .5]]*2), rows)['utility'] == 1.5
    assert mean_outcome(weights([0., 1.], [[.5, .5], [1., 0.]]), rows) == rows[2]
    assert mean_outcome(weights([.25, .75], [[1., 0.], [0., 1.]]), rows)['utility'] == 2.25
    try:normalized([-.1, 1.1])
    except AssertionError:pass
    else:raise AssertionError('negative probability accepted')


def gpu_job(root, cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c, stage_b as b
    torch.set_num_threads(2);torch.cuda.set_device(0);random.seed(168);np.random.seed(168);torch.manual_seed(168)
    roles = c.split_roles(cfg['data']);assert roles == G.read(Path(cfg['action_root'])/'ROLES.private.json')
    i = cfg['context_index'];prior = Path(cfg['v99']);ledger = b.JobLedger(root, {})
    m, n, kind, factor = G.read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
    t = b.make(cfg, roles, ledger, (m, n, (kind, factor), f'dev{i}'), 300)
    methods = [m for m in V.NEW if m.startswith('EXPANDED_')]
    models = {m:V.load12(prior/f'jobs/fit/{m}.private.pt') for m in methods}
    states = {};probs = {m:{} for m in methods}
    for key in [-1]+list(range(12)):
        path = Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt' if key == -1 else Path(cfg['v97'] if key < 9 else cfg['v98'])/f'jobs/train{i}/FIRST_{key}_PREFIX.private.pt'
        value = torch.load(path, map_location='cpu', weights_only=False);assert value['step'] == (100 if key == -1 else 200)
        c.restore(t, value);before = c.snapshot(t);state = t.extract();states[key] = state.tolist()
        for method in methods:probs[method][key] = B.probabilities(state, [models[method]]).tolist()
        assert c.e.same(before, c.snapshot(t)), 'state/probability extraction changed native state or RNG'
    controls = G.read(prior/'jobs/fit/TRAINING_CONTROLS.json')
    traces = {r['method']:r['trace'] for r in G.read(prior/f'jobs/select{i}/TRACES.private.json')}
    expected = {r['method']:r['actions'] for r in G.read(prior/f'jobs/select{i}/CHOICES.json')}
    result = [];audits = []
    for method in V.NEW:
        if method in models:first = probs[method][-1];second = [probs[method][a] for a in range(12)]
        elif method == 'UNIFORM_12':first = [1/12]*12;second = [first[: ] for _ in range(12)]
        else:
            first = [float(a == controls[method]['100']) for a in range(12)]
            second = [[float(a == controls[method]['200']) for a in range(12)] for _ in range(12)]
        generator = torch.Generator().manual_seed(860301+i)
        if method in models or method == 'UNIFORM_12':
            a = int(torch.multinomial(torch.tensor(first), 1, generator=generator));z = int(torch.multinomial(torch.tensor(second[a]), 1, generator=generator));actions = [a, z]
        else:actions = [controls[method]['100'], controls[method]['200']]
        assert actions == expected[method]
        if method in models:
            a, z = actions
            assert traces[method] == [dict(step=100, state=states[-1], probabilities=first, action=a), dict(step=200, state=states[a], probabilities=second[a], action=z)]
        result.append(dict(context=f'dev{i}', method=method, first=first, second=second, argmax=greedy(first, second), sampled_replay=actions))
        audits.append(dict(context=f'dev{i}', method=method, sampled_actions_exact=True, state_probabilities_exact=method in models, native_rng_unchanged=True))
    assert not ledger.count
    G.write(root/'DISTRIBUTIONS.json', result);G.write(root/'REPLAY_AUDIT.json', audits)
    G.write(root/'STATES.private.json', states)
    G.write(root/'FINAL.json', dict(status='COMPLETE', physical={}, query_image_evaluations=0, actor_probability_vectors=78, extracted_states=13, sampled_policy_replays=9, time=time.time()))


def summarize(rows, historical):
    modes = sorted({r['mode'] for r in rows});means = {}
    for mode in modes:
        group = [r for r in rows if r['mode'] == mode];methods = sorted({r['method'] for r in group})
        means[mode] = {m:{k:st.mean(r[k] for r in group if r['method'] == m) for k in KEYS} for m in methods}
        for arm in ('EXPANDED_INIT', 'EXPANDED_CE', 'EXPANDED_RL'):
            means[mode][arm] = {k:st.mean(means[mode][f'{arm}_{s}'][k] for s in (601,602)) for k in KEYS}
    g = means['FIXED_ARGMAX'];e = means['EXACT_T1_EXPECTATION'];primary = g['EXPANDED_RL']
    controls = {m:g[m] for m in ('EXPANDED_CE','EXPANDED_INIT','GLOBAL_12','TIME_12')}
    controls.update(UNIFORM_12_EXPECTATION=e['UNIFORM_12'], CONTINUE_CE=historical['CONTINUE_CE'], POOLED_RL=historical['POOLED_RL'])
    delta = {m:{k:primary[k]-r[k] for k in ('new','old','utility')} for m,r in controls.items()}
    paired = {f'{arm}_{s}':g[f'EXPANDED_RL_{s}']['utility']-g[f'{arm}_{s}']['utility'] for arm in ('EXPANDED_CE','EXPANDED_INIT') for s in (601,602)}
    practical = {m:(delta[m]['new']>=.002 and delta[m]['old']>=-.0025) or (delta[m]['old']>=.005 and delta[m]['new']>=-.0025) for m in ('EXPANDED_CE','EXPANDED_INIT','UNIFORM_12_EXPECTATION','CONTINUE_CE')}
    passed = all(v['utility']>=.0005 for v in delta.values()) and all(v>0 for v in paired.values()) and all(practical.values())
    expected_delta = {m:{k:e['EXPANDED_RL'][k]-e[m][k] for k in ('new','old','utility')} for m in ('EXPANDED_CE','EXPANDED_INIT','UNIFORM_12','GLOBAL_12','TIME_12')}
    return dict(status='PASS_V100_ARGMAX_CURRENT_D1_ONLY' if passed else 'NO_V100_ARGMAX_RL_INCREMENTAL_GAIN',means=means,primary_deltas=delta,paired_seed_utility=paired,practical=practical,expected_deltas=expected_delta,independent_confirmation=False,automatic_research_stop=False)


def coordinator(root, cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock = (root/'COORDINATOR.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior = Path(cfg['v99']);assert G.read(prior/'FINAL.json')['status']=='COMPLETE'
    jobs = [dict(id=f'extract{i}',job='extract',context_index=i,caps={}) for i in range(4)]
    (root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account = Accounting(root, caps={}, jobs=jobs)
    for spec in jobs:
        free = int(subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=12000
        dest = root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
        G.write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=5))
        env = dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES='5',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = ['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")']
        with (dest/'worker.log').open('x') as log:
            process = subprocess.Popen(command,cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            G.write(dest/'LAUNCH.json',dict(pid=process.pid,gpu=5,time=time.time(),commit=cfg['commit']))
            G.write(root/'STATUS.json',dict(status='RUNNING',active=[spec['id']],physical={},time=time.time()))
            code = process.wait()
        G.write(dest/'PROCESS_EXIT.json',dict(exit_code=code,time=time.time()));account.refresh()
        assert code==0 and G.read(dest/'FINAL.json')['status']=='COMPLETE'

    assert not account.count and not account.failure
    distributions = sum((G.read(root/f'jobs/extract{i}/DISTRIBUTIONS.json') for i in range(4)),[])
    assert len(distributions)==len({(r['context'],r['method']) for r in distributions})==36
    G.write(root/'DISTRIBUTION_LOCK.json',dict(status='SEALED',before_metric_lookup=True,distributions=distributions,time=time.time()))
    grid = G.read(Path(cfg['v98'])/'GRID_RESULTS.json')['rows'];lookup={(r['context'],r['first'],r['second']):r for r in grid};assert len(lookup)==576
    historical = G.read(prior/'DEVELOPMENT_RESULTS.json');sample={(r['context'],r['method']):r for r in historical['rows']}
    rows=[]
    def flat(r):return {k:r[k]['macro'] if k in ('new','old') else r[k] for k in KEYS}
    for d in distributions:
        context,method=d['context'],d['method'];outcomes=[flat(lookup[(context,a,b)]) for a in range(12) for b in range(12)]
        expected=mean_outcome(weights(d['first'],d['second']),outcomes);a,b=d['argmax'];g=flat(lookup[(context,a,b)])
        sampled=sample[(context,method)];assert sampled['actions']==d['sampled_replay']
        for mode,value in [('EXACT_T1_EXPECTATION',expected),('FIXED_ARGMAX',g),('ORIGINAL_V99_T1_SAMPLE',flat(sampled))]:rows.append(dict(context=context,method=method,mode=mode,**value))
    assert len(rows)==108;decision=summarize(rows,historical['means'])
    G.write(root/'DECISION.json',decision);G.write(root/'READOUT_RESULTS.json',dict(rows=rows,**decision));B.csvfile(root/'READOUT_RESULTS.csv',rows)
    G.write(root/'COSTS.json',dict(student_updates=0,actor_updates=0,training_image_evaluations=0,development_image_evaluations=0,new_unique_images=0,unique_cached_grid_rows=576,weighted_policy_rows=36,argmax_policy_rows=36,original_sampled_rows=36,actor_probability_vectors=312,extracted_states=52,failed_updates=0))
    G.write(root/'FINAL.json',dict(status='COMPLETE',decision=decision['status'],physical={},publication='PENDING',time=time.time()));G.write(root/'STATUS.json',G.read(root/'FINAL.json'))


if __name__ == '__main__':
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS conditional weights,pointmass,uniform,tie rule,invalid probability; zero update/query')
    else:
        cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
        import importlib.util
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        G=load('grid_base',cfg['grid_entry']);B=load('reward_base',cfg['base_entry']);V=load('policy_base',cfg['v99_entry']);V.G=G
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']),f)
        try:
            if os.environ.get('EXEC_MODE')=='job':gpu_job(root,cfg)
            else:coordinator(root,cfg)
        except BaseException as exc:
            G.write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
