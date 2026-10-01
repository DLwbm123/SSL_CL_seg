"""Finite V3C orchestration over unchanged V3B/native training engines."""
import csv
import gc
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io, policy
from experiments.lcrseg.five_frameworks_v1 import native_runner as native
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS, METHODS, DOMAINS, H, CAPS, phases, permute, check

ROOT, C = io.ROOT, io.C


def status(name, **extra):
    e.write(ROOT/'status.json', dict(status=name, pid=os.getpid(), time=time.time(), **extra))


def source(seed, ledger, cost):
    root = Path(C['source'])/f'SOURCE_S{seed}'
    root.mkdir(exist_ok=False)
    identity = dict(domain='REFUGE', seed=seed, order=1, stage=0,
                    manifest=e.primitive.MANIFEST_SHA, split=e.primitive.SPLIT_SHA)
    permit = e.ExecutionPermit(dict(execution_scope='formal', code_commit=C['commit'],
        authorized_manifest_digests=[e.digest(identity)]), ('V3C',), dict(student=CAPS['source']), e._PERMIT_SEAL)
    original_step, original_save = torch.optim.Adam.step, native.checkpoint.atomic_save
    calls = 0
    def charged_step(opt, *args, **kwargs):
        nonlocal calls
        result = ledger.call('source', f'S{seed}/{calls}', lambda: original_step(opt, *args, **kwargs))
        calls += 1
        if calls == 1 or calls % 25 == 0:
            e.write(ROOT/'progress.json', dict(phase='source', seed=seed, step=calls,
                    time=time.time(), physical=dict(ledger.count)))
        return result
    def save_full(payload, path):
        # The existing source loop is unchanged; add restoration state to its saves.
        return original_save(dict(payload, rng=e.cpu(e.rng_state()), cursor=payload['step'],
                                  code_commit=C['commit']), path)
    started = time.time()
    config = dict(C, execution_commit=C['commit'])
    with patch.object(torch.optim.Adam, 'step', charged_step), \
         patch.object(native.checkpoint, 'atomic_save', save_full), \
         patch.object(native, 'evaluate', lambda *args, **kwargs: ({}, {})):
        receipt = native.source_task(config, dict(id=f'SOURCE_S{seed}', seed=seed), permit,
                                     root, torch.device('cuda:0'))
    assert calls == 8000 and receipt['step'] == 8000
    receipt.update(validation_deferred=True, commit=C['commit'])
    e.write(root/'receipt.json', receipt)
    cost.setdefault('sources', {})[str(seed)] = dict(seconds=time.time()-started, updates=calls)
    e.write(ROOT/'ALL_COSTS.json', dict(cost, physical_calls=dict(ledger.count)))
    gc.collect(); torch.cuda.empty_cache()


def smoke(ledger, cost):
    checks, timings = {}, {}
    for domain in DOMAINS:
        t = e.create(dict(C, source=C['smoke_source']), 161, domain, False, ledger)
        checks[domain] = io.check_native(t, ledger, cost)
        entry = e.snapshot(t)
        for method in METHODS:
            e.restore(t, entry)
            started = time.time()
            action = -1 if method == 'ORIGINAL' else 2
            if method == 'EFFECT_RULE':
                z = io.extract(t, True, cost)
                action = policy.predict(dict(method='EFFECT_RULE'), [z])['actions'][0]
            for k in range(5):
                ledger.update(t, 'smoke', f'{domain}/{method}/{k}',
                              -1 if method == 'ORIGINAL' else action if k == 0 else 2)
            io.save_state(t, ROOT/'smoke_latest.pt', method=method)
            torch.cuda.synchronize()
            timings[domain+'/'+method] = time.time()-started
        del t, entry
        gc.collect(); torch.cuda.empty_cache()
    assert ledger.count['smoke'] == 50
    e.write(ROOT/'INTEGRATION_CHECK.json', dict(status='PASS', checks=checks,
            commit=C['commit'], smoke_calls=50))
    target_estimate = 1.25 * len(SEEDS) * (H/5) * sum(timings.values())
    e.write(ROOT/'THROUGHPUT.json', dict(block_seconds=timings,
        conservative_target_seconds=target_estimate, source_seconds='pending measured source progress',
        H=H, horizon_change_allowed=False))


def main_matrix(ledger, cost):
    endpoints = []
    for seed in SEEDS:
        source(seed, ledger, cost)
        for domain in DOMAINS:
            status('MAIN_MATRIX', seed=seed, domain=domain)
            t = e.create(C, seed, domain, False, ledger)
            entry = e.snapshot(t)
            states = {m: entry for m in METHODS}
            stats = {m: dict(actions=[0,0,0], decision_actions=[0,0,0]) for m in METHODS}
            cell = ROOT/f'S{seed}_{domain}'
            cell.mkdir(exist_ok=False)
            io.export(t, cell/'entry.pt')
            io.save_state(t, cell/'entry_state.pt', phase='main_entry')

            def block(method, step, action):
                assert t.step == step
                e.append(cell/(method+'_actions.private.jsonl'), dict(step=step, action=action))
                if action >= 0: stats[method]['decision_actions'][action] += 1
                for k in range(5):
                    a = -1 if method == 'ORIGINAL' else action if k == 0 else 2
                    ledger.update(t, 'main', f'S{seed}/{domain}/{method}/{step+k}', a)
                    if a >= 0: stats[method]['actions'][a] += 1
                states[method] = e.snapshot(t)
                if t.step % 100 == 0 or t.step in [b for _, b in phases(domain)]:
                    io.save_state(t, cell/(method+'_latest.pt'), method=method,
                                  stats=stats[method], phase_end=t.step)
                if t.step in (300,600,H): io.export(t, cell/f'{method}_{t.step}.pt')

            for start, end in phases(domain):
                rule_actions = []
                for step in range(start, end, 5):
                    for method in METHODS[:3]:
                        e.restore(t, states[method])
                        action = -1 if method == 'ORIGINAL' else 2
                        if method == 'EFFECT_RULE':
                            z = io.extract(t, True, cost)
                            action = int(policy.predict(dict(method=method), [z])['actions'][0])
                            rule_actions.append(action)
                        block(method, step, action)
                    e.write(ROOT/'MATRIX_PROGRESS.json', dict(seed=seed, domain=domain,
                        phase=[start,end], steps={m:states[m]['step'] for m in METHODS},
                        physical=dict(ledger.count), completed=len(endpoints), time=time.time()))
                schedule = permute(rule_actions, seed, domain, start, end)
                schedule.update(commit=C['commit'], seed=seed, domain=domain, frozen_at=time.time())
                e.write(cell/f'PERMUTATION_{start}_{end}.json', schedule)
                e.restore(t, states['PHASE_SHUFFLE'])
                assert t.step == start
                for step, action in zip(range(start,end,5), schedule['actions']):
                    block('PHASE_SHUFFLE', step, action)
                assert stats['EFFECT_RULE']['actions'] == stats['PHASE_SHUFFLE']['actions']
                assert all(states[m]['step'] == end for m in METHODS)
                e.write(ROOT/'MATRIX_PROGRESS.json', dict(seed=seed, domain=domain,
                    phase=[start,end], all_methods_step=end, H=H, completed=len(endpoints),
                    physical=dict(ledger.count), time=time.time()))
            for method in METHODS:
                endpoints.append(dict(seed=seed, domain=domain, method=method, step=H,
                    path=str(cell/f'{method}_{H}.pt'), entry=str(cell/'entry.pt'), **stats[method]))
            e.write(ROOT/'ENDPOINT_REGISTRY.json', endpoints)
            del t, states, entry
            gc.collect(); torch.cuda.empty_cache()
        status('SOURCE_PREPARATION', completed_sources=len(cost['sources']), completed_endpoints=len(endpoints))
    return endpoints


def audit(endpoints, ledger):
    expected = {f'S{s}/{d}/{m}/{i}' for s in SEEDS for d in DOMAINS for m in METHODS for i in range(H)}
    attempted, successful = Counter(), Counter()
    with (ROOT/'PHYSICAL_LEDGER.jsonl').open() as f:
        for line in f:
            row = e.json.loads(line)
            if row['category'] == 'main':
                (attempted if row['event']=='attempt' else successful)[row['key']] += 1
    assert set(attempted) == set(successful) == expected
    assert set(attempted.values()) == set(successful.values()) == {1}
    assert len(endpoints) == 24 and ledger.count['source'] == 24000
    matched = []
    for seed in SEEDS:
        for domain in DOMAINS:
            cell = ROOT/f'S{seed}_{domain}'
            logs = {m: [e.json.loads(x) for x in (cell/(m+'_actions.private.jsonl')).read_text().splitlines()]
                    for m in ('EFFECT_RULE','PHASE_SHUFFLE')}
            assert all([r['step'] for r in rows] == list(range(0,H,5)) for rows in logs.values())
            for start,end in phases(domain):
                rule = [r['action'] for r in logs['EFFECT_RULE'] if start <= r['step'] < end]
                control = [r['action'] for r in logs['PHASE_SHUFFLE'] if start <= r['step'] < end]
                receipt = e.read(cell/f'PERMUTATION_{start}_{end}.json')
                assert rule == receipt['rule_actions'] and control == receipt['actions']
                assert Counter(rule) == Counter(control)
                matched.append(dict(seed=seed, domain=domain, start=start, end=end,
                    counts=receipt['counts'], changed_positions=receipt['changed_positions'], matched=True))
    result = dict(status='PASS', cells=24, main_success=28800, source_success=24000,
                  complete_pairs=6, phases=matched, failed_main_calls=0)
    e.write(ROOT/'MATCH_AUDIT.json', result)
    return result


def report(endpoints, ledger, cost):
    rows=[]
    started=time.time()
    for path in dict.fromkeys([r['entry'] for r in endpoints]+[r['path'] for r in endpoints]):
        io.evaluate_file(Path(path))
    cost['evaluation_seconds']=time.time()-started
    for r in endpoints:
        out=e.read(Path(r['path']).with_suffix('.scores.json'))['scores']
        entry=e.read(Path(r['entry']).with_suffix('.scores.json'))['scores']
        rows.append({k:r[k] for k in ('seed','domain','method','step')} | dict(
            macro_Dice=out[r['domain']]['macro_Dice'], rim=out[r['domain']]['rim'],
            cup=out[r['domain']]['cup'], disc_union=out[r['domain']]['disc_union'],
            entry_macro=entry[r['domain']]['macro_Dice'], old_REFUGE=out['REFUGE']['macro_Dice'],
            old_change=out['REFUGE']['macro_Dice']-entry['REFUGE']['macro_Dice'],
            skip=r['actions'][0], coarse=r['actions'][1], fine=r['actions'][2], missing=False))
    lookup={(r['seed'],r['domain'],r['method']):r for r in rows}
    for r in rows:
        for ref in METHODS:
            r['delta_vs_'+ref]=r['macro_Dice']-lookup[r['seed'],r['domain'],ref]['macro_Dice']
    with (ROOT/'ENDPOINTS.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={m:float(np.mean([r['macro_Dice'] for r in rows if r['method']==m])) for m in METHODS}
    comparisons={}
    for ref in ('ORIGINAL','PHASE_SHUFFLE','FINE'):
        cells=[dict(seed=s, domain=d, delta=lookup[s,d,'EFFECT_RULE']['macro_Dice']-lookup[s,d,ref]['macro_Dice'],
                    old_delta=lookup[s,d,'EFFECT_RULE']['old_REFUGE']-lookup[s,d,ref]['old_REFUGE'])
               for s in SEEDS for d in DOMAINS]
        seeds=[float(np.mean([r['delta'] for r in cells if r['seed']==s])) for s in SEEDS]
        comparisons[ref]=dict(cells=cells, seed_deltas=dict(zip(map(str,SEEDS),seeds)),
            mean=float(np.mean(seeds)), sample_std=float(np.std(seeds,ddof=1)),
            positive_seeds=sum(x>0 for x in seeds), old_delta=float(np.mean([r['old_delta'] for r in cells])))
    eligible=all(comparisons[r]['mean']>0 and comparisons[r]['positive_seeds']>=2 for r in ('ORIGINAL','PHASE_SHUFFLE'))
    summary=dict(means=means, paired_comparisons=comparisons,
        descriptive_confirmation_candidate=eligible, significance_claim=False,
        old_tradeoff=comparisons['ORIGINAL']['old_delta']<0,
        negative_cells=[r for r in rows if r['method']=='EFFECT_RULE' and (r['delta_vs_ORIGINAL']<0 or r['delta_vs_PHASE_SHUFFLE']<0)])
    e.write(ROOT/'SUMMARY.json',summary)
    text='# V3C-RULE-TIMING\n\n24/24 fixed1200 endpoints; three optimization seeds on the existing validation population. No full-sequence CL or significance claim.\n\n'
    text+='Offline phase-matched permutation is a mechanism control, not an online method. One permutation per cell; unchanged/constant phases remain in MATCH_AUDIT.\n\n'
    text+='```json\n'+e.json.dumps(summary,indent=2)+'\n```\n\n'
    text+='Old-domain entry-to-endpoint change is not full-sequence BWT. All negative cells retained. No automatic next experiment.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text)
    cost.update(physical_calls=dict(ledger.count), peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                wall_seconds=time.time()-ledger.session['start'])
    e.write(ROOT/'ALL_COSTS.json',cost)
    status('COMPLETE_PRIVATE',endpoints=len(rows),H=H)


def main():
    import fcntl
    torch.set_num_threads(2);torch.cuda.set_device(0)
    lockfile=(ROOT/'EXECUTOR.lock').open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'no automatic partial-run restart'
    check()
    ledger=e.Ledger(ROOT);ledger.caps=dict(CAPS);cost={}
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3C-RULE-TIMING',commit=C['commit'],
        seeds=SEEDS,methods=METHODS,domains=DOMAINS,H=H,caps=CAPS,
        phases={d:phases(d) for d in DOMAINS},session=ledger.session,
        manifest=e.primitive.MANIFEST_SHA,split=e.primitive.SPLIT_SHA,
        primary=['EFFECT_RULE-ORIGINAL','EFFECT_RULE-PHASE_SHUFFLE'],no_automatic_retry=True))
    try:
        status('INTEGRATION_SMOKE');smoke(ledger,cost)
        status('SOURCE_PREPARATION');endpoints=main_matrix(ledger,cost)
        audit(endpoints,ledger);status('EVALUATION');report(endpoints,ledger,cost)
    except BaseException as exc:
        cost.update(physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start'])
        e.write(ROOT/'ALL_COSTS.json',cost)
        status('STOPPED',error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):
        main()
