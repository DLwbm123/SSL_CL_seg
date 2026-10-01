"""Finite dose calibration using sealed entry states and the existing trainer."""
import csv
import gc
import os
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS, DOMAINS, WEIGHTS, H, CAPS, PREDECESSOR_COMMIT, check

ROOT, C = io.ROOT, io.C


class Ledger(e.Ledger):
    smoke_context = ''

    def call(self, category, key, fn):
        if category == 'smoke':key = self.smoke_context+'/'+key
        return super().call(category, key, fn)


def status(name, **extra):
    e.write(ROOT/'status.json', dict(status=name, pid=os.getpid(), time=time.time(), **extra))


def entry(seed, domain, ledger):
    path = Path(C['predecessor'])/f'S{seed}_{domain}'/'entry_state.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['commit'] == PREDECESSOR_COMMIT
    assert payload['state']['step'] == payload['state']['cursor'] == 0
    t = e.create(C, seed, domain, False, ledger)
    e.restore(t, payload['state'])
    return t, payload['state']


def smoke(ledger, cost):
    checks, timings = {}, {}
    for domain in DOMAINS:
        t, initial = entry(SEEDS[0], domain, ledger)
        for method, weight in WEIGHTS.items():
            ledger.smoke_context = domain+'/'+method
            e.restore(t, initial);t.options['lambda_U'] = weight
            checks[domain+'/'+method] = io.check_native(t, ledger, cost)
            started = time.time()
            e.restore(t, initial)
            for k in range(5):ledger.update(t, 'smoke', f'timing/{k}', -1 if weight == 0 else 2)
            io.save_state(t, ROOT/'smoke_latest.pt', method=method, u_weight=weight)
            torch.cuda.synchronize();timings[domain+'/'+method] = time.time()-started
        ledger.smoke_context = domain+'/zero_alias'
        branches = []
        for weight, action in ((0., 2), (.5, -1)):
            e.restore(t, initial);t.options['lambda_U'] = weight
            for k in range(5):ledger.update(t, 'smoke', f'{weight}/{k}', action)
            state = e.snapshot(t);state['action'] = -1;branches.append(state)
        assert e.same(*branches), 'zero weight must reproduce native L-only state'
        checks[domain+'/zero_alias'] = True
        del t, initial, branches;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke'] == 100
    # Conservative serial estimate includes a durable save every five steps.
    estimate = 1.25*len(SEEDS)*(H/5)*sum(timings.values())+1200
    remaining = ledger.session['optimizer_deadline']-time.time()
    e.write(ROOT/'THROUGHPUT.json', dict(block_seconds=timings, estimate_seconds=estimate, remaining_seconds=remaining))
    assert estimate < remaining, 'fixed matrix does not fit; no automatic horizon reduction'
    e.write(ROOT/'INTEGRATION_CHECK.json', dict(status='PASS', checks=checks, smoke_calls=100))
    lock = dict(protocol='V3D-U-DOSE', commit=C['commit'], predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS, domains=DOMAINS, weights=WEIGHTS, H=H, caps=CAPS, checkpoints=[0,300,600,1200],
        source_updates_reused=24000, new_source_updates=0, evidence='development calibration',
        queue='seed/domain, round-robin methods in five-step blocks', session=ledger.session)
    e.write(ROOT/'RUN_LOCK.json', lock)
    return lock


def matrix(ledger, entry_factory=entry):
    lock = e.read(ROOT/'RUN_LOCK.json')
    seeds, weights = lock['seeds'], lock['weights']
    endpoints = []
    for seed in seeds:
        for domain in DOMAINS:
            t, initial = entry_factory(seed, domain, ledger)
            cell = ROOT/f'S{seed}_{domain}';cell.mkdir(exist_ok=False)
            io.export(t, cell/'entry.pt');io.save_state(t, cell/'entry_state.pt', phase='main_entry', predecessor_commit=lock.get('predecessor_commit'))
            states = {m:initial for m in weights};active_calls = {m:0 for m in weights}
            for step in range(0, H, 5):
                for method, weight in weights.items():
                    e.restore(t, states[method]);t.options.update(lambda_U=weight,u_start=0,u_stop=H)
                    t.options.update(lock.get('method_options',{}).get(method,{}))
                    assert t.step == step and t.cursor == step
                    for k in range(5):
                        expected_active = bool(t.u_weight())
                        ledger.update(t, 'main', f'S{seed}/{domain}/{method}/{step+k}', -1 if weight == 0 else 2)
                        assert t.last['active_U']==expected_active, 'U window execution mismatch'
                        active_calls[method] += int(expected_active)
                    states[method] = e.snapshot(t)
                    if t.step % 100 == 0:io.save_state(t, cell/(method+'_latest.pt'), method=method, u_weight=weight, u_start=t.options['u_start'],u_stop=t.options['u_stop'],active_U_calls=active_calls[method])
                    if t.step in (300, 600, H):io.export(t, cell/f'{method}_{t.step}.pt')
                e.write(ROOT/'MATRIX_PROGRESS.json', dict(seed=seed, domain=domain, all_methods_step=step+5, H=H, physical=dict(ledger.count), time=time.time()))
            endpoints.extend(dict(seed=seed, domain=domain, method=m, lambda_U=w, step=H,
                path=str(cell/f'{m}_{H}.pt'), entry=str(cell/'entry.pt'),active_U_calls=active_calls[m],
                u_start=lock.get('method_options',{}).get(m,{}).get('u_start',0),u_stop=lock.get('method_options',{}).get(m,{}).get('u_stop',H)) for m,w in weights.items())
            e.write(ROOT/'ENDPOINT_REGISTRY.json', endpoints)
            del t, states, initial;gc.collect();torch.cuda.empty_cache()
    return endpoints


def report(endpoints, ledger, cost):
    lock = e.read(ROOT/'RUN_LOCK.json')
    seeds, weights = lock['seeds'], lock['weights']
    expected_endpoints = len(seeds)*len(DOMAINS)*len(weights)
    seconds = sum(io.evaluate_file(Path(p)) for p in dict.fromkeys([r['entry'] for r in endpoints]+[r['path'] for r in endpoints]))
    rows = []
    for r in endpoints:
        out = e.read(Path(r['path']).with_suffix('.scores.json'))['scores']
        start = e.read(Path(r['entry']).with_suffix('.scores.json'))['scores'];d = r['domain']
        rows.append({k:r[k] for k in ('seed','domain','method','lambda_U','step','u_start','u_stop','active_U_calls') if k in r} | dict(
            macro_Dice=out[d]['macro_Dice'], rim=out[d]['rim'], cup=out[d]['cup'], disc_union=out[d]['disc_union'],
            entry_macro=start[d]['macro_Dice'], old_REFUGE=out['REFUGE']['macro_Dice'],
            old_change=out['REFUGE']['macro_Dice']-start['REFUGE']['macro_Dice']))
    lookup = {(r['seed'],r['domain'],r['method']):r for r in rows}
    for r in rows:
        for ref in ('ORIGINAL','FINE_05'):
            for metric in ('macro_Dice','old_REFUGE'):
                r[f'{metric}_delta_vs_{ref}'] = r[metric]-lookup[r['seed'],r['domain'],ref][metric]
    with (ROOT/'ENDPOINTS.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means = {m:{metric:float(np.mean([r[metric] for r in rows if r['method']==m]))
        for metric in ('macro_Dice','old_REFUGE','old_change')} for m in weights}
    seedmeans = {m:{str(s):{metric:float(np.mean([r[metric] for r in rows if r['method']==m and r['seed']==s]))
        for metric in ('macro_Dice','old_REFUGE')} for s in seeds} for m in weights}
    comparisons = {}
    for m in weights:
        comparisons[m] = {}
        for ref in ('ORIGINAL','FINE_05'):
            comparisons[m][ref] = {}
            for metric in ('macro_Dice','old_REFUGE'):
                diffs = [seedmeans[m][str(s)][metric]-seedmeans[ref][str(s)][metric] for s in seeds]
                comparisons[m][ref][metric] = dict(seed_deltas=diffs, mean=float(np.mean(diffs)), sample_std=float(np.std(diffs,ddof=1)),
                    negative_cells=[dict(seed=r['seed'],domain=r['domain'],delta=r[f'{metric}_delta_vs_{ref}']) for r in rows if r['method']==m and r[f'{metric}_delta_vs_{ref}']<0])
    metrics = ('macro_Dice','old_REFUGE')
    pareto = [m for m in weights if not any(all(means[n][k]>=means[m][k] for k in metrics) and any(means[n][k]>means[m][k] for k in metrics) for n in weights if n!=m)]
    # Historical scores are read only when an exact predecessor replay was specified.
    replication = []
    if C.get('predecessor'):
        with (Path(C['predecessor'])/'ENDPOINTS.csv').open() as f:old = {(int(r['seed']),r['domain'],r['method']):r for r in csv.DictReader(f)}
        replication = [dict(seed=r['seed'],domain=r['domain'],method=r['method'],
            **{k:r[k]-float(old[r['seed'],r['domain'],C.get('predecessor_method_map',{'FINE_05':'FINE','ORIGINAL':'ORIGINAL'})[r['method']]][k]) for k in metrics})
            for r in rows if r['method'] in ('ORIGINAL','FINE_05')]
    summary = dict(means=means, seed_means=seedmeans, paired_comparisons=comparisons,
        descriptive_pareto=pareto, predecessor_endpoint_deltas=replication, evidence=lock['evidence'])
    e.write(ROOT/'SUMMARY.json', summary)
    events = [e.json.loads(line) for line in (ROOT/'PHYSICAL_LEDGER.jsonl').read_text().splitlines()]
    attempts = Counter((v['category'],v['key']) for v in events if v['event']=='attempt')
    successes = Counter((v['category'],v['key']) for v in events if v['event']=='success')
    expected = {('main',f'S{s}/{d}/{m}/{k}') for s in seeds for d in DOMAINS for m in weights for k in range(H)}
    assert attempts == successes and all(v==1 for v in attempts.values())
    assert {k for k in attempts if k[0]=='main'} == expected
    assert len(rows)==expected_endpoints and ledger.count['main']==ledger.caps['main'] and ledger.count['smoke']==lock.get('smoke_calls',100)
    assert ledger.count['source']==ledger.caps['source']
    if ledger.count['source']:
        assert {k for k in attempts if k[0]=='source'} == {('source',f'S{s}/{k}') for s in seeds for k in range(8000)}
        for seed in seeds:
            receipt=e.read(Path(C['source'])/f'SOURCE_S{seed}'/'receipt.json')
            assert receipt['status']=='SEALED' and receipt['step']==8000 and receipt['commit']==C['commit']
    assert all(np.isfinite(r[k]) for r in rows for k in metrics)
    if lock.get('expected_active_U'):
        assert all(r['active_U_calls']==lock['expected_active_U'][r['method']] for r in rows)
    e.write(ROOT/'COMPLETION_AUDIT.json', dict(status='PASS', endpoints=expected_endpoints, scores=2*(expected_endpoints+len(seeds)*len(DOMAINS)),
        unique_main_successes=len(expected), physical_calls=dict(ledger.count), failures=0,
        predecessor_commit=lock.get('predecessor_commit'), new_source_updates=ledger.count['source'], reused_source_updates=lock.get('source_updates_reused',0)))
    cost.update(physical_calls=dict(ledger.count), evaluation_seconds=seconds,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(), wall_seconds=time.time()-ledger.session['start'])
    e.write(ROOT/'ALL_COSTS.json', cost)
    text = f"# {lock['protocol']}\n\n{expected_endpoints}/{expected_endpoints} fixed {H}-step endpoints; {len(seeds)} seeds and two domains. {lock['evidence']}.\n\n"
    text += '| Method | U weight | New Dice | Old REFUGE Dice | Old change |\n|---|---:|---:|---:|---:|\n'
    text += ''.join(f'| {m} | {weights[m]} | {v["macro_Dice"]:.6f} | {v["old_REFUGE"]:.6f} | {v["old_change"]:.6f} |\n' for m,v in means.items())
    text += '\nDescriptive mean Pareto set: '+', '.join(pareto)+'. No statistical noninferiority or significance claim. All paired seed results, negative cells, and historical endpoint differences are retained in SUMMARY.json and ENDPOINTS.csv.\n\n'
    text += 'The joint new/old result is primary; there is no post-hoc weighted winner. Reused validation patients limit generalization claims. Final 1200-step student only; old-domain change is not full-sequence BWT. Historical KI identity remains unverified. Subsequent experiments require a separately frozen protocol.\n'
    (ROOT/'FINAL_INTERPRETATION.md').write_text(text)
    return rows, summary


def main():
    import fcntl
    check();torch.set_num_threads(2);torch.cuda.set_device(0)
    lockfile = (ROOT/'EXECUTOR.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'explicit recovery binding required'
    assert not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'no implicit replay'
    ledger = Ledger(ROOT);ledger.caps = dict(CAPS);cost = {}
    try:
        status('INTEGRATION_SMOKE');smoke(ledger,cost)
        status('MAIN_MATRIX',H=H);endpoints = matrix(ledger)
        status('EVALUATION');report(endpoints,ledger,cost);status('COMPLETE_PRIVATE',endpoints=len(endpoints),H=H)
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(cost,physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']))
        status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__ == '__main__':
    with NativeOperations(ROOT/'operations'):main()
