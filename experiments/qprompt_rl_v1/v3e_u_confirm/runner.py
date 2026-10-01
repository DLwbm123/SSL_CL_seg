"""Reuse source training, dose matrix, and evaluator with a new frozen cohort."""
import fcntl
import gc
import os
import time
import traceback
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.qprompt_rl_v1.v3c_rule_timing.runner import source
from experiments.qprompt_rl_v1.v3d_u_dose import runner as dose
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS, DOMAINS, WEIGHTS, H, CAPS, check, decision

ROOT, C = io.ROOT, io.C


def smoke(ledger, cost):
    checks, timings = {}, {}
    for domain in DOMAINS:
        t = e.create(dict(C,source=C['smoke_source']),165,domain,False,ledger)
        initial = e.snapshot(t)
        for method, weight in WEIGHTS.items():
            ledger.smoke_context=domain+'/'+method
            e.restore(t,initial);t.options['lambda_U']=weight
            checks[domain+'/'+method]=io.check_native(t,ledger,cost)
            start=time.time();e.restore(t,initial)
            for k in range(5):ledger.update(t,'smoke',f'timing/{k}',-1 if weight==0 else 2)
            io.save_state(t,ROOT/'smoke_latest.pt',method=method,u_weight=weight)
            torch.cuda.synchronize();timings[domain+'/'+method]=time.time()-start
        del t, initial;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke']==60
    target_estimate=1.25*len(SEEDS)*(H/5)*sum(timings.values())
    # One second per source update is deliberately conservative until seed 168 is measured.
    e.write(ROOT/'THROUGHPUT.json',dict(block_seconds=timings,conservative_target_seconds=target_estimate,
        source_upper_seconds=CAPS['source'],evaluation_reserve_seconds=1800))
    assert target_estimate+1800 < ledger.session['optimizer_deadline']-time.time(), 'target matrix alone does not fit'
    e.write(ROOT/'INTEGRATION_CHECK.json',dict(status='PASS',checks=checks,smoke_calls=60,commit=C['commit']))
    return target_estimate


def fresh_entry(seed, domain, ledger):
    # Independent domains and methods share the exact state after deterministic construction.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);e.random.seed(seed);e.np.random.seed(seed)
    t=e.create(C,seed,domain,False,ledger)
    assert t.step==t.cursor==0
    return t,e.snapshot(t)


def main():
    check();torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists() and not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'explicit recovery binding required'
    ledger=dose.Ledger(ROOT);ledger.caps=dict(CAPS)
    cost=dict(sequence_start='V3D-U-DOSE',previous_rounds=[dict(protocol='V3D-U-DOSE',new_optimizer_calls=28900,reused_source_history=24000)])
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3E-U-CONFIRM',commit=C['commit'],seeds=SEEDS,domains=DOMAINS,
        weights=WEIGHTS,H=H,caps=CAPS,smoke_calls=60,source_updates_reused=0,new_source_updates=40000,
        smoke_source_updates_reused=8000,evidence='five new optimization seeds; reused validation patients',
        selection_predecessor='V3D-U-DOSE',session=ledger.session,checkpoints=[0,300,600,1200],
        primary='FINE_0125 vs ORIGINAL; both means positive, >=4/5 joint-positive seeds, no domain-mean regression in either metric',
        secondary='FINE_0125 vs FINE_05; descriptive tradeoff',no_automatic_retry=True))
    try:
        dose.status('INTEGRATION_SMOKE');target_estimate=smoke(ledger,cost)
        for seed in SEEDS:
            dose.status('SOURCE_PREPARATION',seed=seed);source(seed,ledger,cost)
            remaining_sources=len(SEEDS)-len(cost['sources'])
            observed=max(v['seconds'] for v in cost['sources'].values())
            estimate=1.25*remaining_sources*observed+target_estimate+1800
            e.write(ROOT/'RESOURCE_ADMISSION.json',dict(remaining_sources=remaining_sources,estimate_seconds=estimate,
                seconds_available=ledger.session['optimizer_deadline']-time.time(),time=time.time()))
            assert estimate < ledger.session['optimizer_deadline']-time.time(), 'full frozen matrix no longer fits; do not shrink or extend'
        dose.status('MAIN_MATRIX',H=H);endpoints=dose.matrix(ledger,entry_factory=fresh_entry)
        dose.status('EVALUATION');rows,summary=dose.report(endpoints,ledger,cost)
        cost['sequence_new_optimizer_calls']=28900+sum(ledger.count.values());e.write(ROOT/'ALL_COSTS.json',cost)
        result=decision(rows);e.write(ROOT/'CONFIRMATION_DECISION.json',result)
        summary['confirmation']=result;e.write(ROOT/'SUMMARY.json',summary)
        with (ROOT/'FINAL_INTERPRETATION.md').open('a') as f:
            f.write('\n## Frozen stability decision\n\n'+e.json.dumps(result,indent=2)+'\n')
        dose.status('COMPLETE_PRIVATE',endpoints=len(endpoints),H=H,stable_candidate=result['stable_candidate'])
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(cost,physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']))
        dose.status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):main()
