"""Finite paired intervention, using the existing trainer, queue and evaluator."""
import fcntl
import gc
import time
import traceback
from pathlib import Path
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.qprompt_rl_v1.v3d_u_dose import runner as dose
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS,DOMAINS,WEIGHTS,OPTIONS,ACTIVE,H,CAPS,PREDECESSOR_COMMIT,check,decision
from .teacher import bind

ROOT,C=io.ROOT,io.C


def entry(seed,domain,ledger):
    payload=torch.load(Path(C['predecessor'])/f'S{seed}_{domain}'/'entry_state.pt',map_location='cpu',weights_only=False)
    assert payload['commit']==PREDECESSOR_COMMIT and payload['state']['step']==payload['state']['cursor']==0
    t=e.create(C,seed,domain,False,ledger);e.restore(t,payload['state'])
    return bind(t),payload['state']


def smoke(ledger,cost):
    checks,timings={},{}
    for domain in DOMAINS:
        t,initial=entry(SEEDS[0],domain,ledger)
        for method,weight in WEIGHTS.items():
            ledger.smoke_context=domain+'/'+method
            e.restore(t,initial);t.options.update(lambda_U=weight,**OPTIONS[method])
            checks[domain+'/'+method]=io.check_native(t,ledger,cost)
            start=time.time();e.restore(t,initial)
            for k in range(5):ledger.update(t,'smoke',f'timing/{k}',-1 if weight==0 else 2)
            unchanged=e.same(initial['ema'],e.cpu(t.ema.state_dict()))
            assert unchanged==OPTIONS[method]['freeze_U_teacher']
            key='fixed_teacher_skips' if OPTIONS[method]['freeze_U_teacher'] else 'dynamic_teacher_updates'
            assert t.telemetry[key]==5
            checks[domain+'/'+method]['teacher_unchanged']=unchanged
            io.save_state(t,ROOT/'smoke_latest.pt',method=method,**OPTIONS[method]);torch.cuda.synchronize()
            timings[domain+'/'+method]=time.time()-start
        del t,initial;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke']==80
    estimate=1.25*len(SEEDS)*(H/5)*sum(timings.values())+1800
    e.write(ROOT/'THROUGHPUT.json',dict(block_seconds=timings,estimate_seconds=estimate,remaining_seconds=ledger.session['optimizer_deadline']-time.time()))
    assert estimate<ledger.session['optimizer_deadline']-time.time(), 'full matrix does not fit'
    e.write(ROOT/'INTEGRATION_CHECK.json',dict(status='PASS',smoke_calls=80,checks=checks))


def teacher_audit():
    rows=[]
    for seed in SEEDS:
        for domain in DOMAINS:
            cell=ROOT/f'S{seed}_{domain}'
            start=torch.load(cell/'entry_state.pt',map_location='cpu',weights_only=False)
            assert start['commit']==C['commit']
            for method in WEIGHTS:
                payload=torch.load(cell/(method+'_latest.pt'),map_location='cpu',weights_only=False);s=payload['state']
                assert payload['commit']==C['commit'] and s['step']==s['cursor']==H and payload['method']==method
                unchanged=e.same(start['state']['ema'],s['ema']);fixed=OPTIONS[method]['freeze_U_teacher']
                assert unchanged==fixed
                skips=s['telemetry'].get('fixed_teacher_skips',0);updates=s['telemetry'].get('dynamic_teacher_updates',0)
                assert skips==(H if fixed else 0) and updates==(0 if fixed else H)
                rows.append(dict(seed=seed,domain=domain,method=method,teacher_unchanged=unchanged,skips=skips,updates=updates,active_U_calls=payload['active_U_calls'],teacher_decay=OPTIONS[method]['teacher_decay']))
    e.write(ROOT/'TEACHER_AUDIT.json',dict(status='PASS',rows=rows))


def main():
    check();torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists() and not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'explicit recovery binding required'
    ledger=dose.Ledger(ROOT);ledger.caps=dict(CAPS)
    cost=dict(sequence_start='V3D-U-DOSE',previous_optimizer_calls=189120,previous_diagnostic_VJPs=800,reused_source_history=40000)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3I-TEACHER-TIMESCALE',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS,domains=DOMAINS,weights=WEIGHTS,method_options=OPTIONS,expected_active_U=ACTIVE,H=H,caps=CAPS,
        smoke_calls=80,source_updates_reused=40000,new_source_updates=0,session=ledger.session,
        evidence='development-only EMA .999 vs .99 teacher timescale intervention; reused seeds and validation patients',
        primary='SLOW_05-FINE_05; joint-positive means, >=4/5 joint seeds, no domain-mean regression',
        data_role_audit='V3E FINAL_INTERPRETATION; val development only, test remains closed'))
    try:
        dose.status('INTEGRATION_SMOKE');smoke(ledger,cost)
        dose.status('MAIN_MATRIX',H=H);endpoints=dose.matrix(ledger,entry_factory=entry)
        teacher_audit();dose.status('EVALUATION');rows,summary=dose.report(endpoints,ledger,cost)
        import csv
        with (Path(C['predecessor'])/'ENDPOINTS.csv').open() as f:old={(int(r['seed']),r['domain'],r['method']):r for r in csv.DictReader(f)}
        controls=[dict(seed=r['seed'],domain=r['domain'],method=r['method'],**{k:r[k]-float(old[r['seed'],r['domain'],r['method']][k]) for k in ('macro_Dice','old_REFUGE')}) for r in rows if r['method']!='SLOW_05']
        e.write(ROOT/'CONTROL_REPLICATION.json',dict(rows=controls,max_abs=max(abs(r[k]) for r in controls for k in ('macro_Dice','old_REFUGE'))))
        assert len(controls)==30 and all(all(r[k]==0 for k in ('macro_Dice','old_REFUGE')) for r in controls), 'control differs from V3H'
        result=decision(rows);e.write(ROOT/'TIMESCALE_DECISION.json',result)
        summary['timescale_intervention']=result;e.write(ROOT/'SUMMARY.json',summary)
        cost['sequence_new_optimizer_calls']=189120+sum(ledger.count.values());e.write(ROOT/'ALL_COSTS.json',cost)
        with (ROOT/'FINAL_INTERPRETATION.md').open('a') as f:f.write('\n## Frozen teacher timescale comparison\n\n'+e.json.dumps(result,indent=2)+'\n')
        dose.status('COMPLETE_PRIVATE',endpoints=len(endpoints),H=H,timescale_intervention_supported=result['timescale_intervention_supported'])
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(cost,physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']))
        dose.status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):main()
