import fcntl
import gc
import time
import traceback
from pathlib import Path
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e, runner as io
from experiments.qprompt_rl_v1.v3d_u_dose import runner as dose
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS,DOMAINS,WEIGHTS,WINDOWS,ACTIVE,H,CAPS,PREDECESSOR_COMMIT,check,decision

ROOT,C=io.ROOT,io.C


def entry(seed,domain,ledger):
    p=Path(C['predecessor'])/f'S{seed}_{domain}'/'entry_state.pt'
    v=torch.load(p,map_location='cpu',weights_only=False)
    assert v['commit']==PREDECESSOR_COMMIT and v['state']['step']==v['state']['cursor']==0
    t=e.create(C,seed,domain,False,ledger);e.restore(t,v['state'])
    return t,v['state']


def smoke(ledger,cost):
    checks,timings={},{}
    for domain in DOMAINS:
        t,initial=entry(SEEDS[0],domain,ledger)
        for method,weight in WEIGHTS.items():
            ledger.smoke_context=domain+'/'+method
            e.restore(t,initial);t.options.update(lambda_U=weight,**WINDOWS[method])
            checks[domain+'/'+method]=io.check_native(t,ledger,cost)
            start=time.time();e.restore(t,initial)
            for k in range(5):ledger.update(t,'smoke',f'timing/{k}',-1 if weight==0 else 2)
            io.save_state(t,ROOT/'smoke_latest.pt',method=method,**WINDOWS[method]);torch.cuda.synchronize()
            timings[domain+'/'+method]=time.time()-start
        for method in ('EARLY_05','LATE_05'):
            # Engineering state only: cross the gate with real Adam and exact replay.
            # This synthetic cursor has no scientific endpoint or historical interpretation.
            ledger.smoke_context=domain+'/'+method+'/synthetic_boundary'
            e.restore(t,initial);t.step=t.cursor=599;t.options.update(lambda_U=.5,**WINDOWS[method])
            checks[domain+'/'+method+'/boundary']=io.check_native(t,ledger,cost)
        del t,initial;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke']==100
    estimate=1.25*len(SEEDS)*(H/5)*sum(timings.values())+1800
    e.write(ROOT/'THROUGHPUT.json',dict(block_seconds=timings,estimate_seconds=estimate,remaining_seconds=ledger.session['optimizer_deadline']-time.time()))
    assert estimate<ledger.session['optimizer_deadline']-time.time(), 'full matrix does not fit'
    e.write(ROOT/'INTEGRATION_CHECK.json',dict(status='PASS',smoke_calls=100,checks=checks))


def main():
    check();torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists() and not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'explicit recovery binding required'
    ledger=dose.Ledger(ROOT);ledger.caps=dict(CAPS)
    cost=dict(sequence_start='V3D-U-DOSE',previous_rounds=[dict(protocol='V3D-U-DOSE',new_optimizer_calls=28900),dict(protocol='V3E-U-CONFIRM',new_optimizer_calls=76060)],reused_source_history=40000)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3F-U-TIMING',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS,domains=DOMAINS,weights=WEIGHTS,method_options=WINDOWS,expected_active_U=ACTIVE,H=H,caps=CAPS,
        smoke_calls=100,source_updates_reused=40000,new_source_updates=0,session=ledger.session,
        evidence='development-only temporal ablation; reused optimization seeds and validation patients',
        primary='LATE_05-EARLY_05; joint-positive means, >=4/5 joint seeds, no domain-mean regression',
        data_role_audit='V3E FINAL_INTERPRETATION; val development only, test remains closed'))
    try:
        dose.status('INTEGRATION_SMOKE');smoke(ledger,cost)
        dose.status('MAIN_MATRIX',H=H);endpoints=dose.matrix(ledger,entry_factory=entry)
        dose.status('EVALUATION');rows,summary=dose.report(endpoints,ledger,cost)
        result=decision(rows);e.write(ROOT/'TIMING_DECISION.json',result)
        summary['timing']=result;e.write(ROOT/'SUMMARY.json',summary)
        cost['sequence_new_optimizer_calls']=104960+sum(ledger.count.values());e.write(ROOT/'ALL_COSTS.json',cost)
        with (ROOT/'FINAL_INTERPRETATION.md').open('a') as f:f.write('\n## Frozen temporal comparison\n\n'+e.json.dumps(result,indent=2)+'\n')
        dose.status('COMPLETE_PRIVATE',endpoints=len(endpoints),H=H,timing_supported=result['timing_supported'])
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(cost,physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']))
        dose.status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):main()
