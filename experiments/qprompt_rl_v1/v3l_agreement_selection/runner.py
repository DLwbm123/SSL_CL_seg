"""Selection ablation using the shared trainer, matrix, evaluator and accounting."""
import csv
import fcntl
import gc
import time
import traceback
from pathlib import Path
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e,runner as io
from experiments.qprompt_rl_v1.v3d_u_dose import runner as dose
from experiments.qprompt_rl_v1.v3i_teacher_timescale.teacher import bind as teacher_bind
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .protocol import SEEDS,DOMAINS,WEIGHTS,OPTIONS,ACTIVE,H,CAPS,PREDECESSOR_COMMIT,check,decision
from .selection import bind

ROOT,C=io.ROOT,io.C


def entry(seed,domain,ledger):
    payload=torch.load(Path(C['predecessor'])/f'S{seed}_{domain}'/'entry_state.pt',map_location='cpu',weights_only=False)
    assert payload['commit']==PREDECESSOR_COMMIT and payload['state']['step']==payload['state']['cursor']==0
    t=e.create(C,seed,domain,False,ledger);e.restore(t,payload['state'])
    return bind(teacher_bind(t),ledger,ROOT),payload['state']


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
            fixed=OPTIONS[method]['freeze_U_teacher'];unchanged=e.same(initial['ema'],e.cpu(t.ema.state_dict()))
            assert unchanged==fixed
            assert t.telemetry.get('selection_updates',0)==(5 if method in ('AGREE_05','RANDOM_05') else 0)
            checks[domain+'/'+method]['teacher_unchanged']=unchanged
            io.save_state(t,ROOT/'smoke_latest.pt',method=method,**OPTIONS[method]);torch.cuda.synchronize()
            timings[domain+'/'+method]=time.time()-start
        del t,initial;gc.collect();torch.cuda.empty_cache()
    assert ledger.count['smoke']==100
    estimate=1.25*len(SEEDS)*(H/5)*sum(timings.values())+1800
    e.write(ROOT/'THROUGHPUT.json',dict(block_seconds=timings,estimate_seconds=estimate,remaining_seconds=ledger.session['optimizer_deadline']-time.time()))
    assert estimate<ledger.session['optimizer_deadline']-time.time(), 'full matrix does not fit'
    e.write(ROOT/'INTEGRATION_CHECK.json',dict(status='PASS',smoke_calls=100,checks=checks))


def audit():
    teachers=[]
    for seed in SEEDS:
        for domain in DOMAINS:
            cell=ROOT/f'S{seed}_{domain}';start=torch.load(cell/'entry_state.pt',map_location='cpu',weights_only=False)
            assert start['commit']==C['commit']
            for method in WEIGHTS:
                p=torch.load(cell/(method+'_latest.pt'),map_location='cpu',weights_only=False);s=p['state'];fixed=OPTIONS[method]['freeze_U_teacher']
                assert p['commit']==C['commit'] and p['method']==method and s['step']==s['cursor']==H
                assert e.same(start['state']['ema'],s['ema'])==fixed
                assert s['telemetry'].get('fixed_teacher_skips',0)==(H if fixed else 0)
                assert s['telemetry'].get('dynamic_teacher_updates',0)==(0 if fixed else H)
                selected=s['telemetry'].get('selection_updates',0);assert selected==(H if OPTIONS[method]['u_selection']!='none' else 0)
                teachers.append(dict(seed=seed,domain=domain,method=method,teacher_fixed=fixed,selection_updates=selected,active_U=p['active_U_calls']))
    records=[e.json.loads(x) for x in (ROOT/'SELECTION_LEDGER.private.jsonl').read_text().splitlines()]
    lookup={(r['seed'],r['domain'],r['kind'],r['step']):r for r in records}
    expected={(s,d,k,i) for s in SEEDS for d in DOMAINS for k in ('agree','random') for i in range(H)}
    assert len(records)==len(lookup)==24000 and set(lookup)==expected
    aggregates=[]
    for seed in SEEDS:
        for domain in DOMAINS:
            total=[0,0,0];kept=[0,0,0];valid=0
            for i in range(H):
                a,b=lookup[seed,domain,'agree',i],lookup[seed,domain,'random',i]
                assert all(a[k]==b[k] for k in ('population','selected','valid_pixels'))
                valid+=a['valid_pixels']
                for c in range(3):
                    total[c]+=sum(row[c] for row in a['population']);kept[c]+=sum(row[c] for row in a['selected'])
            aggregates.append(dict(seed=seed,domain=domain,steps=H,admitted_by_class=total,selected_by_class=kept,valid_pixels=valid,
                                   retained_fraction=sum(kept)/sum(total),exact_random_class_count_match=True))
    e.write(ROOT/'TEACHER_AUDIT.json',dict(status='PASS',rows=teachers))
    e.write(ROOT/'SELECTION_AUDIT.json',dict(status='PASS',matched_step_pairs=12000,records=24000,rows=aggregates,
        limitation='Same teacher-predicted class counts per image and step; not a gradient-norm or U-truth match.'))


def main():
    check();torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists() and not (ROOT/'PHYSICAL_LEDGER.jsonl').exists(), 'explicit recovery binding required'
    ledger=dose.Ledger(ROOT);ledger.caps=dict(CAPS)
    cost=dict(sequence_start='V3D-U-DOSE',previous_optimizer_calls=237200,previous_diagnostic_VJPs=800,reused_source_history=40000)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3L-AGREEMENT-SELECTION',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS,domains=DOMAINS,weights=WEIGHTS,method_options=OPTIONS,expected_active_U=ACTIVE,H=H,caps=CAPS,smoke_calls=100,
        source_updates_reused=40000,new_source_updates=0,session=ledger.session,
        evidence='development-only selection intervention; reused optimization seeds and validation patients',
        primary='AGREE_05-RANDOM_05 joint positive means, >=4/5 joint seeds, no domain mean regression; practical candidate also versus FIXED_05 and FINE_05',
        data_role_audit='V3E FINAL_INTERPRETATION; val development only, test remains closed'))
    try:
        dose.status('INTEGRATION_SMOKE');smoke(ledger,cost)
        dose.status('MAIN_MATRIX',H=H);endpoints=dose.matrix(ledger,entry_factory=entry)
        audit();dose.status('EVALUATION');rows,summary=dose.report(endpoints,ledger,cost)
        with (Path(C['predecessor'])/'ENDPOINTS.csv').open() as f:old={(int(r['seed']),r['domain'],r['method']):r for r in csv.DictReader(f)}
        controls=[dict(seed=r['seed'],domain=r['domain'],method=r['method'],**{k:r[k]-float(old[r['seed'],r['domain'],r['method']][k]) for k in ('macro_Dice','old_REFUGE')}) for r in rows if r['method'] in ('ORIGINAL','FINE_05','FIXED_05')]
        e.write(ROOT/'CONTROL_REPLICATION.json',dict(rows=controls,max_abs=max(abs(r[k]) for r in controls for k in ('macro_Dice','old_REFUGE'))))
        assert len(controls)==30 and all(all(r[k]==0 for k in ('macro_Dice','old_REFUGE')) for r in controls), 'control differs from V3I'
        result=decision(rows);e.write(ROOT/'SELECTION_DECISION.json',result)
        summary['selection_intervention']=result;e.write(ROOT/'SUMMARY.json',summary)
        cost['sequence_new_optimizer_calls']=237200+sum(ledger.count.values());e.write(ROOT/'ALL_COSTS.json',cost)
        with (ROOT/'FINAL_INTERPRETATION.md').open('a') as f:f.write('\n## Frozen selection comparison\n\n'+e.json.dumps(result,indent=2)+'\n')
        dose.status('COMPLETE_PRIVATE',endpoints=len(endpoints),H=H,selection_supported=result['selection_supported'],practical_candidate=result['practical_candidate'])
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(cost,physical_calls=dict(ledger.count),wall_seconds=time.time()-ledger.session['start']))
        dose.status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations'):main()
