"""Finite zero-update L/U agreement diagnostic using existing full states."""
import csv
import fcntl
import gc
import os
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e
from experiments.qprompt_rl_v1.v3i_teacher_timescale.protocol import SEEDS,DOMAINS
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .metrics import Counts

ROOT=Path(os.environ['EXEC_RUN']);C=e.read(os.environ['EXEC_CONFIG'])
STATES=('ENTRY','ORIGINAL','FINE_05','FIXED_05','SLOW_05')
PREDECESSOR_COMMIT='2acdec5fd217bf31585f343befa525198be8f71a'
SIZES={'RIM_ONE_r3':(16,63),'Drishti_GS':(10,41)}


def status(name,**extra):e.write(ROOT/'status.json',dict(status=name,time=time.time(),pid=os.getpid(),**extra))


def summarize(cells):
    result=[]
    keys=('agreement_fraction_admitted','teacher_error_delta','teacher_common_balanced_delta','rejected_error_enrichment')
    for state in STATES:
        for role in ('L','U'):
            for domain in DOMAINS:
                selected=[x for x in cells if (x['state'],x['role'],x['domain'])==(state,role,domain)]
                metrics={}
                for key in keys:
                    values=[x['delta'][key] for x in selected if x['delta'][key] is not None]
                    metrics[key]=dict(n=len(values),mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None,
                                      seed_values=[dict(seed=x['seed'],value=x['delta'][key]) for x in selected])
                result.append(dict(state=state,role=role,domain=domain,metrics=metrics))
    return dict(rows=result,primary='FINE_05 and SLOW_05 L: agreement-minus-admitted teacher error and common-class balanced error, with U retained coverage; all states retained',
                interpretation='L training proxy, U truth unavailable; no causal, independent-generalization or performance-success claim, no threshold selection')


def main(ops):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'no automatic replay'
    ledger=e.Ledger(ROOT);ledger.caps={k:0 for k in ('source','main','smoke','development','controller','replay')}
    (ROOT/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3K-AGREEMENT-DIAGNOSTIC',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,seeds=SEEDS,domains=DOMAINS,states=STATES,
        training_L_U_sizes=SIZES,expected_batches=1650,expected_L_reads=650,expected_U_reads=2600,expected_inference_images=6500,optimizer_cap=0,session=ledger.session,admission_threshold=.7))
    previous=e.read(Path(C['calibration'])/'RELIABILITY.json');previous={(r['seed'],r['domain'],r['state'],r['model']):r for r in previous}
    started=time.time();rows=[];cells=[];immutable=[];batches=0;reproduced=0
    original=e.primitive.CurrentData.__getitem__
    def train_only(ds,i):
        if ds.role not in ('train_labeled','train_unlabeled'):raise PermissionError('only current training L/U allowed')
        item=original(ds,i)
        if ds.role=='train_unlabeled':assert set(item)=={'image','geometry'}, 'U truth is forbidden'
        return item
    try:
        status('AGREEMENT',batches=0,states=0)
        with patch.object(e.primitive.CurrentData,'__getitem__',train_only):
            for seed in SEEDS:
                for domain in DOMAINS:
                    t=e.create(C,seed,domain,False,ledger)
                    assert (len(t.provider._l),len(t.provider._u))==SIZES[domain]
                    for state in STATES:
                        path=Path(C['predecessor'])/f'S{seed}_{domain}'/('entry_state.pt' if state=='ENTRY' else state+'_latest.pt')
                        payload=torch.load(path,map_location='cpu',weights_only=False);native_step=0 if state=='ENTRY' else 1200
                        assert payload['commit']==PREDECESSOR_COMMIT and payload['state']['step']==payload['state']['cursor']==native_step
                        if state!='ENTRY':assert payload['method']==state
                        e.restore(t,payload['state']);del payload
                        before=e.snapshot(t)
                        with t.readonly(),torch.no_grad():
                            for role,ds in (('L',t.provider._l),('U',t.provider._u)):
                                count=Counts(role=='L')
                                for i in range(0,len(ds),2):
                                    assert time.time()<ledger.session['hard_deadline'] and batches<1650
                                    key=f'{seed}/{domain}/{state}/{role}/{i//2}'
                                    e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='attempt',key=key,time=time.time()))
                                    items=[ds[j] for j in range(i,min(i+2,len(ds)))];x=torch.stack([v['image'] for v in items]).cuda()
                                    valid=torch.stack([v['geometry'] for v in items]).cuda();y=torch.stack([v['label'] for v in items]).cuda() if role=='L' else None
                                    count.add(t.ema(x,mode='teacher').softmax(1),t.clean(x),valid,y)
                                    batches+=1;e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='success',key=key,time=time.time()))
                                metrics,delta,raw=count.finish();identity=dict(seed=seed,domain=domain,state=state,role=role,images=len(ds))
                                rows.extend(dict(identity,**r) for r in metrics);cells.append(dict(identity,delta=delta,**raw))
                                if role=='L':
                                    for model in ('teacher','student'):
                                        assert raw['truth']['ALL'][model]==previous[seed,domain,state,model]['confusion'], 'L predictions differ from V3J'
                                    assert raw['truth']['ADMITTED']['teacher']==previous[seed,domain,state,'teacher']['admitted_confusion']
                                    reproduced+=1
                        assert e.same(before,e.snapshot(t)), 'diagnostic mutated full trainer state'
                        immutable.append(dict(seed=seed,domain=domain,state=state,immutable=True))
                        ops.flush('state_complete');status('AGREEMENT',batches=batches,states=len(immutable),seed=seed,domain=domain,state=state)
                    del t,before,count;gc.collect();torch.cuda.empty_cache()
        counts=dict(ops.counts)
        assert batches==1650 and len(immutable)==50 and len(rows)==400 and len(cells)==100 and reproduced==50
        assert counts['sample_train_labeled']==650 and counts['sample_train_unlabeled']==2600
        assert counts['features']==counts['native_readout']==3300 and counts['hdf5_open']==3900
        assert all(counts.get(k,0)==0 for k in ('optimizer_steps','autograd_grad','backward','sample_val','sample_test','update_dense_ema','ema_updates'))
        events=[e.json.loads(v) for v in (ROOT/'DIAGNOSTIC_LEDGER.jsonl').read_text().splitlines()]
        attempts=[v['key'] for v in events if v['event']=='attempt'];success=[v['key'] for v in events if v['event']=='success']
        assert attempts==success and len(attempts)==len(set(attempts))==1650
        with (ROOT/'AGREEMENT.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        e.write(ROOT/'STRATA.json',cells);e.write(ROOT/'SUMMARY.json',summarize(cells))
        e.write(ROOT/'COMPLETION_AUDIT.json',dict(status='PASS',batches=1650,L_reads=650,U_reads=2600,inference_images=6500,optimizer_calls=0,VJPs=0,U_labels=0,val_reads=0,test_reads=0,immutable_states=immutable,V3J_prediction_replications=reproduced))
        e.write(ROOT/'ALL_COSTS.json',dict(operations=counts,wall_seconds=time.time()-started,preparation_inclusive_seconds=time.time()-ledger.session['start'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_optimizer_calls=0,sequence_new_optimizer_calls=237200,previous_diagnostic_VJPs=800,reused_source_history=40000))
        status('COMPLETE_PRIVATE',batches=1650,states=50,optimizer_calls=0)
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(operations=dict(ops.counts),wall_seconds=time.time()-started,successful_batches=batches,new_optimizer_calls=sum(ledger.count.values())))
        status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations',update_cap=0) as operations:main(operations)
