"""Finite read-only calibration on complete current-domain training-L sets."""
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
SIZES={'RIM_ONE_r3':16,'Drishti_GS':10}


def status(name,**extra):e.write(ROOT/'status.json',dict(status=name,time=time.time(),pid=os.getpid(),**extra))


def summarize(rows):
    lookup={(r['seed'],r['domain'],r['state'],r['model']):r for r in rows}
    keys=('admitted_class_balanced_error','admitted_error','ece10','balanced_accuracy','balanced_nll','admission','confidence')
    result={}
    for model in ('teacher','student'):
        result[model]={}
        for k in keys:
            values=[]
            for seed in SEEDS:
                pair=[lookup[seed,d,'SLOW_05',model][k]-lookup[seed,d,'FINE_05',model][k] for d in DOMAINS if lookup[seed,d,'SLOW_05',model][k] is not None and lookup[seed,d,'FINE_05',model][k] is not None]
                values.append(dict(seed=seed,difference=statistics.mean(pair) if len(pair)==2 else None))
            finite=[v['difference'] for v in values if v['difference'] is not None]
            result[model][k]=dict(seed_differences=values,mean=statistics.mean(finite) if finite else None,sample_sd=statistics.stdev(finite) if len(finite)>1 else None)
    entry={k:[dict(seed=s,difference=lookup[s,'Drishti_GS','ENTRY','teacher'][k]-lookup[s,'RIM_ONE_r3','ENTRY','teacher'][k]) for s in SEEDS if lookup[s,'Drishti_GS','ENTRY','teacher'][k] is not None and lookup[s,'RIM_ONE_r3','ENTRY','teacher'][k] is not None] for k in keys}
    return dict(primary='Training-L teacher accepted class-balanced error: SLOW-FINE; ENTRY Drishti-RIM descriptive domain contrast',slow_minus_fine=result,entry_drishti_minus_rim=entry,
        interpretation='Training proxy only, no true U error rates, no independent validation or causal inference; no threshold selection.')


def main(ops):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'no automatic replay'
    ledger=e.Ledger(ROOT);ledger.caps={k:0 for k in ('source','main','smoke','development','controller','replay')}
    (ROOT/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3J-TEACHER-CALIBRATION',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS,domains=DOMAINS,states=STATES,training_L_sizes=SIZES,expected_batches=325,expected_sample_reads=650,
        expected_inference_images=1300,optimizer_cap=0,session=ledger.session,admission_threshold=.7,bins=10))
    started=time.time();rows=[];reliability=[];immutable=[];batches=0
    original=e.primitive.CurrentData.__getitem__
    def labeled_only(ds,i):
        if ds.role!='train_labeled':raise PermissionError('calibration may read training L only')
        return original(ds,i)
    try:
        status('CALIBRATION',batches=0,states=0)
        with patch.object(e.primitive.CurrentData,'__getitem__',labeled_only):
            for seed in SEEDS:
                for domain in DOMAINS:
                    t=e.create(C,seed,domain,False,ledger);ds=t.provider._l
                    assert len(ds)==SIZES[domain]
                    teacher_entry=None
                    for state in STATES:
                        path=Path(C['predecessor'])/f'S{seed}_{domain}'/('entry_state.pt' if state=='ENTRY' else state+'_latest.pt')
                        payload=torch.load(path,map_location='cpu',weights_only=False);native_step=0 if state=='ENTRY' else 1200
                        assert payload['commit']==PREDECESSOR_COMMIT and payload['state']['step']==payload['state']['cursor']==native_step
                        if state!='ENTRY':assert payload['method']==state
                        e.restore(t,payload['state']);del payload
                        before=e.snapshot(t);counts={m:Counts() for m in ('teacher','student')}
                        with t.readonly(),torch.no_grad():
                            for i in range(0,len(ds),2):
                                assert time.time()<ledger.session['hard_deadline'] and batches<325
                                key=f'{seed}/{domain}/{state}/{i//2}'
                                e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='attempt',key=key,time=time.time()))
                                items=[ds[j] for j in range(i,min(i+2,len(ds)))];x=torch.stack([v['image'] for v in items]).cuda();y=torch.stack([v['label'] for v in items]).cuda()
                                counts['teacher'].add(t.ema(x,mode='teacher').softmax(1),y);counts['student'].add(t.clean(x),y)
                                batches+=1;e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='success',key=key,time=time.time()))
                        assert e.same(before,e.snapshot(t)), 'calibration mutated full trainer state'
                        immutable.append(dict(seed=seed,domain=domain,state=state,immutable=True))
                        for model,count in counts.items():
                            metrics,raw=count.finish();identity=dict(seed=seed,domain=domain,state=state,model=model,images=len(ds))
                            rows.append(dict(identity,**metrics));reliability.append(dict(identity,**raw))
                            if model=='teacher':
                                if state=='ENTRY':teacher_entry=(metrics,raw)
                                if state=='FIXED_05':assert e.same(teacher_entry,(metrics,raw)), 'fixed teacher predictions differ from entry'
                        ops.flush('state_complete');status('CALIBRATION',batches=batches,states=len(immutable),seed=seed,domain=domain,state=state)
                    del t,before,counts;gc.collect();torch.cuda.empty_cache()
        counts=dict(ops.counts)
        assert batches==325 and len(immutable)==50 and len(rows)==100 and counts['sample_train_labeled']==650
        assert counts['features']==counts['native_readout']==650
        assert all(counts.get(k,0)==0 for k in ('optimizer_steps','autograd_grad','backward','sample_train_unlabeled','sample_val','sample_test','update_dense_ema','ema_updates'))
        events=[e.json.loads(v) for v in (ROOT/'DIAGNOSTIC_LEDGER.jsonl').read_text().splitlines()]
        attempts=[v['key'] for v in events if v['event']=='attempt'];success=[v['key'] for v in events if v['event']=='success']
        assert attempts==success and len(attempts)==len(set(attempts))==325
        with (ROOT/'CALIBRATION.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        e.write(ROOT/'RELIABILITY.json',reliability);e.write(ROOT/'SUMMARY.json',summarize(rows))
        e.write(ROOT/'COMPLETION_AUDIT.json',dict(status='PASS',batches=325,labeled_image_reads=650,inference_images=1300,optimizer_calls=0,VJPs=0,U_reads=0,val_reads=0,test_reads=0,immutable_states=immutable,fixed_entry_teacher_predictions_equal=10))
        e.write(ROOT/'ALL_COSTS.json',dict(operations=counts,wall_seconds=time.time()-started,preparation_inclusive_seconds=time.time()-ledger.session['start'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_optimizer_calls=0,sequence_new_optimizer_calls=237200,previous_diagnostic_VJPs=800,reused_source_history=40000))
        status('COMPLETE_PRIVATE',batches=325,states=50,optimizer_calls=0)
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(operations=dict(ops.counts),wall_seconds=time.time()-started,successful_batches=batches,new_optimizer_calls=sum(ledger.count.values())))
        status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations',update_cap=0) as operations:main(operations)
