"""Finite read-only diagnostic queue over existing full trainer states."""
import csv
import fcntl
import gc
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e
from experiments.qprompt_rl_v1.v3b_lrref_endpoint.maths import u_loss
from experiments.qprompt_rl_v1.v3f_u_timing.protocol import SEEDS, DOMAINS
from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
from .metrics import gradient_metrics, kl

ROOT=Path(os.environ['EXEC_RUN'])
C=e.read(os.environ['EXEC_CONFIG'])
STATES=('ENTRY','ORIGINAL','FINE_05','EARLY_05','LATE_05')
CURSORS=(0,5,10,15,20,25,30,35)
PREDECESSOR_COMMIT='baafffbac4c56495af85164c0d4265a6473e9996'


def status(name,**extra):
    e.write(ROOT/'status.json',dict(status=name,time=time.time(),pid=os.getpid(),**extra))


def probe(t,entry_q):
    l,p,q,valid=t.components();u,_=u_loss(p,q,valid,2)
    params=[x for x in t.model.parameters() if x.requires_grad]
    gl=torch.autograd.grad(l,params,retain_graph=True,allow_unused=True)
    gu=torch.autograd.grad(u,params,allow_unused=True)
    result=gradient_metrics(gl,gu)
    with torch.no_grad():
        v=valid.bool();assert v.any()
        result.update(loss_L=float(l),loss_U=float(u),confidence=float(q.max(1).values[v].mean()),
            admission=float((q.max(1).values[v]>.7).float().mean()),
            teacher_student_KL=float(kl(q,p)[v].mean()),
            disagreement=float((q.argmax(1)[v]!=p.argmax(1)[v]).float().mean()),
            entry_to_teacher_KL=0. if entry_q is None else float(kl(entry_q.to(q),q)[v].mean()))
        for c in range(3):result['teacher_class_'+str(c)]=float((q.argmax(1)[v]==c).float().mean())
        x,y,_=t.provider.labeled(t.cursor);ql=t.ema(x,mode='teacher').softmax(1)
        vl=y!=255;pred=ql.argmax(1);assert vl.any()
        present=torch.unique(y[vl])
        result.update(teacher_L_accuracy=float((pred[vl]==y[vl]).float().mean()),
            teacher_L_balanced_accuracy=float(torch.stack([(pred[y==c]==c).float().mean() for c in present]).mean()),
            L_present_classes=len(present),valid_U_pixels=int(v.sum()),valid_L_pixels=int(vl.sum()))
    assert all(v is None or math.isfinite(v) for v in result.values())
    return result,q.detach().cpu() if entry_q is None else None


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows):
    keys=[k for k in rows[0] if k not in ('seed','domain','state','native_step','probe_cursor')]
    cells=[]
    for s in SEEDS:
        for d in DOMAINS:
            for state in STATES:
                group=[r for r in rows if (r['seed'],r['domain'],r['state'])==(s,d,state)]
                assert len(group)==8
                cell=dict(seed=s,domain=d,state=state,probes=len(group))
                for k in keys:
                    values=[r[k] for r in group if r[k] is not None]
                    cell[k]=statistics.mean(values) if values else None
                cell['undefined_cosines']=sum(r['gradient_cosine'] is None for r in group);cells.append(cell)
    indexed={(r['seed'],r['domain'],r['state']):r for r in cells}
    paired={}
    for k in keys:
        diffs=[];domains={d:[] for d in DOMAINS}
        for s in SEEDS:
            pair=[]
            for d in DOMAINS:
                a=indexed[s,d,'LATE_05'][k];b=indexed[s,d,'EARLY_05'][k]
                if a is not None and b is not None:pair.append(a-b);domains[d].append(a-b)
            diffs.append(dict(seed=s,difference=statistics.mean(pair) if len(pair)==2 else None))
        values=[r['difference'] for r in diffs if r['difference'] is not None]
        paired[k]=dict(seed_differences=diffs,mean=statistics.mean(values) if values else None,
            sample_sd=statistics.stdev(values) if len(values)>1 else None,
            positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),
            domain_means={d:statistics.mean(v) if v else None for d,v in domains.items()})
    return cells,dict(primary='LATE_05-EARLY_05 final states; gradient_cosine and entry_to_teacher_KL',
        paired=paired,interpretation='Descriptive training probes only; no causal confirmation, no validation or test score, no full timecourse.')


def main(ops):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    handle=(ROOT/'EXECUTOR.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'RUN_LOCK.json').exists(), 'no automatic replay'
    ledger=e.Ledger(ROOT);ledger.caps={k:0 for k in ('source','main','smoke','development','controller','replay')}
    (ROOT/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    e.write(ROOT/'RUN_LOCK.json',dict(protocol='V3G-U-DIAGNOSTICS',commit=C['commit'],predecessor_commit=PREDECESSOR_COMMIT,
        seeds=SEEDS,domains=DOMAINS,states=STATES,probe_cursors=CURSORS,expected_probes=400,expected_VJPs=800,
        optimizer_cap=0,session=ledger.session,data_roles=['train_labeled','train_unlabeled']))
    start=time.time();rows=[];immutable=[]
    original=e.primitive.CurrentData.__init__
    def training_only(ds,data,stage,role,*args,**kwargs):
        if role not in ('train_labeled','train_unlabeled'):raise PermissionError('diagnostic training data only')
        return original(ds,data,stage,role,*args,**kwargs)
    try:
        status('DIAGNOSTICS',probes=0,states=0)
        with patch.object(e.primitive.CurrentData,'__init__',training_only):
            for seed in SEEDS:
                for domain in DOMAINS:
                    t=e.create(C,seed,domain,False,ledger);cached={}
                    for state in STATES:
                        path=Path(C['predecessor'])/f'S{seed}_{domain}'/('entry_state.pt' if state=='ENTRY' else state+'_latest.pt')
                        payload=torch.load(path,map_location='cpu',weights_only=False)
                        native_step=0 if state=='ENTRY' else 1200
                        assert payload['commit']==PREDECESSOR_COMMIT
                        assert payload['state']['step']==payload['state']['cursor']==native_step
                        if state!='ENTRY':assert payload['method']==state
                        e.restore(t,payload['state']);del payload
                        before=e.snapshot(t);cursor=t.cursor
                        try:
                            for probe_cursor in CURSORS:
                                assert time.time()<ledger.session['hard_deadline'] and len(rows)<400
                                key=f'{seed}/{domain}/{state}/{probe_cursor}'
                                e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='attempt',key=key,time=time.time()))
                                t.cursor=probe_cursor
                                with t.readonly():values,q=probe(t,cached.get(probe_cursor))
                                if state=='ENTRY':cached[probe_cursor]=q
                                row=dict(seed=seed,domain=domain,state=state,native_step=native_step,probe_cursor=probe_cursor,**values)
                                e.append(ROOT/'PROBES.jsonl',row);rows.append(row)
                                e.append(ROOT/'DIAGNOSTIC_LEDGER.jsonl',dict(event='success',key=key,time=time.time()))
                        finally:t.cursor=cursor
                        assert e.same(before,e.snapshot(t)), 'diagnostic mutated full training state'
                        immutable.append(dict(seed=seed,domain=domain,state=state,immutable=True))
                        ops.flush('state_complete');status('DIAGNOSTICS',probes=len(rows),states=len(immutable),seed=seed,domain=domain,state=state)
                    del t,before,cached;gc.collect();torch.cuda.empty_cache()
        counts=dict(ops.counts)
        assert len(rows)==400 and len(immutable)==50 and counts.get('autograd_grad')==800
        assert all(counts.get(k,0)==0 for k in ('optimizer_steps','sample_val','sample_test','update_dense_ema','ema_updates'))
        events=[e.json.loads(v) for v in (ROOT/'DIAGNOSTIC_LEDGER.jsonl').read_text().splitlines()]
        attempts=[v['key'] for v in events if v['event']=='attempt'];success=[v['key'] for v in events if v['event']=='success']
        assert len(attempts)==len(success)==len(set(attempts))==400 and attempts==success
        cells,summary=summarize(rows);write_csv(ROOT/'DIAGNOSTICS.csv',rows);write_csv(ROOT/'CELL_SUMMARY.csv',cells)
        e.write(ROOT/'SUMMARY.json',summary)
        e.write(ROOT/'COMPLETION_AUDIT.json',dict(status='PASS',probes=400,VJPs=800,optimizer_calls=0,validation_reads=0,test_reads=0,immutable_states=immutable))
        e.write(ROOT/'ALL_COSTS.json',dict(operations=counts,wall_seconds=time.time()-start,preparation_inclusive_seconds=time.time()-ledger.session['start'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_optimizer_calls=0,sequence_new_optimizer_calls=153060,reused_source_history=40000))
        status('COMPLETE_PRIVATE',probes=400,states=50,optimizer_calls=0)
    except BaseException as exc:
        e.write(ROOT/'ALL_COSTS.json',dict(operations=dict(ops.counts),wall_seconds=time.time()-start,successful_probes=len(rows),new_optimizer_calls=sum(ledger.count.values())))
        status('STOPPED',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    with NativeOperations(ROOT/'operations',update_cap=0) as operations:main(operations)
