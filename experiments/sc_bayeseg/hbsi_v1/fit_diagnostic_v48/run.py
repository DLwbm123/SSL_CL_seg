"""Bounded train-only fit diagnostic from the sealed W target entrance."""
import json,os,time
from pathlib import Path
from statistics import mean
import torch
from ..data_adapter import CurrentTrainingData
from ..stream_state import digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import data
from ..paired_semantic_v19.method import balanced_risk
from ..semantic_alignment_v23.method import Objective as Original
from ..semantic_evidence_v22.method import sample_semantics
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from .method import Objective


class Subset(CurrentTrainingData):
    def __init__(self,base,records):
        self.__dict__.update(base.__dict__);self.original_counts=base.counts
        self.records={'L':records,'U':[]}
    @property
    def counts(self):return self.original_counts


def choose(base):
    groups={}
    for r in base.records['L']:groups.setdefault(str(r['image']),[]).append(r)
    assert len(groups)==4 and base.config.classes==2
    chosen=[];flags=[]
    for key,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['index']);nonempty=[bool((base.read(r,'label')==1).any()) for r in rows]
        a=[r for r,f in zip(rows,nonempty) if not f];b=[r for r,f in zip(rows,nonempty) if f]
        selection=[a[len(a)//2],b[len(b)//2]] if a and b else [rows[len(rows)//3],rows[2*len(rows)//3]]
        assert selection[0]['index']!=selection[1]['index']
        chosen.extend(selection);flags.extend([bool((base.read(r,'label')==1).any()) for r in selection])
    return chosen,flags


def batch(base,records):
    b=Subset(base,records).batch('L',1,torch.Generator().manual_seed(0))
    b['x']=b['x'].cuda();b['observations']={k:v.cuda() for k,v in b['observations'].items()};return b


def fit_readout(s,base,records,flags):
    before=s.generator.get_state();rows=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            b=batch(base,[r]);truth=b['observations']['y_obs'].to(b['x'].dtype);y=truth[:,1].bool();repeats=[]
            for seed in (48000000,48000001,48000002):
                rng=torch.Generator().manual_seed(seed);probs=[];risks=[]
                for _ in range(8):
                    theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
                    for _ in range(4):
                        prob,_=sample_semantics(s,b['x'],theta,a,rng);probs.append(prob);risks.append(float(balanced_risk(prob,truth).mean()))
                prob=torch.stack(probs).mean(0);pred=prob.argmax(1)==1
                tp=int((pred&y).sum());pc=int(pred.sum());gt=int(y.sum());pixels=y.numel()
                repeats.append(dict(Dice=2*tp/(pc+gt) if pc+gt else 1.,precision=tp/pc if pc else None,recall=tp/gt if gt else None,volume_ratio=pc/gt if gt else None,FP_fraction=int((pred&~y).sum())/int((~y).sum()),mean_probability_risk=float(balanced_risk(prob,truth).mean()),expected_risk=mean(risks),foreground_pixels=gt))
            rows.append(dict(case=i//2+1,slice_slot=i%2+1,native_nonempty=flags[i],repeats=repeats))
    assert torch.equal(before,s.generator.get_state())
    nonempty=[mean(v['Dice'] for v in r['repeats']) for r in rows if r['native_nonempty']]
    assert nonempty and all(v['foreground_pixels']>0 for r in rows if r['native_nonempty'] for v in r['repeats'])
    empty=[mean(v['FP_fraction'] for v in r['repeats']) for r in rows if not r['native_nonempty']]
    patient=[mean(mean(v['Dice'] for v in r['repeats']) for r in rows if r['case']==i and r['native_nonempty']) for i in range(1,5) if any(r['case']==i and r['native_nonempty'] for r in rows)]
    fg=mean(nonempty);fp=mean(empty) if empty else None
    return dict(step=s.steps,rows=rows,foreground_Dice=fg,patient_foreground_Dice=patient,empty_FP_fraction=fp,fit_gate=fg>=.9 and len(patient)==4 and min(patient)>=.85 and (fp is None or fp<=.01),semantic_sample_forwards=768,training_rng_unchanged=True)


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='FIT_DIAGNOSTIC_V48'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=3000,max_updates=400,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    def load():
        s=MultiscaleState.load(Path(p['v46_root'])/'results/W_target_attempt001/ACTIVE_INITIAL.pt')
        assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False);return s
    try:
        s=load();base=data(p,s.config,'BMC','L');assert s.active['evidence']==base.evidence and base.counts['U']==0
        records,flags=choose(base);subset=Subset(base,records)
        write(root/'PRIVATE_SELECTION.json',[dict(image=str(r['image']),index=r['index'],nonempty=f) for r,f in zip(records,flags)])
        write(root/'SELECTION.json',dict(patients=4,slices=8,nonempty_slices=sum(flags),empty_slices=8-sum(flags),original_counts=base.counts,selection_before_model_predictions=True))
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=180,updates=0):
            initial=digest(s.payload());rng=s.generator.get_state();b=batch(base,[records[0]])
            old,parts=Original(s,Refiner(s.model,s.phi,counter),False).estimate(b,None,base.counts,s.generator);end=s.generator.get_state()
            params=[q for g in s.optimizer.param_groups for q in g['params'] if q.requires_grad]
            expected=torch.autograd.grad(old,params,allow_unused=True)
            s.generator.set_state(rng);obj=Objective(s,Refiner(s.model,s.phi,counter),'Full',[])
            _,v=obj.components(b,base.counts,s.generator,preflight=True);new=v['full_objective_same_samples']
            torch.testing.assert_close(new,old,atol=1e-10,rtol=1e-9);assert torch.equal(end,s.generator.get_state())
            actual=torch.autograd.grad(new,params,allow_unused=True);maxerr=0.
            for a,e in zip(actual,expected):
                assert (a is None)==(e is None)
                if a is not None:
                    assert torch.isfinite(a).all();torch.testing.assert_close(a,e,atol=1e-10,rtol=1e-8);maxerr=max(maxerr,float((a-e).abs().max()))
            s.generator.set_state(rng);assert digest(s.payload())==initial
            write(root/'PREFLIGHT.json',dict(status='PASS',same_full_loss_and_gradients=True,max_gradient_error=maxerr,source_unchanged=True,objective_forwards=2,reverse_calls=2,Adam_updates=0))
            del obj,params,expected,actual,old,new,v,b,s
        results={};signatures=[]
        for arm in ('Full','Task'):
            s=load();signatures.append(digest(s.payload()));assert all(v==signatures[0] for v in signatures)
            memory=digest(s.memories['RUNMC'].state_dict());unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            status('READOUT',arm=arm,step=0)
            reads=[];audits=[];holder=[]
            with ledger.attempt('read_'+arm+'_0',seconds=300,updates=0):reads.append(fit_readout(s,base,records,flags))
            def factory(state,ref):
                obj=Objective(state,ref,arm,audits);holder.append(obj);return obj
            def after(step):
                holder[0].after_step(step)
                if step in (100,200):
                    status('READOUT',arm=arm,step=step)
                    reads.append(fit_readout(s,base,records,flags));write(root/f'READOUT_{arm}_{step}.json',reads[-1]);status('TRAIN',arm=arm)
            status('TRAIN',arm=arm)
            run_stage(s,subset,ledger,counter,arm,200,900,fingerprint(p),data_seed=263000,checkpoint_stride=100,objective_factory=factory,after_checkpoint=after)
            assert len(audits)==3 and digest(s.memories['RUNMC'].state_dict())==memory
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            results[arm]=dict(readouts=reads,gradient_audits=audits);write(root/f'ARM_{arm}.json',results[arm]);del s
        f=results['Full']['readouts'][-1];t=results['Task']['readouts'][-1];gap=t['foreground_Dice']-f['foreground_Dice']
        decision='task_only_fit_advantage' if t['fit_gate'] and not f['fit_gate'] and gap>=.10 else 'both_fit' if t['fit_gate'] and f['fit_gate'] else 'neither_fit' if not t['fit_gate'] and not f['fit_gate'] else 'inconclusive'
        write(root/'RESULTS.json',dict(arms=results,decision=decision,task_minus_full_Dice=gap,formal_updates=400,preflight_reverse_calls=2,audit_reverse_calls=12,readout_sample_forwards=4608,source_unchanged=True,common_entrance=True,old_memory_and_Sb_unchanged=True,validation_cells=0,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
