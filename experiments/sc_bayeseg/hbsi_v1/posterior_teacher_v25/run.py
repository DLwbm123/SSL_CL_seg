"""Cache once, preflight once, then three fixed-endpoint matched branches."""
import copy,json,os,time
from pathlib import Path
import torch
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from .teacher import build_cache,CachedData
from .method import Objective,teacher_risk


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='FIXED_POSTERIOR_TEACHER_V25' and p['U_coefficient']==1 and p['teacher']=='v23_L1600'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=18000,max_updates=1200,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v23_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==1600 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    try:
        status('TEACHER_CACHE')
        with ledger.attempt('teacher_cache',seconds=1200,updates=0) as out:
            s=load();d=data(p,s.config,'RUNMC','LU');assert s.active['evidence']==d.evidence
            cache,summary=build_cache(s,d,root/'PRIVATE_TEACHER.pt');write(root/'TEACHER_SUMMARY.json',summary);del s,d
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=600,updates=0) as out:
            s=load();d=CachedData(data(p,s.config,'RUNMC','LU'),cache)
            assert digest(s.payload())==cache['teacher_payload'];before=digest(s.payload());rng=s.generator.get_state()
            L=d.batch('L',1,torch.Generator().manual_seed(166200));U=d.batch('U',1,torch.Generator().manual_seed(166201))
            for b in (L,U):
                b['x']=b['x'].cuda()
                if 'observations' in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            t=U['teacher_mean'];w=U['teacher_weight'];student=t*.8+.1
            r,_=teacher_risk(t,student,torch.ones_like(w));scaled,_=teacher_risk(t,student,torch.ones_like(w)*.3)
            assert torch.allclose(r,scaled,atol=1e-12)
            z,_=teacher_risk(t,t,w);assert abs(float(z))<1e-12
            bg=torch.zeros_like(t);bg[:,0]=1;assert torch.isfinite(teacher_risk(bg,student,w)[0])
            parts=[]
            for arm in ('A','B','C'):
                s.generator.set_state(rng);loss,record=Objective(s,Refiner(s.model,s.phi,counter),arm).estimate(L,U,d.counts,s.generator);loss.backward()
                assert all(q.grad is None or torch.isfinite(q.grad).all() for group in s.optimizer.param_groups for q in group['params'])
                assert s.q.mean.grad is not None and s.phi.c_head.weight.grad is not None
                assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
                parts.append(dict(arm=arm,parts={k:float(v) for k,v in record.items()}));s.optimizer.zero_grad(set_to_none=True)
                if arm=='A':end=s.generator.get_state()
                else:assert torch.equal(end,s.generator.get_state())
            assert len({v['parts']['task'] for v in parts})==1 and len({v['parts']['anchor'] for v in parts})==1
            s.generator.set_state(rng);assert digest(s.payload())==before
            write(root/'PREFLIGHT.json',dict(status='PASS',real_batch=parts,source_unchanged=True,paired_L_task_anchor_rng=True,weight_scale_invariance=True,zero_self_KL=True,all_background_finite=True,reverse_calls=3,optimizer_updates=0));del s,d,L,U
        common=None;cells={}
        for arm in ('A','B','C'):
            s=load();d=CachedData(data(p,s.config,'RUNMC','LU'),cache);initial=digest(s.payload())
            if common is None:common=initial
            else:assert initial==common
            assert initial==cache['teacher_payload']
            unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            write(root/f'BRANCH_{arm}.json',dict(common_payload=initial,prior_hash=digest(s.prior.state_dict()),initial_steps=1600,Adam_inherited=True,teacher_payload=cache['teacher_payload']))
            status('TRAIN',arm=arm);holder={}
            def factory(st,ref):
                holder['obj']=Objective(st,ref,arm,root/f'PRIVATE_AUDIT_{arm}.jsonl');return holder['obj']
            run_stage(s,d,ledger,counter,arm,2000,3600,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=factory,after_checkpoint=lambda step:holder['obj'].after_update(step))
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=900,updates=0) as out:cells[arm+'_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=1200,preflight_reverse_calls=3,training_audit_reverse_calls=15,source_updates=0,teacher_updates=0,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
