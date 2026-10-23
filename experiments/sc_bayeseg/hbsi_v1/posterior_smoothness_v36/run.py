"""Two matched new-mechanism arms; no retraining of closed original configurations."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from .method import Objective,auxiliary,check_fixture,predictive
from ..appearance_transport_v24.transport import DonorData


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='POSTERIOR_SMOOTHNESS_V36'
    assert p['U_path']=='posterior_common_noise_VAT' and p['xi']==1e-3 and p['epsilon']==1.0 and p['auxiliary_weight']==1
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=12000,max_updates=800,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v23_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==1600 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    try:
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=600,updates=0) as out:
            s=load();d=data(p,s.config,'RUNMC','LU');assert s.active['evidence']==d.evidence
            before=digest(s.payload());rng=s.generator.get_state();L=d.batch('L',1,torch.Generator().manual_seed(163800));U=DonorData(d,True).batch('U',1,torch.Generator().manual_seed(163801));LD=DonorData(d,False).batch('U',1,torch.Generator().manual_seed(163801))
            for b in (L,U,LD):
                b['x']=b['x'].cuda()
                if 'observations'in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            paths=check_fixture()
            with torch.no_grad():
                p0=predictive(s,U['x']);p1=predictive(s,U['x'])
                assert torch.equal(p0,p1)
            aux,_=auxiliary(s,U['x'])
            assert all(v.grad is None for group in s.optimizer.param_groups for v in group['params'])
            aux.backward()
            assert s.phi.enc.weight.grad is not None and s.phi.enc.weight.grad.norm()>0
            assert s.phi.up.weight.grad is not None and s.phi.up.weight.grad.norm()>0
            assert s.q.mean.grad is not None and s.q.mean.grad.norm()>0
            assert s.phi.c_head.weight.grad is not None and s.phi.c_head.weight.grad.norm()>0
            assert s.phi.r_head.weight.grad is not None and s.phi.r_head.weight.grad.norm()>0
            s.optimizer.zero_grad(set_to_none=True)
            parts=[]
            for use_u in (False,True):
                s.generator.set_state(rng);start=time.monotonic();loss,record=Objective(s,Refiner(s.model,s.phi,counter),use_u).estimate(L,U if use_u else LD,d.counts,s.generator);loss.backward();torch.cuda.synchronize()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for group in s.optimizer.param_groups for v in group['params'])
                assert s.q.mean.grad is not None and s.phi.c_head.weight.grad is not None
                assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
                parts.append(dict(use_u=use_u,seconds=time.monotonic()-start,parts={k:float(v) for k,v in record.items()}));s.optimizer.zero_grad(set_to_none=True)
                if not use_u:end_rng=s.generator.get_state()
                else:assert torch.equal(s.generator.get_state(),end_rng)
            assert parts[0]['parts']['task']==parts[1]['parts']['task'] and parts[0]['parts']['anchor']==parts[1]['parts']['anchor']
            s.generator.set_state(rng);assert digest(s.payload())==before
            write(root/'PREFLIGHT.json',dict(status="PASS",smoothness_checks=paths,common_noise_replay=True,direction_grad_does_not_fill_param_grad=True,auxiliary_semantic_gradient=True,auxiliary_path_reverse_calls=1,real_batch=parts,paired_original_L_task_anchor=True,paired_global_rng=True,source_updates=0,real_batch_reverse_calls=2,fixture_reverse_calls=2,U_label_capability=False,counts=d.counts))
            if 400*sum(v['seconds'] for v in parts)+3600>12000:raise RuntimeError('cost exceeds frozen budget')
            del s,d,L,U,LD
        cells={};common=None
        for arm,use_u in [('L',False),('LU',True)]:
            s=load();d=DonorData(data(p,s.config,'RUNMC','LU'),use_u);assert s.active['evidence']==d.evidence;initial=digest(s.payload())
            if common is None:common=initial
            else:assert initial==common
            old_prior=digest(s.prior.state_dict());unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            write(root/f'BRANCH_{arm}.json',dict(common_payload=initial,initial_steps=1600,original_Adam_inherited=True,prior_hash=old_prior,unused_heads_frozen=True,counts=d.counts,auxiliary_image_pool='U' if use_u else 'L',auxiliary_labels_read=False))
            status('TRAIN',arm=arm)
            holder={}
            def factory(st,ref):
                holder['obj']=Objective(st,ref,use_u,root/f'PRIVATE_AUDIT_{arm}.jsonl');return holder['obj']
            run_stage(s,d,ledger,counter,arm,2000,3600,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=factory,after_checkpoint=lambda step:holder['obj'].after_update(step))
            audit=[json.loads(v) for v in (root/f'PRIVATE_AUDIT_{arm}.jsonl').read_text().splitlines()]
            assert [v['step'] for v in audit]==[1600,1799,1999]
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=900,updates=0) as out:cells[arm+'_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=800,preflight_real_reverse_calls=2,preflight_fixture_reverse_calls=2,auxiliary_path_reverse_calls=1,source_updates=0,common_entrance=True,training_audit_reverse_calls=12,formal_direction_reverse_calls=800,preflight_direction_reverse_calls=3,preflight_additional_replay_semantic_forwards=8,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
