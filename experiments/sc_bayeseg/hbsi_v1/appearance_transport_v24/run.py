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
from .method import Objective
from .checks import check_paths
from .transport import DonorData


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='APPEARANCE_TRANSPORT_V24'
    assert p['U_path']=='low_frequency_appearance_donor' and p['beta']==.05 and p['task_views']=='equal_mean_original_transported'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=18000,max_updates=2400,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v19_root'])/'results/s20261023_L_A_attempt001/ACTIVE_STEP_000400.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==400 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    try:
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=600,updates=0) as out:
            s=load();d=data(p,s.config,'RUNMC','LU');s.active['evidence']=copy.deepcopy(d.evidence)
            before=digest(s.payload());rng=s.generator.get_state();L=d.batch('L',1,torch.Generator().manual_seed(163800));U=DonorData(d,True).batch('U',1,torch.Generator().manual_seed(163801));LD=DonorData(d,False).batch('U',1,torch.Generator().manual_seed(163801))
            for b in (L,U,LD):
                b['x']=b['x'].cuda()
                if 'observations'in b:b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            paths=check_paths(s,L)
            parts=[]
            for use_u in (False,True):
                s.generator.set_state(rng);start=time.monotonic();loss,record=Objective(s,Refiner(s.model,s.phi,counter),use_u).estimate(L,U if use_u else LD,d.counts,s.generator);loss.backward();torch.cuda.synchronize()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for group in s.optimizer.param_groups for v in group['params'])
                assert s.q.mean.grad is not None and s.phi.c_head.weight.grad is not None
                assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
                parts.append(dict(use_u=use_u,seconds=time.monotonic()-start,parts={k:float(v) for k,v in record.items()}));s.optimizer.zero_grad(set_to_none=True)
                if not use_u:end_rng=s.generator.get_state()
                else:assert torch.equal(s.generator.get_state(),end_rng)
            assert parts[0]['parts']['task_original']==parts[1]['parts']['task_original'] and parts[0]['parts']['anchor']==parts[1]['parts']['anchor']
            s.generator.set_state(rng);assert digest(s.payload())==before
            write(out/'CHECKS.json',dict(gradient_paths=paths,real_batch=parts,paired_original_L_task_anchor=True,paired_global_rng=True,source_updates=0,real_batch_reverse_calls=2,path_reverse_calls=1,U_label_capability=False,counts=d.counts))
            if 1200*sum(v['seconds'] for v in parts)+3600>18000:raise RuntimeError('cost exceeds frozen budget')
            del s,d,L,U,LD
        cells={};common=None
        for arm,use_u in [('L',False),('LU',True)]:
            s=load();d=DonorData(data(p,s.config,'RUNMC','LU'),use_u);s.active['evidence']=copy.deepcopy(d.evidence);initial=digest(s.payload())
            if common is None:common=initial
            else:assert initial==common
            old_prior=digest(s.prior.state_dict());unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            write(root/f'BRANCH_{arm}.json',dict(common_payload=initial,initial_steps=400,original_Adam_inherited=True,prior_hash=old_prior,unused_heads_frozen=True,counts=d.counts,appearance_donor_pool='U' if use_u else 'L',donor_labels_read=False))
            status('TRAIN',arm=arm)
            run_stage(s,d,ledger,counter,arm,1600,6600,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=lambda st,ref:Objective(st,ref,use_u))
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=900,updates=0) as out:cells[arm+'_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=2400,preflight_reverse_calls=3,source_updates=0,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
