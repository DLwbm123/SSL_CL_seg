"""Two source-continuation update policies under the unchanged v23 L objective."""
import json,os,time
from pathlib import Path
import torch
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..posterior_teacher_v25.method import Objective as OriginalA

ARMS=('FIX_Q','FIX_PHI')


def freeze(s,arm):
    name={'FIX_Q':'q','FIX_PHI':'phi'}[arm];module=getattr(s,name)
    params=list(module.parameters());module.requires_grad_(False)
    return dict(module=name,tensors=digest(module.state_dict()),Adam_slots=digest([s.optimizer.state.get(q,{}) for q in params]))


def check_frozen(s,expected):
    module=getattr(s,expected['module']);params=list(module.parameters())
    assert all(q.grad is None and not q.requires_grad for q in params)
    assert digest(module.state_dict())==expected['tensors']
    assert digest([s.optimizer.state.get(q,{}) for q in params])==expected['Adam_slots']


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='SOURCE_UPDATE_V44' and p['arms']==list(ARMS)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=4800,max_updates=800,max_bytes=3_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v23_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==1600 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    def objective(s,refiner):return OriginalA(s,refiner,'A')
    try:
        previous=Path(p['v25_root']);old=json.loads((previous/'PRIVATE_PROTOCOL.json').read_text())
        assert json.loads((previous/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((previous/'EXIT.json').read_text())['exit_code']==0
        results=json.loads((previous/'RESULTS.json').read_text())
        assert results['new_formal_updates']==1200 and all(c['complete'] for c in results['cells'].values())
        oldledger=json.loads((previous/'results/RESOURCE_LEDGER.json').read_text())
        assert len(oldledger['training_sessions'])==3 and all(v['status']=='COMPLETE' and v['completed_steps']==2000 for v in oldledger['training_sessions'].values())
        assert all(p[k]==old[k] for k in ('config','data_root','data_version','v19_root','v23_root'))
        dependencies=('training.py','stream_state.py','data_adapter.py','global_posterior.py','recognition.py','conditional_memory.py','paired_semantic_v19/run.py','paired_semantic_v19/method.py','posterior_teacher_v25/method.py','semantic_alignment_v23/method.py','semantic_alignment_v23/candidates.py','semantic_evidence_v22/method.py')
        for dependency in dependencies:
            assert (root/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes()==(previous/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes(),dependency
        reference=json.loads((previous/'BRANCH_A.json').read_text());status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            s=load();d=data(p,s.config,'RUNMC','LU');assert s.active['evidence']==d.evidence
            before=digest(s.payload());assert before==reference['common_payload'];rng=s.generator.get_state()
            assert digest(s.prior.state_dict())==reference['prior_hash']
            L=d.batch('L',1,torch.Generator().manual_seed(166200));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
            loss,ref=objective(s,Refiner(s.model,s.phi,counter)).estimate(L,None,d.counts,s.generator);refend=s.generator.get_state()
            frozen=json.loads((previous/'PREFLIGHT.json').read_text())['real_batch'][0]['parts']
            assert all(abs(float(ref[k])-frozen[k])<1e-10 for k in ('scaled','task','anchor','image_L','global_kl','domain_kl'))
            s.generator.set_state(rng);assert digest(s.payload())==before
            ref={k:v.detach() for k,v in ref.items()};del loss,s
            checks=[]
            for arm in ARMS:
                s=load();assert digest(s.payload())==before;policy=freeze(s,arm)
                loss,parts=objective(s,Refiner(s.model,s.phi,counter)).estimate(L,None,d.counts,s.generator);loss.backward()
                assert all(torch.equal(parts[k],ref[k]) for k in ref)
                assert torch.equal(refend,s.generator.get_state())
                check_frozen(s,policy)
                for group in s.optimizer.param_groups:
                    for q in group['params']:assert q.grad is None or torch.isfinite(q.grad).all()
                other=s.phi if arm=='FIX_Q' else s.q
                assert any(q.grad is not None and float(q.grad.norm())>0 for q in other.parameters())
                assert any(q.grad is not None and float(q.grad.norm())>0 for q in s.current.parameters())
                assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
                s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==before
                checks.append(dict(arm=arm,loss=float(loss),same_original_objective=True,frozen_gradients_none=True,frozen_tensors_and_inherited_Adam_slots_unchanged=True,other_module_and_domain_gradients_nonzero=True,source_restored=True));del s,loss
            write(root/'PREFLIGHT.json',dict(status='PASS',original_A_fixed_batch_reproduced=True,common_global_rng=True,policies=checks,complete_objective_forward_evaluations=3,model_reverse_calls=2,optimizer_updates=0))
            write(root/'MATCHING.json',dict(status='PASS',same_config_data_evidence=True,critical_dependencies_match_v25=True,common_payload=before,prior_hash=reference['prior_hash'],reused_control='v25_A2000',old_endpoints_rescored=False,teacher_cache_loaded=False))
        cells={}
        for arm in ARMS:
            s=load();d=data(p,s.config,'RUNMC','LU');assert digest(s.payload())==reference['common_payload']
            initial=digest(s.payload());prior=digest(s.prior.state_dict());policy=freeze(s,arm)
            unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            other='phi' if arm=='FIX_Q' else 'q';other_initial=digest(getattr(s,other).state_dict());domain_initial=digest(s.current.state_dict())
            write(root/f'BRANCH_{arm}.json',dict(common_payload=initial,prior_hash=prior,initial_steps=1600,Adam_inherited=True,policy=policy,requires_grad_reapply_on_restore=True))
            status('TRAIN',arm=arm)
            def guard(step):
                assert all(q.grad is None for q in getattr(s,policy['module']).parameters())
                if step==2000:
                    check_frozen(s,policy)
                    assert digest(getattr(s,other).state_dict())!=other_initial and digest(s.current.state_dict())!=domain_initial
                    write(root/f'FROZEN_CHECK_{arm}.json',dict(status='PASS',step=step,frozen_tensors_unchanged=True,inherited_Adam_slots_unchanged=True,other_module_changed=True,domain_changed=True))
            run_stage(s,d,ledger,counter,arm,2000,1200,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=objective,after_checkpoint=guard)
            assert digest(getattr(s,policy['module']).state_dict())==policy['tensors']
            assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_'+domain,seconds=450,updates=0) as out:cells[arm+'_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=800,preflight_model_reverse_calls=2,preflight_complete_objective_forward_evaluations=3,training_audit_reverse_calls=0,source_updates=0,common_entrance=True,frozen_state_and_Adam_checks='PASS',unused_heads_unchanged=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
