"""One frozen 400-update target-block intervention; sealed A is never rescored."""
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
from .method import Objective


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='OBSERVATION_BLOCK_V43' and p['arms']==['T']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=3000,max_updates=400,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    checkpoint=Path(p['v23_root'])/'results/L_attempt001/ACTIVE_STEP_001600.pt'
    def load():
        s=StreamState.load(checkpoint);assert s.steps==1600 and s.active['id']=='RUNMC'
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
        return s
    try:
        previous=Path(p['v25_root']);old=json.loads((previous/'PRIVATE_PROTOCOL.json').read_text())
        assert json.loads((previous/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((previous/'EXIT.json').read_text())['exit_code']==0
        results=json.loads((previous/'RESULTS.json').read_text())
        assert results['new_formal_updates']==1200 and all(c['complete'] for c in results['cells'].values())
        oldledger=json.loads((previous/'results/RESOURCE_LEDGER.json').read_text())
        assert len(oldledger['training_sessions'])==3 and all(v['status']=='COMPLETE' and v['completed_steps']==2000 for v in oldledger['training_sessions'].values())
        assert all(p[k]==old[k] for k in ('config','data_root','data_version','v19_root','v23_root'))
        dependencies=('training.py','stream_state.py','data_adapter.py','paired_semantic_v19/run.py','paired_semantic_v19/method.py','posterior_teacher_v25/method.py','semantic_alignment_v23/method.py','semantic_alignment_v23/candidates.py','semantic_evidence_v22/method.py')
        for dependency in dependencies:
            assert (root/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes()==(previous/'source/sc_bayeseg/hbsi_v1'/dependency).read_bytes(),dependency
        reference=json.loads((previous/'BRANCH_A.json').read_text());status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            s=load();d=data(p,s.config,'RUNMC','LU');assert s.active['evidence']==d.evidence
            before=digest(s.payload());assert before==reference['common_payload'];rng=s.generator.get_state()
            assert digest(s.prior.state_dict())==reference['prior_hash']
            L=d.batch('L',1,torch.Generator().manual_seed(166200));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
            original,ref=OriginalA(s,Refiner(s.model,s.phi,counter),'A').estimate(L,None,d.counts,s.generator)
            refend=s.generator.get_state()
            frozen=json.loads((previous/'PREFLIGHT.json').read_text())['real_batch'][0]['parts']
            assert all(abs(float(ref[k])-frozen[k])<1e-10 for k in ('task','anchor','image_L','global_kl','domain_kl','scaled'))
            obj=Objective(s,Refiner(s.model,s.phi,counter));s.generator.set_state(rng)
            loss,parts=obj.components(L,d.counts,s.generator,preflight=True)
            assert torch.equal(refend,s.generator.get_state())
            assert torch.allclose(original,parts['full_objective_same_samples'],atol=1e-10,rtol=0)
            assert torch.allclose(loss,original+.01*parts['score_x']-parts['anchor'],atol=1e-10,rtol=0)
            retained=parts['task']-.01*parts['local_prior_minus_q']+(parts['global_kl']+parts['domain_kl'])/(d.counts['L']*s.config.height*s.config.width)
            params=[q for g in s.optimizer.param_groups for q in g['params'] if q.requires_grad]
            expected=torch.autograd.grad(retained,params,allow_unused=True)
            # Fresh training path has the removed block under no_grad.
            s.generator.set_state(rng);actual,record=obj.estimate(L,None,d.counts,s.generator);actual.backward()
            assert torch.equal(refend,s.generator.get_state())
            assert torch.allclose(actual,loss,atol=1e-10,rtol=0)
            maxerr=0.
            for q,e in zip(params,expected):
                assert (q.grad is None)==(e is None)
                if e is not None:
                    assert torch.isfinite(q.grad).all() and torch.allclose(q.grad,e,atol=1e-10,rtol=1e-9)
                    maxerr=max(maxerr,float((q.grad-e).abs().max()))
            assert s.q.mean.grad is not None and s.phi.c_head.weight.grad is not None
            assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
            s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==before
            write(root/'PREFLIGHT.json',dict(status='PASS',source_unchanged=True,original_A_fixed_batch_reproduced=True,algebra_match=True,retained_gradient_match=True,max_gradient_error=maxerr,global_rng_match=True,model_objective_forward_evaluations=3,model_reverse_calls=2,optimizer_updates=0,parts={k:float(v) for k,v in record.items()}))
            write(root/'MATCHING.json',dict(status='PASS',same_config_data_evidence=True,critical_dependencies_match_v25=True,common_payload=before,prior_hash=reference['prior_hash'],reused_control='v25_A2000',old_endpoints_rescored=False,teacher_cache_loaded=False))
            del s,d,L,params,expected,original,loss,parts,retained,actual,obj
        s=load();d=data(p,s.config,'RUNMC','LU');assert digest(s.payload())==reference['common_payload']
        unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
        write(root/'BRANCH_T.json',dict(common_payload=digest(s.payload()),prior_hash=digest(s.prior.state_dict()),initial_steps=1600,Adam_inherited=True))
        status('TRAIN',arm='T')
        run_stage(s,d,ledger,counter,'T',2000,1200,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=Objective)
        assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',arm='T',domain=domain)
            with ledger.attempt('eval_T_'+domain,seconds=450,updates=0) as out:cells['T_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
        write(root/'ARM_T.json',cells)
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=400,preflight_model_reverse_calls=2,preflight_objective_forwards=3,training_audit_reverse_calls=0,source_updates=0,common_entrance=True,unused_heads_unchanged=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
