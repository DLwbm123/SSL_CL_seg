"""One projected-gradient target-adaptation arm from the sealed W entrance."""
import json,os,time
from pathlib import Path
import torch
from ..stream_state import digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import data,evaluate
from ..semantic_alignment_v23.method import Objective as Original
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..pixel_ce_v49.diagnostics import prepare,train_evaluate
from .method import Objective


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='GRADIENT_GUARD_V52'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=5100,max_updates=800,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    source=Path(p['v46_root'])
    def load():
        s=MultiscaleState.load(source/'results/W_target_attempt001/ACTIVE_INITIAL.pt');assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False);return s
    try:
        assert json.loads((source/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((source/'EXIT.json').read_text())['exit_code']==0
        oldp=json.loads((source/'PRIVATE_PROTOCOL.json').read_text());assert all(p[k]==oldp[k] for k in ('config','data_root','data_version','v19_root'))
        for dep in ('training.py','stream_state.py','data_adapter.py','paired_semantic_v19/run.py','paired_semantic_v19/method.py','semantic_alignment_v23/method.py','semantic_alignment_v23/candidates.py','semantic_evidence_v22/method.py','unified_sslcl_v18/method.py','multiscale_recognition_v46/model.py','observation_block_v43/method.py'):
            assert (root/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes()==(source/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes(),dep
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            s=load();base=data(p,s.config,'BMC','L');assert base.evidence==s.active['evidence'] and base.counts==dict(L=134,U=0)
            initial=digest(s.payload());prior=digest(s.prior.state_dict());oldmem=digest(s.memories['RUNMC'].state_dict());rng=s.generator.get_state()
            b=base.batch('L',1,torch.Generator().manual_seed(263000));b['x']=b['x'].cuda();b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            params=[v for g in s.optimizer.param_groups for v in g['params'] if v.requires_grad]
            original,_=Original(s,Refiner(s.model,s.phi,counter),False).estimate(b,None,base.counts,s.generator);end=s.generator.get_state()
            expected=torch.autograd.grad(original,params,allow_unused=True)
            s.generator.set_state(rng);control,_=Objective(s,Refiner(s.model,s.phi,counter),False).estimate(b,None,base.counts,s.generator)
            torch.testing.assert_close(control,original,rtol=1e-9,atol=1e-10);assert torch.equal(end,s.generator.get_state())
            actual=torch.autograd.grad(control,params,allow_unused=True);maxerr=0.
            for a,e in zip(actual,expected):
                assert (a is None)==(e is None)
                if a is not None:torch.testing.assert_close(a,e,rtol=1e-8,atol=1e-10);maxerr=max(maxerr,float((a-e).abs().max()))
            s.generator.set_state(rng);projected=Objective(s,Refiner(s.model,s.phi,counter),True)
            new,_=projected.estimate(b,None,base.counts,s.generator)
            torch.testing.assert_close(new,original,rtol=1e-9,atol=1e-10);assert torch.equal(end,s.generator.get_state())
            s.optimizer.zero_grad(set_to_none=True);new.backward()
            for param,wanted in projected.expected:
                if param.grad is None:assert not bool(wanted.abs().any())
                else:torch.testing.assert_close(param.grad,wanted,atol=1e-9,rtol=1e-9)
            assert all(v.grad is None for m in s.memories.values() for v in m.parameters())
            assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
            groups=projected.summary()['groups'];assert all(v['steps']==1 for v in groups.values())
            s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==initial
            assert all(p[k] == json.loads((Path(p['v49_root'])/'PRIVATE_PROTOCOL.json').read_text())[k] for k in ('config','data_root','data_version','v19_root'))
            definitions=prepare(base,root)
            assert json.loads((root/'TRAIN_DEFINITION.json').read_text())==json.loads((Path(p['v49_root'])/'TRAIN_DEFINITION.json').read_text())
            write(root/'PREFLIGHT.json',dict(status='PASS',same_full_loss_and_gradient=True,max_gradient_error=maxerr,projected_gradient_formula=True,source_unchanged=True,model_objective_forwards=3,model_reverse_calls=5,Adam_updates=0,original_loss=float(original),projected_loss=float(new),sample_projection=groups))
            write(root/'MATCHING.json',dict(status='PASS',source_payload=initial,prior=prior,old_memory=oldmem,dependencies_equal=True,evidence_equal=True,fresh_Adam=True,same_sampling_RNG=True,old_validation_rescored=False,source_retrained=False))
            del s,params,expected,actual,original,control,new,projected,b
        s=load();assert digest(s.payload())==initial;unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
        holder=[]
        def create(state,ref):
            obj=Objective(state,ref,True);holder.append(obj);return obj
        def after(step):holder[0].after_step(step)
        status('TRAIN',arm='G')
        run_stage(s,base,ledger,counter,'G',800,2400,fingerprint(p),data_seed=263000,checkpoint_stride=200,objective_factory=create,after_checkpoint=after)
        audit=holder[0].summary();assert all(v['steps']==800 for v in audit['groups'].values()) and len(audit['adam_audits'])==4
        assert digest(s.memories['RUNMC'].state_dict())==oldmem and all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        status('TRAIN_DIAGNOSTIC',arm='G')
        with ledger.attempt('train_diagnostic_G',seconds=1200,updates=0) as out:train=train_evaluate(s,base,definitions,out/'train')
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',arm='G',domain=domain)
            with ledger.attempt('eval_G_'+domain,seconds=600,updates=0) as out:cells[domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,train_diagnostics=dict(G=train),gradient_audit=audit,formal_updates=800,preflight_model_reverse_calls=5,training_extra_reverse_calls=1600,train_diagnostic_semantic_sample_forwards=12864,control_train_readout_reused=True,old_memory_and_Sb_unchanged=True,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
