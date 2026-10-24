"""One targeted empty-only false-positive intervention with full-volume training readout."""
import json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from ..stream_state import digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import data,evaluate
from ..paired_semantic_v19.method import balanced_risk
from ..semantic_alignment_v23.method import Objective as Original
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from .method import Objective,risk_parts
from ..pixel_ce_v49.diagnostics import prepare,train_evaluate


def fixture():
    logits=torch.linspace(-2,2,24,dtype=torch.float64).reshape(3,2,2,2).requires_grad_()
    y=torch.tensor([[[0,1],[1,0]],[[0,0],[0,1]],[[0,0],[0,0]]]);truth=F.one_hot(y,2).permute(0,3,1,2).double();p=logits.softmax(1)
    r=risk_parts(p,truth);torch.testing.assert_close(r['pixel_ce'],F.cross_entropy(logits,y,reduction='none').mean((1,2)),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(r['balanced_ce']+r['dice'],balanced_risk(p,truth),rtol=0,atol=0)
    torch.testing.assert_close(r['empty_penalty'][:2],torch.zeros(2,dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(r['empty_penalty'][2:],-torch.log1p(-r['max_foreground'][2:]),rtol=1e-12,atol=1e-12)
    hi=torch.tensor([.9],dtype=torch.float64);lo=torch.tensor([.1],dtype=torch.float64)
    assert -torch.log1p(-hi)>-torch.log1p(-lo)
    r['empty_penalty'].mean().backward();assert torch.isfinite(logits.grad).all()
    return dict(status='PASS',cases=['both_classes','single_foreground_pixel','all_background','empty_max_penalty','positive_no_penalty'],fixture_reverse_calls=1,Adam_updates=0)


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='EMPTY_ONLY_V51'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=4800,max_updates=800,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    source=Path(p['v46_root'])
    def load():
        s=MultiscaleState.load(source/'results/W_target_attempt001/ACTIVE_INITIAL.pt');assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
        s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False);return s
    try:
        assert json.loads((source/'STATUS.json').read_text())['stage']=='COMPLETE' and json.loads((source/'EXIT.json').read_text())['exit_code']==0
        oldp=json.loads((source/'PRIVATE_PROTOCOL.json').read_text());assert all(p[k]==oldp[k] for k in ('config','data_root','data_version','v19_root'))
        for dep in ('training.py','stream_state.py','data_adapter.py','paired_semantic_v19/run.py','paired_semantic_v19/method.py','semantic_alignment_v23/method.py','semantic_alignment_v23/candidates.py','semantic_evidence_v22/method.py','unified_sslcl_v18/method.py','multiscale_recognition_v46/model.py'):
            assert (root/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes()==(source/'source/sc_bayeseg/hbsi_v1'/dep).read_bytes(),dep
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0):
            fcheck=fixture();s=load();base=data(p,s.config,'BMC','L');assert base.evidence==s.active['evidence'] and base.counts==dict(L=134,U=0)
            initial=digest(s.payload());prior=digest(s.prior.state_dict());oldmem=digest(s.memories['RUNMC'].state_dict());rng=s.generator.get_state()
            for candidate_seed in range(263000,263020):
                b=base.batch('L',1,torch.Generator().manual_seed(candidate_seed))
                if int(b['observations']['y_obs'][:,1].sum())==0:break
            else:raise AssertionError('no empty L fixture in bounded seed search')
            b['x']=b['x'].cuda();b['observations']={k:v.cuda() for k,v in b['observations'].items()}
            original,ov=Original(s,Refiner(s.model,s.phi,counter),False).estimate(b,None,base.counts,s.generator);end=s.generator.get_state()
            params=[v for g in s.optimizer.param_groups for v in g['params'] if v.requires_grad]
            expected=torch.autograd.grad(original,params,allow_unused=True)
            s.generator.set_state(rng);control,cv=Objective(s,Refiner(s.model,s.phi,counter),False).estimate(b,None,base.counts,s.generator)
            torch.testing.assert_close(control,original,rtol=0,atol=0);assert torch.equal(end,s.generator.get_state())
            actual=torch.autograd.grad(control,params,allow_unused=True);maxerr=0.
            for a,e in zip(actual,expected):
                assert (a is None)==(e is None)
                if a is not None:torch.testing.assert_close(a,e,rtol=0,atol=0);maxerr=max(maxerr,float((a-e).abs().max()))
            s.generator.set_state(rng);new,nv=Objective(s,Refiner(s.model,s.phi,counter),True).estimate(b,None,base.counts,s.generator)
            torch.testing.assert_close(new,original+nv['empty_penalty'],atol=1e-10,rtol=1e-10);assert torch.equal(end,s.generator.get_state())
            assert float(nv['empty_penalty'])>0
            new.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
            assert all(any(v.grad is not None and float(v.grad.norm())>0 for v in g['params']) for g in s.optimizer.param_groups)
            assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
            assert all(v.grad is None for m in s.memories.values() for v in m.parameters())
            s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==initial
            write(root/'PREFLIGHT.json',dict(status='PASS',fixture=fcheck,control_loss_and_gradient_exact=True,max_gradient_error=maxerr,new_minus_control_equals_empty_penalty=True,actual_empty_batch=True,model_objective_forwards=3,model_reverse_calls=3,Adam_updates=0,source_unchanged=True,original_loss=float(original),new_loss=float(new)))
            write(root/'MATCHING.json',dict(status='PASS',source_payload=initial,prior=prior,old_memory=oldmem,dependencies_equal=True,evidence_equal=True,fresh_Adam=True,same_sampling_RNG=True,old_validation_rescored=False,source_retrained=False))
            definitions=prepare(base,root)
            assert json.loads((root/'TRAIN_DEFINITION.json').read_text())==json.loads((Path(p['v49_root'])/'TRAIN_DEFINITION.json').read_text())
            del s,params,expected,actual,original,control,new,b
        s=load();assert digest(s.payload())==initial;unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
        status('TRAIN',arm='C3')
        run_stage(s,base,ledger,counter,'C3',800,1800,fingerprint(p),data_seed=263000,checkpoint_stride=200,objective_factory=lambda state,ref:Objective(state,ref,True))
        assert digest(s.memories['RUNMC'].state_dict())==oldmem and all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        status('TRAIN_DIAGNOSTIC',arm='C3')
        with ledger.attempt('train_diagnostic_C3',seconds=1200,updates=0) as out:train2=train_evaluate(s,base,definitions,out/'train')
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',arm='C3',domain=domain)
            with ledger.attempt('eval_C3_'+domain,seconds=600,updates=0) as out:cells[domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,train_diagnostics=dict(C3=train2),formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,preflight_model_objective_forwards=3,train_diagnostic_semantic_sample_forwards=12864,control_train_readout_reused=True,old_memory_and_Sb_unchanged=True,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
