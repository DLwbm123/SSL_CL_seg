"""Same full objective and budgets, two conditional-recognition backbones."""
import json,os,time
from dataclasses import replace
from pathlib import Path
import torch
from ..config import Config
from ..stream_state import StreamState,digest
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..local_inference import Refiner
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..semantic_alignment_v23.method import Objective
from ..semantic_evidence_v22.method import sample_semantics
from .model import MultiscaleState

CLASSES={'S':StreamState,'W':MultiscaleState}


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());assert p['purpose']=='MULTISCALE_RECOGNITION_V46'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=replace(Config(**p['config']),seed=20261023);ledger=ResourceLedger(root/'results',max_seconds=16800,max_updates=4800,max_bytes=4_000_000_000);counter=Counters()
    assert cfg.hidden==16 and cfg.height==cfg.width==128
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));f.replace(root/'STATUS.json')
    def freeze(s):s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
    def objective(s,ref):return Objective(s,ref,False)
    def initial(arm):
        s=CLASSES[arm](cfg);d=data(p,cfg,'RUNMC','L');assert d.counts==dict(L=80,U=0)
        s.begin_task('RUNMC',d.manifest_hash,d.evidence);s.to_device('cuda');freeze(s);return s,d
    def real_batch(d,seed):
        b=d.batch('L',1,torch.Generator().manual_seed(seed));b['x']=b['x'].cuda();b['observations']={k:v.cuda() for k,v in b['observations'].items()};return b
    def check_backward(s,d,seed):
        b=real_batch(d,seed);before=digest(s.payload());rng=s.generator.get_state();start=time.monotonic()
        loss,parts=objective(s,Refiner(s.model,s.phi,counter)).estimate(b,None,d.counts,s.generator);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for g in s.optimizer.param_groups for v in g['params'])
        assert all(any(v.grad is not None and float(v.grad.norm())>0 for v in g['params']) for g in s.optimizer.param_groups)
        assert s.phi.s_head.weight.grad is None and s.phi.b_head.weight.grad is None
        assert all(v.grad is None for m in s.memories.values() for v in m.parameters())
        if isinstance(s,MultiscaleState):assert any(v.grad is not None and float(v.grad.norm())>0 for v in s.phi.backbone.parameters())
        elapsed=time.monotonic()-start;s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);assert digest(s.payload())==before
        return dict(loss=float(loss),seconds=elapsed,source_restored=True,finite_gradients=True),b
    def sample(s,x):
        rng=torch.Generator().manual_seed(46000000);values=[]
        with torch.no_grad():
            for _ in range(2):
                theta=s.q.rsample(rng);a=s.current.rsample(theta,rng)
                for _ in range(2):values.append(sample_semantics(s,x,theta,a,rng)[0])
        return torch.stack(values).mean(0)
    resumed=os.environ.get('RESUME_SEALED_SOURCE')=='1'
    try:
        checks={};common=None
        if not resumed:
            status('PREFLIGHT')
            with ledger.attempt('preflight',seconds=300,updates=0) as out:
                for arm in ('S','W'):
                    s,d=initial(arm)
                    signature=dict(q=digest(s.q.state_dict()),prior=digest(s.prior.state_dict()),current=digest(s.current.state_dict()),rng=digest(s.generator.get_state()),heads={n:digest(getattr(s.phi,n).state_dict()) for n in ('c_head','r_head','s_head','b_head')},evidence=d.evidence)
                    if common is None:common=signature
                    else:assert signature==common
                    source,b=check_backward(s,d,163000);cp=out/f'PRIVATE_{arm}_ENTRY.pt';s.save(cp);restored=CLASSES[arm].load(cp).to_device('cuda');freeze(restored)
                    assert digest(s.payload())==digest(restored.payload())
                    torch.testing.assert_close(sample(s,b['x']),sample(restored,b['x']),rtol=0,atol=0)
                    params=dict(phi=sum(v.numel() for v in s.phi.parameters()),q=sum(v.numel() for v in s.q.parameters()),current=sum(v.numel() for v in s.current.parameters()))
                    del restored,b
                    s.finalize_task();d=data(p,cfg,'BMC','L');s.begin_task('BMC',d.manifest_hash,d.evidence);freeze(s);s.to_device('cuda')
                    target,b=check_backward(s,d,263000);del b
                    checks[arm]=dict(source=source,target=target,parameters=params,checkpoint_roundtrip_and_sample_exact=True)
                    del s,d
                write(root/'PREFLIGHT.json',dict(status='PASS',common_initial_q_heads_prior_domain_rng=True,policies=checks,model_reverse_calls=4,optimizer_updates=0,serialization_semantic_forwards=16,target_fixture_uses_untrained_source=True))
        cells={}
        if resumed:
            assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
            assert ledger.data['updates_reserved']==1600
            assert set(ledger.data['training_sessions'])=={'W_source'}
            assert ledger.data['training_sessions']['W_source']['status']=='COMPLETE'
            assert all(v['status']=='PASS' for v in ledger.data['attempts'])
            for domain in ('RUNMC','BMC'):
                key='W_source_'+domain
                cells[key]=json.loads((root/'results'/('eval_'+key)/'validation/SUMMARY.json').read_text())
                assert cells[key]['complete']
            previous=MultiscaleState.load(root/'results/W_source_attempt001/PRIVATE_SEALED_STATE.pt')
            counter.values.update(previous.receipts['RUNMC']['cost']);del previous
            assert counter.values['outer_backward_calls']==1600
        for arm in ('W','S'):
            if resumed and arm=='W':
                s=MultiscaleState.load(root/'results/W_source_attempt001/PRIVATE_SEALED_STATE.pt').to_device('cuda');freeze(s)
                unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
                s.steps=s.receipts['RUNMC']['steps']
            else:
                s,d=initial(arm);unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
                write(root/f'BRANCH_{arm}.json',dict(q=digest(s.q.state_dict()),prior=digest(s.prior.state_dict()),initial_steps=0,source_target_updates=[1600,800],architecture=arm))
                status('TRAIN_SOURCE',arm=arm)
                run_stage(s,d,ledger,counter,arm+'_source',1600,4000,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=objective)
                assert all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
                for domain in ('RUNMC','BMC'):
                    status('EVALUATE_SOURCE',arm=arm,domain=domain)
                    with ledger.attempt('eval_'+arm+'_source_'+domain,seconds=450,updates=0) as out:cells[arm+'_source_'+domain]=evaluate(p,s,domain,'RUNMC',out/'validation')
            source_path=root/'results'/f'{arm}_source_attempt001/PRIVATE_SEALED_STATE.pt'
            source_digest=digest(s.payload());s=CLASSES[arm].load(source_path);s.steps=s.receipts['RUNMC']['steps'];assert digest(s.payload())==source_digest
            old=digest(s.memories['RUNMC'].state_dict());d=data(p,cfg,'BMC','L');s.begin_task('BMC',d.manifest_hash,d.evidence);freeze(s);s.to_device('cuda')
            assert not s.optimizer.state and d.counts['U']==0
            status('TRAIN_TARGET',arm=arm)
            run_stage(s,d,ledger,counter,arm+'_target',800,2400,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=objective)
            assert digest(s.memories['RUNMC'].state_dict())==old and all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE_TARGET',arm=arm,domain=domain)
                with ledger.attempt('eval_'+arm+'_target_'+domain,seconds=450,updates=0) as out:cells[arm+'_target_'+domain]=evaluate(p,s,domain,domain,out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in cells.items() if k.startswith(arm+'_')});del s,d
        write(root/'RESULTS.json',dict(cells=cells,new_formal_updates=4800,preflight_model_reverse_calls=4,preflight_optimizer_updates=0,serialization_semantic_forwards=16,counter=counter.export(),old_memory_and_Sb_unchanged=True,sealed_source_receipt_continuation=resumed))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
