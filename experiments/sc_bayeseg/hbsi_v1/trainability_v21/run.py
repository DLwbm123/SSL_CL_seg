"""Selective adaptation of the existing probabilistic mainline; one training engine."""
import json
import os
from pathlib import Path
import time
import torch
from ..stream_state import StreamState, digest
from ..paired_semantic_v19.run import data, evaluate
from ..paired_semantic_v19.method import UnifiedObjective
from ..local_inference import Refiner
from ..budgets import ResourceLedger, Counters
from ..training import run_stage
from ..qualification import fingerprint
from ..real_diagnostic import write


def freeze(s,arm):
    s.phi.requires_grad_(False)
    if arm=='DOMAIN_ONLY':s.q.requires_grad_(False)
    elif arm!='FIX_PHI':raise ValueError('unknown policy')
    return {name:digest(getattr(s,name).state_dict()) for name in ('phi','q') if not any(v.requires_grad for v in getattr(s,name).parameters())}


def check_frozen(s,expected):
    for name,identity in expected.items():
        module=getattr(s,name)
        assert digest(module.state_dict())==identity, 'frozen state changed'
        assert all(v.grad is None for v in module.parameters()), 'frozen gradient'
        assert all(not s.optimizer.state.get(v) for v in module.parameters()), 'frozen Adam slots changed'


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='TARGET_TRAINABILITY_V21_Q1'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    donor=Path(p['v19_root']);checkpoint=donor/'results/s20261023_L_M0_B_attempt001/ACTIVE_INITIAL.pt'
    ledger=ResourceLedger(root/'results',max_seconds=7200,max_updates=1600,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**extra):
        tmp=root/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2));tmp.replace(root/'STATUS.json')
    baseline={}
    for phase,filename,prefix in [('source','SOURCE_20261023_L.json','s20261023_L_A'),('FULL','TARGET_20261023_L_M0.json','s20261023_L_M0_B')]:
        values=json.loads((donor/filename).read_text())
        baseline[phase]={domain:values[prefix+'_'+domain] for domain in ('RUNMC','BMC')}
    write(root/'REUSED_BASELINE.json',baseline)
    results={}
    try:
        common=None
        for arm in ('FIX_PHI','DOMAIN_ONLY'):
            s=StreamState.load(checkpoint)
            assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
            donor_payload=digest(s.payload());s.runtime=None
            d=data(p,s.config,'BMC','L')
            assert d.counts['U']==0 and s.active['evidence']==d.evidence
            s.to_device('cuda');payload=digest(s.payload())
            if common is None:common=payload
            else:assert payload==common
            frozen=freeze(s,arm)
            write(root/f'POLICY_{arm}.json',dict(frozen_modules=list(frozen),frozen_payloads=frozen,donor_payload=donor_payload,common_payload=payload,source_steps=0,requires_grad_reapply_required_on_restore=True))
            status('PREFLIGHT',arm=arm)
            with ledger.attempt(f'preflight_{arm}',seconds=300,updates=0) as out:
                L=d.batch('L',1,torch.Generator().manual_seed(263000));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
                rng=s.generator.get_state();before=digest(s.payload())
                loss,_=UnifiedObjective(s,Refiner(s.model,s.phi,counter)).estimate(L,None,d.counts,s.generator);loss.backward()
                for group in s.optimizer.param_groups:
                    for param in group['params']:
                        if param.requires_grad:assert param.grad is not None and torch.isfinite(param.grad).all()
                        else:assert param.grad is None
                s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng)
                assert digest(s.payload())==before
                check_frozen(s,frozen)
                write(out/'CHECKS.json',dict(finite_trainable_gradients=True,frozen_gradients_none=True,original_rng_restored=True,preflight_reverse_calls=1,preflight_optimizer_updates=0))
            status('TRAIN',arm=arm)
            def guard(step):
                if step%400==0:check_frozen(s,frozen)
            run_stage(s,d,ledger,counter,arm,800,1800,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=UnifiedObjective,after_checkpoint=guard)
            # finalize_task removes optimizer; frozen tensors must still be unchanged.
            assert all(digest(getattr(s,name).state_dict())==value for name,value in frozen.items())
            for domain in ('RUNMC','BMC'):
                status('EVALUATE',arm=arm,domain=domain)
                with ledger.attempt(f'eval_{arm}_{domain}',seconds=900,updates=0) as out:
                    results[arm+'_'+domain]=evaluate(p,s,domain,domain,out/'validation')
            write(root/f'ARM_{arm}.json',{key:value for key,value in results.items() if key.startswith(arm+'_')})
            del s,d
        write(root/'RESULTS.json',dict(cells=results,reused=baseline,new_updates=1600,preflight_reverse_calls=2,frozen_state_checks='PASS',counter=counter.export()))
        status('COMPLETE',scientific_claim='requires_analysis')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
