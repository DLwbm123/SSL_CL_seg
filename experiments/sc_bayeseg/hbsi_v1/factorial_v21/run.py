"""One new factorial cell; archived F11 is reused only after bounded matching."""
import importlib.util
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
from ..trainability_v21.run import check_frozen


def policy(s,arm):
    s.q.requires_grad_(arm in ('F10','F11'))
    s.phi.requires_grad_(arm in ('F01','F11'))
    return {n:digest(getattr(s,n).state_dict()) for n in ('q','phi') if not any(p.requires_grad for p in getattr(s,n).parameters())}


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='TARGET_FACTORIAL_V21_F01'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    donor=Path(p['v19_root']);q1=Path(p['q1_root']);checkpoint=donor/'results/s20261023_L_M0_B_attempt001/ACTIVE_INITIAL.pt'
    ledger=ResourceLedger(root/'results',max_seconds=5400,max_updates=824,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        temp=root/'STATUS.tmp';temp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));temp.replace(root/'STATUS.json')
    def load():
        s=StreamState.load(checkpoint);assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
        s.runtime=None;s.to_device('cuda');return s
    try:
        status('MATCHING')
        # A repair may consume the still-reserved 24 audit updates from a 0-update failure.
        audit_reserved=any(v['name']=='matching' for v in ledger.data['attempts'])
        with ledger.attempt('matching_repair002' if audit_reserved else 'matching',seconds=600,updates=0 if audit_reserved else 24) as out:
            here=root/'source/sc_bayeseg/hbsi_v1';old=donor/'source/sc_bayeseg/hbsi_v1';qsource=q1/'source/sc_bayeseg/hbsi_v1'
            files=sorted(f.name for f in here.glob('*.py'))+['paired_semantic_v19/method.py','paired_semantic_v19/run.py','unified_sslcl_v18/method.py']
            for name in files:assert (here/name).read_bytes()==(old/name).read_bytes()==(qsource/name).read_bytes(),name
            oldledger=json.loads((donor/'results/RESOURCE_LEDGER.json').read_text());qledger=json.loads((q1/'results/RESOURCE_LEDGER.json').read_text())
            orig=oldledger['training_sessions']['s20261023_L_M0_B']['identity']
            for arm in ('FIX_PHI','DOMAIN_ONLY'):
                identity=qledger['training_sessions'][arm]['identity']
                for key in ('code_fingerprint','config_fingerprint','evidence_fingerprint','prior_hash'):assert identity[key]==orig[key],key
            oldspec=importlib.util.spec_from_file_location('sc_bayeseg.hbsi_v1.paired_semantic_v19.archived_matching',old/'paired_semantic_v19/method.py')
            oldmodule=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(oldmodule)
            a,b=load(),load();initial=digest(a.payload());assert digest(b.payload())==initial
            assert initial==json.loads((q1/'POLICY_FIX_PHI.json').read_text())['common_payload']
            d=data(p,a.config,'BMC','L');assert d.counts['U']==0 and d.evidence==a.active['evidence']
            def copy_step(s,objective_type,i):
                L=d.batch('L',1,torch.Generator().manual_seed(263000+2*i));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
                loss,_=objective_type(s,Refiner(s.model,s.phi,Counters())).estimate(L,None,d.counts,s.generator)
                s.optimizer.zero_grad(set_to_none=True);loss.backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for group in s.optimizer.param_groups for v in group['params'])
                s.optimizer.step();s.steps+=1
                return float(loss.detach())
            for i in range(10):
                assert copy_step(a,oldmodule.UnifiedObjective,i)==copy_step(b,UnifiedObjective,i)
                assert digest(a.payload())==digest(b.payload()),'F11 replay mismatch'
            del a,b
            for arm in ('F00','F10','F01','F11'):
                s=load();frozen=policy(s,arm);copy_step(s,UnifiedObjective,0);check_frozen(s,frozen);del s
            assert digest(load().payload())==initial
            match=dict(reuse_F11='PASS',core_and_objective_readout_files_equal=True,config_evidence_prior_equal=True,common_payload_equal=True,F11_replay_steps_per_implementation=10,replay_payload_loss_RNG_Adam_equal=True,policy_copy_steps=4,discarded_Adam_updates=24,reverse_calls=24,source_updates=0)
            write(root/'MATCHING.json',match);write(out/'CHECKS.json',match)
        s=load();s.runtime=None;d=data(p,s.config,'BMC','L');frozen=policy(s,'F01');old_memory=digest(s.memories['RUNMC'].state_dict());phi_before=digest(s.phi.state_dict());domain_before=digest(s.current.state_dict())
        write(root/'POLICY.json',dict(arm='F01',frozen_modules=list(frozen),frozen_payloads=frozen,requires_grad_reapply_required_on_restore=True))
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=300,updates=0) as out:
            L=d.batch('L',1,torch.Generator().manual_seed(263000));L['x']=L['x'].cuda();L['observations']={k:v.cuda() for k,v in L['observations'].items()}
            before=digest(s.payload());rng=s.generator.get_state()
            loss,_=UnifiedObjective(s,Refiner(s.model,s.phi,counter)).estimate(L,None,d.counts,s.generator);loss.backward()
            for group in s.optimizer.param_groups:
                for v in group['params']:
                    if v.requires_grad:assert v.grad is not None and torch.isfinite(v.grad).all()
                    else:assert v.grad is None
            s.optimizer.zero_grad(set_to_none=True);s.generator.set_state(rng);check_frozen(s,frozen);assert digest(s.payload())==before
            write(out/'CHECKS.json',dict(finite_trainable_gradients=True,frozen_gradients_none=True,RNG_restored=True,reverse_calls=1,optimizer_updates=0))
        status('TRAIN',arm='F01')
        def guard(step):
            if step%400==0:check_frozen(s,frozen)
        run_stage(s,d,ledger,counter,'F01',800,1800,fingerprint(p),data_seed=263000,checkpoint_stride=400,objective_factory=UnifiedObjective,after_checkpoint=guard)
        assert digest(s.q.state_dict())==frozen['q'] and digest(s.memories['RUNMC'].state_dict())==old_memory
        assert digest(s.phi.state_dict())!=phi_before and digest(s.memories['BMC'].state_dict())!=domain_before
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',domain=domain)
            with ledger.attempt('eval_F01_'+domain,seconds=900,updates=0) as out:cells[domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,formal_updates=800,discarded_copy_updates=24,preflight_reverse_calls=1,frozen_checks='PASS',phi_and_domain_changed=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
