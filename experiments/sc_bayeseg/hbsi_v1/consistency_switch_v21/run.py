"""Bounded consistency and domain-memory diagnostics; never trains source states."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from .semantic_probe import run as semantic_run, quality
from ..stream_state import StreamState, digest
from ..budgets import ResourceLedger
from ..real_diagnostic import write
from ..paired_semantic_v19.run import evaluate


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='CONSISTENCY_SWITCH_V21_Q0'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=7200,max_updates=0,max_bytes=2_000_000_000)
    def atomic(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
    def status(stage,**extra):atomic(root/'STATUS.json',dict(stage=stage,time=time.time(),**extra))
    try:
        prob=torch.tensor([[[[.95,.05,.2,.8]],[[.05,.95,.8,.2]]]],dtype=torch.float64)
        draws=prob[None].repeat(4,1,1,1,1)
        truth=torch.tensor([[[[1,1,0,0]],[[0,0,1,1]]]],dtype=torch.bool)
        q=quality(draws,draws,truth)
        assert q['label_audit']['precision']==.5 and q['label_audit']['recall']==.5
        assert q['label_audit']['high_confidence_error_rate']==.5
        assert q['classes']['foreground']['coverage']==.5 and q['mask_agreement']==1.
        assert 'label_audit' not in quality(draws,draws)
        write(root/'SELF_CHECK.json',dict(quality_fixture='PASS',unlabeled_audit_has_no_GT='PASS',source_optimizer_updates=0))
        status('CONSISTENCY_DIAGNOSTIC')
        with ledger.attempt('Q_sem',seconds=1200,updates=0):
            sem=semantic_run(p,root,lambda v:atomic(root/'Q_SEM_PROGRESS.json',v));write(root/'Q_SEM_RESULTS.json',sem)
        status('MEMORY_SWITCH_DIAGNOSTIC')
        donor=Path(p['v19_root'])
        source=StreamState.load(donor/'results/s20261023_L_A_attempt001/PRIVATE_SEALED_STATE.pt')
        initial=StreamState.load(donor/'results/s20261023_L_M0_B_attempt001/ACTIVE_INITIAL.pt')
        post=StreamState.load(donor/'results/s20261023_L_M0_B_attempt001/PRIVATE_SEALED_STATE.pt')
        for module in ('q','phi','model','projection'):
            assert digest(getattr(source,module).state_dict())==digest(getattr(initial,module).state_dict()), 'initial checkpoint differs from donor'
        assert digest(source.memories['RUNMC'].state_dict())==digest(post.memories['RUNMC'].state_dict())
        # Donor scores are reused verbatim, including original prediction repeat denominator.
        t0=json.loads((donor/'SOURCE_20261023_L.json').read_text())['s20261023_L_A_BMC']
        tpost=json.loads((donor/'TARGET_20261023_L_M0.json').read_text())['s20261023_L_M0_B_BMC']
        assert t0['complete'] and tpost['complete']
        results={'T0':dict(summary=t0,reused=True),'T_post_target':dict(summary=tpost,reused=True)}
        for name,state,memory in [('T_switch',initial,'BMC'),('T_post_source',post,'RUNMC')]:
            state.to_device('cuda');before=digest(state.payload())
            if name=='T_switch':
                state.memories['BMC']=copy.deepcopy(state.current).requires_grad_(False)
            probe_before=digest(state.payload());status('MEMORY_READOUT',readout=name)
            with ledger.attempt(name,seconds=900,updates=0) as out:
                summary=evaluate(p,state,'BMC',memory,out/'validation')
            assert digest(state.payload())==probe_before
            if name=='T_switch':del state.memories['BMC']
            assert digest(state.payload())==before
            results[name]=dict(summary=summary,reused=False,payload_unchanged=True)
            write(root/f'{name}.json',results[name])
        write(root/'Q_SWITCH_RESULTS.json',dict(status='COMPLETE',cells=results,donor_checks='PASS',source_updates=0,new_evaluation_cells=2,reused_cells=2,training_steps=0))
        status('COMPLETE',reverse_calls=sem['reverse_calls'],discarded_candidate_steps=sem['candidate_steps_discarded'],source_updates=0)
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
