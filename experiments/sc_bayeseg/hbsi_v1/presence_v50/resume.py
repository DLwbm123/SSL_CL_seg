"""Bounded continuation after an operational preflight comparison error."""
import json,os,time
from pathlib import Path
import torch
from ..stream_state import digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import data,evaluate
from ..budgets import ResourceLedger,Counters
from ..training import run_stage
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..pixel_ce_v49.diagnostics import train_evaluate
from .method import Objective


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='PRESENCE_V50'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=4800,max_updates=800,max_bytes=2_000_000_000);counter=Counters()
    def status(stage,**kw):
        f=root/'STATUS.tmp';f.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));f.replace(root/'STATUS.json')
    try:
        status('REPAIR_PREFLIGHT')
        with ledger.attempt('preflight_repair002',seconds=120,updates=0):
            attempts=ledger.data['attempts']
            assert len(attempts)==2 and attempts[0]['name']=='preflight' and attempts[0]['status']=='FAILED'
            assert attempts[0]['error_type']=='AssertionError' and ledger.data['updates_reserved']==0
            assert json.loads((root/'failure_entry001/EXIT.json').read_text())['exit_code']==1
            assert json.loads((root/'PREFLIGHT.json').read_text())['status']=='PASS'
            match=json.loads((root/'MATCHING.json').read_text());assert match['status']=='PASS'
            summary=json.loads((root/'TRAIN_DEFINITION.json').read_text())
            assert summary==json.loads((Path(p['v49_root'])/'TRAIN_DEFINITION.json').read_text())
            source=Path(p['v46_root']);s=MultiscaleState.load(source/'results/W_target_attempt001/ACTIVE_INITIAL.pt')
            assert s.steps==0 and s.active['id']=='BMC' and not s.optimizer.state
            s.runtime=None;s.to_device('cuda');s.phi.s_head.requires_grad_(False);s.phi.b_head.requires_grad_(False)
            assert digest(s.payload())==match['source_payload'] and digest(s.prior.state_dict())==match['prior']
            assert digest(s.memories['RUNMC'].state_dict())==match['old_memory']
            base=data(p,s.config,'BMC','L');assert base.evidence==s.active['evidence'] and base.counts==dict(L=134,U=0)
            groups={}
            for r in base.records['L']:groups.setdefault(str(r['image']),[]).append(r)
            assert len(groups)==4 and len(summary['cases'])==4
            manifest=json.loads((root/'PRIVATE_TRAIN_MANIFEST.json').read_text());definitions=[]
            for d,man,(_,records) in zip(summary['cases'],manifest,sorted(groups.items())):
                records=sorted(records,key=lambda r:r['index']);r=records[0]
                assert d['case']==man['case'] and str(r['image'])==man['image'] and str(r['label'])==man['label']
                assert len(records)==len(d['strata'])==len(d['areas'])==len(d['exposures'])
                definitions.append(dict(**d,record=r))
            assert sum(sum(d['exposures']) for d in definitions)==800
            unused={n:digest(getattr(s.phi,n).state_dict()) for n in ('s_head','b_head')}
            write(root/'REPAIR_RECORD.json',dict(status='BOUNDED_CONTINUATION',failure='compared runtime definitions list with serialized summary dict',original_failure_preflight_reverses=3,original_failure_fixture_reverses=1,repair_model_forwards=0,repair_model_reverses=0,repair_Adam=0,original_start=json.loads((root/'LAUNCH.json').read_text())['started'],original_deadline=json.loads((root/'LAUNCH.json').read_text())['deadline'],source_payload_unchanged=True,train_definition_equal=True))
        status('TRAIN',arm='C2')
        run_stage(s,base,ledger,counter,'C2',800,1800,fingerprint(p),data_seed=263000,checkpoint_stride=200,objective_factory=lambda state,ref:Objective(state,ref,True))
        assert digest(s.memories['RUNMC'].state_dict())==match['old_memory'] and all(digest(getattr(s.phi,n).state_dict())==v for n,v in unused.items())
        status('TRAIN_DIAGNOSTIC',arm='C2')
        with ledger.attempt('train_diagnostic_C2',seconds=1200,updates=0) as out:train2=train_evaluate(s,base,definitions,out/'train')
        cells={}
        for domain in ('RUNMC','BMC'):
            status('EVALUATE',arm='C2',domain=domain)
            with ledger.attempt('eval_C2_'+domain,seconds=600,updates=0) as out:cells[domain]=evaluate(p,s,domain,domain,out/'validation')
        write(root/'RESULTS.json',dict(cells=cells,train_diagnostics=dict(C2=train2),formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,preflight_model_objective_forwards=3,repair_model_reverse_calls=0,train_diagnostic_semantic_sample_forwards=12864,control_train_readout_reused=True,old_memory_and_Sb_unchanged=True,common_entrance=True,counter=counter.export()))
        status('COMPLETE')
    except BaseException as exc:status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
