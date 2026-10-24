"""Two new zero-update readouts after the frozen parent has completed."""
import copy,json,os,subprocess,time
from pathlib import Path
import torch
from ..stream_state import StreamState,digest
from ..multiscale_recognition_v46.model import MultiscaleState
from ..paired_semantic_v19.run import evaluate
from ..paired_semantic_v19.method import predict
from ..budgets import ResourceLedger
from ..real_diagnostic import write

CLASSES={'S':StreamState,'W':MultiscaleState}


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='ENTRY_DECOMPOSITION_V47'
    parent=Path(p['v46_root']);launch=json.loads((root/'LAUNCH.json').read_text())
    def status(stage,**kw):
        tmp=root/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)));tmp.replace(root/'STATUS.json')
    try:
        status('WAITING_FOR_V46')
        while True:
            ex=parent/'EXIT.json';st=json.loads((parent/'STATUS.json').read_text())
            if ex.exists():
                assert json.loads(ex.read_text())['exit_code']==0,'parent failed; no new inference'
                assert st['stage']=='COMPLETE';break
            assert time.time()<p['dependency_deadline'],'parent original deadline exhausted'
            time.sleep(15)
        r=json.loads((parent/'RESULTS.json').read_text());pl=json.loads((parent/'results/RESOURCE_LEDGER.json').read_text())
        assert r['new_formal_updates']==4800 and r['counter']['outer_backward_calls']==4800
        assert len(r['cells'])==8 and all(v['complete'] and v['patients']==4 for v in r['cells'].values())
        assert len(pl['training_sessions'])==4 and pl['updates_reserved']==4800
        assert all(v['status']=='PASS' for v in pl['attempts'])
        assert all(v['status']=='COMPLETE' and v['completed_steps']==(1600 if k.endswith('source') else 800) for k,v in pl['training_sessions'].items())
        status('WAITING_FOR_GPU')
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>3000:break
            assert time.time()<launch['deadline']-1800,'no remaining inference admission window'
            time.sleep(15)
        write(root/'DEPENDENCY_READY.json',dict(parent_complete=True,parent_formal_updates=4800,parent_cells=8,GPU_free_MiB=free,compute_started=time.time(),original_deadline=launch['deadline']))
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ledger=ResourceLedger(root/'results',max_seconds=1800,max_updates=0,max_bytes=2_000_000_000)
        cells={};checks={}
        for arm,cls in CLASSES.items():
            status('CHECK_ENTRY',arm=arm)
            with ledger.attempt('check_'+arm,seconds=180,updates=0):
                source=cls.load(parent/'results'/f'{arm}_source_attempt001/PRIVATE_SEALED_STATE.pt')
                entry=cls.load(parent/'results'/f'{arm}_target_attempt001/ACTIVE_INITIAL.pt')
                assert entry.steps==0 and entry.active['id']=='BMC' and not entry.optimizer.state
                for name in ('q','phi','model','projection'):
                    assert digest(getattr(source,name).state_dict())==digest(getattr(entry,name).state_dict()),name
                assert digest(source.memories['RUNMC'].state_dict())==digest(entry.memories['RUNMC'].state_dict())
                assert digest(source.q.state_dict())==digest(entry.prior.state_dict())
                assert not entry.current.m.count_nonzero() and not entry.current.C.count_nonzero()
                torch.testing.assert_close(entry.current.sd,torch.full_like(entry.current.sd,.1),rtol=1e-12,atol=1e-12)
                source.to_device('cuda');entry.to_device('cuda')
                x=torch.linspace(0,1,entry.config.height*entry.config.width,device='cuda',dtype=torch.float64).reshape(1,1,entry.config.height,entry.config.width).expand(1,entry.config.channels,-1,-1)
                before=digest(entry.payload());source_before=digest(source.payload())
                a=predict(x,source,'RUNMC',47001,outer=2,local=2);b=predict(x,entry,'RUNMC',47001,outer=2,local=2)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                assert digest(source.payload())==source_before and digest(entry.payload())==before
                del source,x,a,b
                checks[arm]=dict(source_parameters_and_memory_exact=True,source_q_equals_target_prior=True,real_active_initial=True,fresh_optimizer=True,reset_domain_parameters=True,old_memory_routing_exact=True,probe_semantic_forwards=8)
                write(root/'MATCHING.json',dict(status='IN_PROGRESS',arms=checks))
            entry.memories['BMC']=copy.deepcopy(entry.current).requires_grad_(False)
            probe=digest(entry.payload());status('EVALUATE_ZERO',arm=arm)
            with ledger.attempt('eval_'+arm+'_zero_BMC',seconds=600,updates=0) as out:
                cells[arm]=evaluate(p,entry,'BMC','BMC',out/'validation')
            assert digest(entry.payload())==probe
            del entry.memories['BMC'];assert digest(entry.payload())==before
            del entry
        write(root/'MATCHING.json',dict(status='PASS',arms=checks,source_updates=0,optimizer_updates=0,reverse_calls=0,probe_semantic_forwards=16))
        write(root/'RESULTS.json',dict(cells=cells,new_evaluation_cells=2,reused_endpoint_cells=4,formal_updates=0,optimizer_updates=0,reverse_calls=0,source_state_unchanged=True))
        status('COMPLETE')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise

if __name__=='__main__':main()
