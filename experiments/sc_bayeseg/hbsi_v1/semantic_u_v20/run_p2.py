"""Fixed P2 continuation; does not restart or replay P0/P1 or the donor queue."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from .method import Objective
from ..stream_state import StreamState, digest
from ..budgets import ResourceLedger, Counters, BudgetExceeded
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..training import run_stage
from ..paired_semantic_v19.run import data, evaluate


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='SEMANTIC_U_V20_P2' and p['max_training_updates']==12800
    prior_results=json.loads((Path(p['p1_root'])/'RESULTS.json').read_text())
    assert json.loads((Path(p['p1_root'])/'EXIT.json').read_text())['exit_code']==0
    assert all(c['complete'] and all(v['empty_predictions']==0 for v in c['repeats']) for c in prior_results['cells'].values())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=30000,max_updates=12800,max_bytes=5_000_000_000)
    counter=Counters()
    def status(stage,**extra):
        tmp=root/'STATUS.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**extra),indent=2));tmp.replace(root/'STATUS.json')
    def check_deadline():
        if time.time()>=p['absolute_deadline']:raise BudgetExceeded('original combined P0/P1/P2 deadline')
    results={}
    try:
        for seed in (20261023,20261024,20261025):
            start_step=800 if seed==20261023 else 400
            common=None
            for arm,image,sem in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
                if seed==20261023:
                    path=Path(p['p1_root'])/f'results/P1_{arm}_attempt001/ACTIVE_STEP_000800.pt'
                else:
                    path=Path(p['donor_root'])/f'results/s{seed}_L_A_attempt001/ACTIVE_STEP_000400.pt'
                if not path.exists():
                    status('WAIT_DONOR',seed=seed,arm=arm)
                    while not path.exists():
                        check_deadline();ledger.check();time.sleep(30)
                check_deadline()
                s=StreamState.load(path)
                assert s.steps==start_step and s.config.seed==seed and s.active['id']=='RUNMC'
                donor_digest=digest(s.payload());prior_before=digest(s.prior.state_dict())
                d=data(p,s.config,'RUNMC','LU');s.active['evidence']=copy.deepcopy(d.evidence);s.runtime=None
                s.to_device('cuda')
                initial=digest(s.payload())
                if seed!=20261023:
                    if common is None:common=initial
                    else:assert initial==common, 'different common entrance'
                assert digest(s.prior.state_dict())==prior_before
                name=f's{seed}_{arm}'
                write(root/f'BRANCH_{name}.json',dict(seed=seed,arm=arm,donor_payload=donor_digest,initial_payload=initial,initial_step=start_step,prior_hash=prior_before,optimizer_inherited=True))
                status('P2_TRAIN',seed=seed,arm=arm)
                run_stage(s,d,ledger,counter,name,1600,min(1800,p['absolute_deadline']-time.time()),fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=lambda state,refiner:Objective(state,refiner,image,sem))
                for domain in ('RUNMC','BMC'):
                    check_deadline();status('P2_EVALUATE',seed=seed,arm=arm,domain=domain)
                    with ledger.attempt(f'eval_{name}_{domain}',seconds=min(600,p['absolute_deadline']-time.time()),updates=0) as out:
                        results[f'{name}_{domain}']=evaluate(p,s,domain,'RUNMC',out/'validation')
                write(root/f'ARM_{name}.json',{k:v for k,v in results.items() if k.startswith(name+'_')})
                del s,d
        write(root/'RESULTS.json',dict(cells=results,new_formal_updates=12800,P1_updates=1600,combined_new_updates=14400,original_combined_start=p['original_combined_start'],absolute_deadline=p['absolute_deadline']))
        status('COMPLETE',scientific_claim='requires_analysis')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
