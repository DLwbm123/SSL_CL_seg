"""P0 then P1, reusing the frozen training engine and patient-level evaluation."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from . import diagnostic
from .method import Objective, components, consistency, posterior_probability
from ..stream_state import StreamState, digest
from ..budgets import ResourceLedger, Counters
from ..qualification import fingerprint
from ..real_diagnostic import write
from ..training import run_stage
from ..paired_semantic_v19.run import data, evaluate


def make_branch(path, p):
    s = StreamState.load(path)
    donor = digest(s.payload())
    d = data(p, s.config, 'RUNMC', 'LU')
    # This is a preregistered fork, not a resume of the v19 training session.
    s.active['evidence'] = copy.deepcopy(d.evidence)
    s.runtime = None
    assert s.steps == 400 and s.active['id'] == 'RUNMC'
    return s.to_device('cuda'), d, donor


def preflight(path, p, out):
    s,d,donor = make_branch(path,p)
    before = digest(s.payload())
    L,U = diagnostic.batch(d,'L',163800),diagnostic.batch(d,'U',163801)
    checks = {}
    # Loss components shared across branches must be exact regardless of U execution.
    with torch.no_grad():
        a = components(s,L,None,d.counts)
        b = components(s,L,U,d.counts)
        for key in ('task','image_L','kl'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        prob=posterior_probability(s,L['x'])
        torch.testing.assert_close(prob.sum(1),torch.ones_like(prob[:,0]))
    checks['paired_L_global_draws'] = 'PASS'
    teacher=torch.tensor([[[[.8,.2]],[[.2,.8]]]],dtype=torch.float64,requires_grad=True)
    student=torch.tensor([[[[.7,.3]],[[.3,.7]]]],dtype=torch.float64,requires_grad=True)
    loss,_=consistency(teacher,student);loss.backward()
    assert teacher.grad is None and student.grad.abs().sum()>0
    checks['stopped_teacher_and_student_gradient'] = 'PASS'
    elapsed={}
    for name, image, sem in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
        s.optimizer.zero_grad(set_to_none=True)
        start=time.monotonic()
        loss,parts=Objective(s,None,image,sem).estimate(L,U,d.counts,s.generator)
        loss.backward();torch.cuda.synchronize()
        elapsed[name]=time.monotonic()-start
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in diagnostic.parameters(s))
        assert s.q.raw_scale.grad.abs().sum()>0
        if sem: assert float(parts['semantic_U']) >= -1e-12
    s.optimizer.zero_grad(set_to_none=True)
    assert digest(s.payload())==before
    checkpoint=out/'PRIVATE_CLONE_CHECK.pt';s.save(checkpoint)
    restored=StreamState.load(checkpoint).to_device('cuda')
    assert digest(restored.payload())==before, 'clone checkpoint lost Adam/prior/RNG'
    prior=digest(s.prior.state_dict())
    for _ in range(3):
        clone,_,other=make_branch(path,p)
        assert other==donor and digest(clone.payload())==before
        assert digest(clone.prior.state_dict())==prior
        del clone
    checks.update(all_arm_finite_gradients='PASS',common_payload_and_optimizer='PASS',original_prior_preserved='PASS',checkpoint_roundtrip='PASS',real_optimizer_updates=0,reverse_calls=5,step_seconds=elapsed)
    if 800*sum(elapsed.values())+5000>14000:
        raise RuntimeError('compute admission failed')
    return checks


def main():
    root=Path(os.environ['RUN_ROOT']);p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    assert p['purpose']=='SEMANTIC_U_V20_P0_P1' and p['max_training_updates']==1600
    source=Path(p['donor_root']);path=source/'results/s20261023_L_A_attempt001/ACTIVE_STEP_000400.pt'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=ResourceLedger(root/'results',max_seconds=14400,max_updates=1600,max_bytes=5_000_000_000)
    counter=Counters()
    def replace_json(path, value):
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(path)
    def status(stage,**extra):
        replace_json(root/'STATUS.json',dict(stage=stage,time=time.time(),**extra))
    try:
        status('P0_DIAGNOSTIC')
        with ledger.attempt(f"P0_attempt{len(ledger.data['attempts'])+1:03}",seconds=2400,updates=0):
            result=diagnostic.run(p,source,lambda v:replace_json(root/'P0_PROGRESS.json',v))
            write(root/'P0_RESULTS.json',result)
        status('PREFLIGHT')
        with ledger.attempt('preflight',seconds=600,updates=0) as out:
            checks=preflight(path,p,out);write(root/'PREFLIGHT.json',checks)
        results={}
        for arm,image,sem in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
            status('P1_TRAIN',arm=arm)
            s,d,donor=make_branch(path,p)
            write(root/f'BRANCH_{arm}.json',dict(donor_payload=donor,initial_payload=digest(s.payload()),initial_step=s.steps,prior_hash=s.prior_hash,optimizer_inherited=True))
            run_stage(s,d,ledger,counter,f'P1_{arm}',800,1800,fingerprint(p),data_seed=163000,checkpoint_stride=400,objective_factory=lambda state,refiner:Objective(state,refiner,image,sem))
            for domain in ('RUNMC','BMC'):
                status('P1_EVALUATE',arm=arm,domain=domain)
                with ledger.attempt(f'eval_{arm}_{domain}',seconds=600,updates=0) as out:
                    results[f'{arm}_{domain}']=evaluate(p,s,domain,'RUNMC',out/'validation')
            write(root/f'ARM_{arm}.json',{k:v for k,v in results.items() if k.startswith(arm+'_')})
            del s,d
        write(root/'RESULTS.json',dict(cells=results,formal_training_updates=1600,diagnostic_reverse_calls=result['reverse_calls'],diagnostic_candidate_adam_steps=result['candidate_adam_steps_discarded'],preflight_reverse_calls=checks['reverse_calls']))
        status('COMPLETE',scientific_claim='requires_analysis',P2='separate_admission_required')
    except BaseException as exc:
        status('FAILED',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
