"""Zero-optimizer mathematical and transaction-schema regressions."""
import copy,json,tempfile,os,time
from pathlib import Path
import torch
from .method import *
from .data import make_schedule,UImages
from .runtime import PLAN,CAPS,TASKS
from r1_12h.core import atomic,events


def main():
    torch.set_num_threads(2);torch.manual_seed(261)
    p=torch.tensor([[[[.4]],[[.2]],[[.4]]]],requires_grad=True);q=torch.tensor([[[[.1]],[[.5]],[[.4]]]]);valid=torch.ones(1,1,1,dtype=torch.bool)
    lc,coverage=u_loss(p,q,valid,1);expected=.1*math.log(.1/.4)+.9*math.log(.9/.6);assert abs(float(lc.detach())-expected)<1e-6 and coverage=={'1':1.,'2':0.}
    assert float(u_loss(p,q,valid,2)[0])==0
    zero=u_loss(p,q,~valid,1)[0];zero.backward();assert p.grad.abs().sum()==0
    y=torch.zeros(1,1,1,dtype=torch.long);score=quality(p,y);assert score is not None and float(quality(q,y)-score)<0;assert quality(p,torch.full_like(y,255)) is None
    z,flags=state_vector(q,p,valid,torch.tensor(2.),0,1.);assert z.shape==(16,) and not z.requires_grad and not flags['conditional_unsupported']
    for arm in ('RL','NC_RL','REG'):
        policy=Controller(arm);ref=copy.deepcopy(policy);behavior=policy.distribution(z).detach().clone();actions=torch.tensor([2,2,2,2]);rewards=torch.full((4,),-.1)
        loss,diag=controller_loss(policy,ref,z,actions,rewards,behavior,.1);loss.backward()
        if arm!='REG':
            gradient=policy.logits.grad if arm=='NC_RL' else policy.net[-1].bias.grad;assert gradient[2]>0 and diag['kind']=='POLICY'
        else:assert diag['kind']=='REG'
        assert all(a is not b for a,b in zip(policy.parameters(),ref.parameters()))
        assert torch.allclose(behavior,torch.ones(3)/3)
        assert torch.equal(behavior,behavior.clone())
        # Actual action sealed before new reward; future distribution cannot overwrite it.
        sealed=draw(behavior,123)[0]
        with torch.no_grad():
            if arm=='NC_RL':policy.logits.add_(torch.tensor([9.,-9.,0.]))
            else:policy.net[-1].bias.add_(torch.tensor([9.,-9.,0.]))
        assert sealed==draw(behavior,123)[0] and not torch.equal(policy.distribution(z),behavior)
    param=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([param],lr=.000123);sch=LocalSchedule(opt)
    for t in (0,999,1000,1199):sch.t=t;sch.apply();assert abs(opt.param_groups[0]['lr']-.000123*(1-t/1200)**.9)<1e-12 and opt.param_groups[0]['lr']>0
    for domain,nl,nu in [('RIM_ONE_r3',16,63),('Drishti_GS',10,41)]:
        a=make_schedule(261,domain,nl,nu);assert a==make_schedule(261,domain,nl,nu) and a!=make_schedule(262,domain,nl,nu)
        for row in a['steps']:assert not(set(row['indices'])&set(row['online']) or set(row['fit'])&set(row['audit']))
    assert len(PLAN['R3b_jobs'])==108 and sum(x['planned_physical_student_calls'] for x in PLAN['R3b_jobs'])==144000
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.json'
        for step in (0,1,19,20,21,99,100,101,1199,1200):
            atomic(path,dict(complete=True,meta=dict(step=step),losses={'loss':1},runtime={'loss':2}));assert json.loads(path.read_text())['meta']['step']==step
        atomic(Path(tmp)/'U_IMAGE_ONLY.private.json',dict(authorization='R3_U_IMAGE_ONLY',image_only=True,files=[dict(domain='RIM_ONE_r3',label='forbidden')]))
        try:UImages(tmp,'RIM_ONE_r3')
        except PermissionError:pass
        else:raise AssertionError('U label field accepted')
    from .runtime import broker,rpc,process_identity,save
    from .runner import Worker
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);atomic(root/'SESSION.json',dict(deadline=time.time()+60));server,budget=broker(root)
        rpc(root,action='claim',task='mock',token='one',pid=os.getpid(),identity=process_identity(os.getpid()))
        try:rpc(root,action='claim',task='mock',token='two',pid=os.getpid(),identity=process_identity(os.getpid()))
        except RuntimeError:pass
        else:raise AssertionError('duplicate lease')
        rpc(root,action='attempt',task='mock',token='one',kind='synthetic_controller',transaction='mock-failure')
        assert budget.state['synthetic_controller']==1
        atomic(root/'SESSION.json',dict(deadline=0))
        try:rpc(root,action='attempt',task='mock',token='one',kind='synthetic_controller',transaction='expired')
        except RuntimeError:pass
        else:raise AssertionError('deadline bypass')
        server.shutdown();server.server_close()
        w=Worker.__new__(Worker);w.model=torch.nn.Linear(2,2);w.teacher=copy.deepcopy(w.model);w.opt=torch.optim.AdamW(w.model.parameters(),lr=.0001);w.sch=LocalSchedule(w.opt);w.policy=Controller('RL');w.reference=copy.deepcopy(w.policy);w.popt=torch.optim.Adam(w.policy.parameters());w.meta=dict(code_commit='mock',config_sha='mock');w.source_scheduler={'last_epoch':2000};w.elapsed=0.
        # Adam CPU step tensors must never alias a reusable rollback snapshot. No optimizer call.
        for param in w.model.parameters():w.opt.state[param]=dict(step=torch.tensor(2000.),exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
        frozen=w.snapshot();w.restore(frozen)
        for state in w.opt.state.values():state['step'].add_(1);state['exp_avg'].add_(1)
        assert all(int(state['step'])==2000 and not state['exp_avg'].any() for state in frozen['optimizer']['state'].values())
        w.restore(frozen)
        for t in (0,1,19,20,21,99,100,101,1199,1200):
            w.t=t;w.retained=t;w.sch.t=t;w.sch.apply();state=w.snapshot();folder=root/'mock-state';folder.mkdir(exist_ok=True);save(folder,state)
            with torch.no_grad():w.model.weight.add_(1);w.policy.net[-1].bias.add_(2)
            w.restore(torch.load(folder/'latest.pt',weights_only=False))
            assert w.t==t and w.sch.t==t and torch.equal(w.model.weight,state['student']['weight']) and torch.equal(w.policy.net[-1].bias,state['policy']['net.2.bias']) and torch.equal(torch.get_rng_state(),state['rng']['cpu'])
        from .report import aggregate
        queue={task:dict(status='PENDING') for task in TASKS}
        # Exercise the real collector while an incomplete publication is visible.
        task=next(t for t in TASKS if t.startswith('R3B'));atomic(root/'tasks'/task/'EVALUATION.json',dict(complete=False))
        aggregate(root,queue,{},False);assert json.loads((root/'reports/PRIMARY_DECISION.json').read_text())['endpoints']==0
    print(json.dumps(dict(status='PASSED',optimizer_calls=0,schedule_cells=6,fold_disjoint=True,anchored_negative_credit=True,sealed_action=True,mixed_distribution=True,coarse_contract=True,hidden_GT_rejected=True)))

if __name__=='__main__':main()
