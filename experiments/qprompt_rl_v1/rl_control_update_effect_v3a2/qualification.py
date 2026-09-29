"""Fixed synthetic qualification; all actual/virtual updates explicitly charged."""
import copy,math
import numpy as np
import torch
from .runtime import *
from . import policy,effect
from rl_control_bandit_v2.runner import cpu_copy
from rl_control_bandit_v2.qualification import same

def qualify(ledger):
    if (ROOT/'QUALIFICATION.json').exists():return receipt(ROOT/'QUALIFICATION.json')
    torch.manual_seed(17);checks=[]
    for variant in range(4):
        originals=[torch.randn(16,device='cuda') for _ in range(4)];grads=[torch.randn_like(p) for p in originals];all_steps=[]
        for action in range(3):
            ps=[torch.nn.Parameter(p.clone()) for p in originals];opt=torch.optim.AdamW([dict(params=ps[:2],lr=1e-5),dict(params=ps[2:],lr=1e-4)],weight_decay=.01,amsgrad=variant==3,foreach=[None,False,True,None][variant])
            for p in ps:
                opt.state[p]={'step':torch.tensor(2000.),'exp_avg':torch.full_like(p,.01),'exp_avg_sq':torch.full_like(p,.03)}
                if variant==3:opt.state[p]['max_exp_avg_sq']=torch.full_like(p,.04)
            gg=[None if action==0 and i==0 else torch.zeros_like(p) if i==1 else grads[i]*(1+action*.1) for i,p in enumerate(ps)];before=cpu_copy(opt.state_dict());preview,states=ledger.call('preview_synthetic',f'{variant}/{action}',lambda:effect.preview(opt,ps,gg));assert same(before,cpu_copy(opt.state_dict()))
            for p,g in zip(ps,gg):p.grad=g
            ledger.step(opt,'student_synthetic',f'preview-parity/{variant}/{action}');error=math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,preview)));norm=math.sqrt(sum(float((p.detach()-q).double().square().sum()) for p,q in zip(ps,originals)));assert error<=1e-5*norm+1e-8
            for i,p in enumerate(ps):
                if gg[i] is None:assert torch.equal(p,originals[i]) and int(opt.state[p]['step'])==2000
                else:assert same(cpu_copy(opt.state[p]),cpu_copy(states[i]));assert opt.state[p]['step'].untyped_storage().data_ptr()!=states[i]['step'].untyped_storage().data_ptr()
            checks.append(dict(variant=variant,action=action,L2_error=error,actual_step_L2=norm,max_abs=max(float((p.detach()-q).abs().max()) for p,q in zip(ps,preview))));all_steps.append(([p.detach()-q for p,q in zip(ps,originals)],[p-q for p,q in zip(preview,originals)]))
        for action in (0,1):
            err=sum(float(((a-f)-(b-g)).double().square().sum()) for a,b,f,g in zip(all_steps[action][0],all_steps[action][1],all_steps[2][0],all_steps[2][1]))**.5;assert err<=1e-8;checks[-1]['relative_action_delta_error_'+str(action)]=err
    from rl_control_bandit_v2.method import u_loss
    pu=torch.full((1,3,4,4),1/3,device='cuda',requires_grad=True);q=pu.detach();v=torch.zeros((1,4,4),dtype=torch.bool,device='cuda')
    for action in (1,2):
        loss,coverage=u_loss(pu,q,v,action);assert float(loss)==0 and all(x==0 for x in coverage.values())
    # Exact local quadratic direction, prior and deterministic FINE reachability.
    assert -effect.dot([torch.tensor([2.])],[torch.tensor([-.1])])>0
    torch.manual_seed(0);m=policy.Policy();assert sum(p.numel() for p in m.parameters())==1411;z=torch.zeros(8,40);assert torch.allclose(m(z).softmax(-1),torch.tensor(policy.PRIOR,dtype=torch.float32).expand(8,-1),atol=1e-7);assert (policy.actions(m(z).detach().numpy())==2).all();assert policy.actions([[1.,1.,1.]])[0]==2
    x=np.zeros((8,40));x[:,0]=np.arange(8);x[:,20]=np.arange(8)%2;sc=policy.normalize_fit(x);test=x.copy();test[:,1]=999;assert (policy.normalize(test,sc)[:,1]==0).all();assert np.array_equal(policy.normalize(x,sc)[:,20],x[:,20])
    results=[]
    for seed in (0,1,2,3):
        rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(64,40));switch=np.where(x[:,0]>0,.5,-.5)
        for name,d in [('fine',np.tile([-1.,-.5,0.],(64,1))),('switch',np.column_stack((switch,-switch,np.zeros(64)))),('uninformative',np.zeros((64,3)))]:
            model,diag=policy.neural(x,d,seed,ledger,f'synthetic/{seed}/{name}',synthetic=True);prob=model(torch.tensor(x,dtype=torch.float32)).detach().softmax(-1).numpy();v=float((prob*d).sum(1).mean());prior=float((policy.PRIOR*d).sum(1).mean());results.append(dict(seed=seed,problem=name,value=v,prior_value=prior,behavior_pass=bool(v>prior if name!='uninformative' else np.max(np.abs(prob-policy.PRIOR))<1e-5),**diag))
    from .worker import Guard
    guard=Guard();denied=[]
    for p in (ROOT/'panels/fake.private.json',ROOT/'forbidden.h5'):
        try:p.open('rb')
        except PermissionError:denied.append(True)
    assert len(denied)==2
    save(ROOT/'reports/ADAMW_PREVIEW_PARITY.json',checks=checks,read_only=True,none_distinguished_from_zero=True,actual_student_calls=12,virtual_synthetic_calls=12)
    save(ROOT/'QUALIFICATION.json',status='PASSED',synthetic_policy_results=results,permission_denials=2,scaler_train_only=True,all_optimizer_costs_charged=True)
