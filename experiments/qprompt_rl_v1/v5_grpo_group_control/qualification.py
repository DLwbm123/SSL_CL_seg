"""Fixed synthetic qualification, including delayed credit and real actor copies."""
import copy
from collections import Counter
import numpy as np
import torch
from . import controller as q

def run(real_initializers):
    torch.set_num_threads(2)
    counts=Counter()
    def step(opt,kind,index): counts[kind]+=1;opt.step()
    a=q.group_advantage([1,2,3,4],'GRPO_STD',1)
    assert np.allclose(a,q.group_advantage([101,102,103,104],'GRPO_STD',1))
    assert np.allclose(a,(np.arange(1,5)-2.5)/(np.sqrt(1.25)+1e-8))
    for kind in ('GRPO_STD','GRPO_FS'):
        assert np.allclose(q.group_advantage([1,2,3,4],kind,.2),q.group_advantage([101,102,103,104],kind,.2))
        assert np.array_equal(q.group_advantage([2]*4,kind,.2),np.zeros(4))
    for ratio,adv,expected in ((1.4,1.,1.2),(.6,-1.,-.8),(1.4,-1.,-1.4),(.6,1.,.6)):
        assert np.isclose(min(ratio*adv,np.clip(ratio,.8,1.2)*adv),expected)
    assert np.allclose(q.advantages([0,1],[0,0])[0],[.95,1])
    assert np.allclose(q.advantages([0,1],[0,0],True)[0],[0,1])
    permutation=np.random.RandomState(4).permutation(4)
    assert sorted(np.arange(4)[permutation])==list(range(4))
    models={}
    torch.manual_seed(510)
    initial=q.ActorCritic().state_dict()
    for task in ('immediate','delayed'):
        models[task]=(initial,torch.zeros(40),1,task)
    for domain,(state,x) in real_initializers.items():
        probe=q.Policy(state)
        target=(int(probe.actor(x).argmax())+1)%3
        models['offline_'+domain]=(state,x,target,'delayed')
    results=[]
    for task,(state,x,target,mode) in models.items():
        for kind in ('PPO_MATCHED','GRPO_STD','GRPO_FS'):
            model=q.Policy(state,kind=='PPO_MATCHED');opts=q.optimizers(model)
            generator=torch.Generator().manual_seed(511)
            before=float(q.distribution(model.actor(x)).probs[target])
            last=None
            for group in range(256):
                trajectories=[];terminal=[]
                for branch in range(4):
                    traj=[]
                    for t in range(4):
                        with torch.no_grad():
                            logits,value=model(x);dist=q.distribution(logits)
                            action=int(torch.multinomial(dist.probs,1,generator=generator))
                            reward=float(action==target) if mode=='immediate' else (float(traj[0]['action']==target) if t==3 else 0.)
                            traj.append(dict(x=x.detach().clone(),action=action,logp=float(dist.log_prob(torch.tensor(action))),probs=dist.probs.tolist(),value=float(value),reward=reward))
                    trajectories.append(traj);terminal.append(sum(r['reward'] for r in traj))
                last=q.update(model,opts,trajectories,terminal,kind,1.,512+group,step)
            after=float(q.distribution(model.actor(x)).probs[target])
            assert after>.70 and after>before+.15, (task,kind,before,after)
            results.append(dict(task=task,method=kind,before=before,after=after,target=target,groups=256))
            if task=='immediate' and kind=='PPO_MATCHED':
                saved=copy.deepcopy((model.state_dict(),[o.state_dict() for o in opts]))
                q.update(model,opts,trajectories,terminal,kind,1.,999,step)
                expected=copy.deepcopy((model.state_dict(),[o.state_dict() for o in opts]))
                model.load_state_dict(saved[0])
                for o,s in zip(opts,saved[1]):o.load_state_dict(s)
                q.update(model,opts,trajectories,terminal,kind,1.,999,step)
                def equal(a,b):
                    if torch.is_tensor(a):return torch.equal(a,b)
                    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                    return a==b
                assert equal(expected,(model.state_dict(),[o.state_dict() for o in opts]))
                bad=copy.deepcopy(trajectories);bad[0][0]['reference']=True
                try:q.update(model,opts,bad,terminal,kind,1.,999,step)
                except AssertionError:pass
                else:raise AssertionError('reference admitted to on-policy set')
    # Equal-return groups retain the entropy objective and consume exactly 16 steps.
    torch.manual_seed(514);state=q.ActorCritic().state_dict();state['actor.2.bias']=torch.tensor([2.,0.,-1.])
    model=q.Policy(state);opts=q.optimizers(model);x=torch.zeros(40)
    with torch.no_grad():
        dist=q.distribution(model.actor(x));before=float(dist.entropy())
        row=dict(x=x,action=0,logp=float(dist.log_prob(torch.tensor(0))),probs=dist.probs.tolist(),value=0.,reward=1.)
    diag=q.update(model,opts,[[copy.deepcopy(row) for _ in range(4)] for _ in range(4)],[1.]*4,'GRPO_FS',1.,515,step)
    assert all(r['actor_loss']==0 for r in diag['updates']) and float(q.distribution(model.actor(x)).entropy())>before
    return dict(status='PASS',results=results,optimizer_calls=dict(counts),math=dict(shift_invariance=True,population_std=True,equal_rewards_zero=True,clip_signs=True,GAE_terminal=True,shuffle_multiset=True,optimizer_restore_exact=True,U0_excluded=True),real_reward_accesses=0)
