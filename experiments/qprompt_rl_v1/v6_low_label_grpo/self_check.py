"""Small CPU check for changed group/minibatch controls and matched exposure."""
import copy
import torch
from experiments.qprompt_rl_v1.v5_grpo_group_control import controller as q


def controller_check():
    torch.set_num_threads(2);torch.manual_seed(6301)
    base=q.ActorCritic().state_dict();out=[]
    for g in (2,4,8):
        model=q.Policy(base);opts=q.optimizers(model);calls=[];samples=0
        for group in range(8//g):
            traj=[]
            for i in range(g):
                rows=[]
                for t in range(4):
                    x=torch.randn(40);a=i%3
                    with torch.no_grad():dist=q.distribution(model.actor(x))
                    rows.append(dict(x=x,action=a,logp=float(dist.log_prob(torch.tensor(a))),probs=dist.probs.tolist(),value=0.,reward=float(i),reference=False))
                traj.append(rows)
            def step(opt,kind,k):calls.append((kind,group,k));opt.step()
            diag=q.update(model,opts,traj,list(range(g)),'GRPO_FS',1.,700+group,step,group_size=g,minibatches=g)
            assert len(diag['updates'])==4*g
            assert any(r['actor_raw_grad_norm']>0 for r in diag['updates'])
            samples+=diag['loss_denominator']*4
        assert len(calls)==32 and len(set(calls))==32 and samples==128
        assert any(not torch.equal(v,base['actor.'+k]) for k,v in model.actor.state_dict().items())
        out.append(dict(group_size=g,groups=8//g,actor_calls=len(calls),sample_presentations=samples))
    return dict(status='PASS',variants=out,synthetic_actor_calls=96,student_calls=0)

if __name__=='__main__':print(controller_check())
