"""Matched CPU policy optimizers. Student tensors never enter this graph."""
import copy
import numpy as np
import torch
from torch import nn
from experiments.qprompt_rl_v1.v4_rl_block_control.controller import ActorCritic, normalize, advantages

class Policy(nn.Module):
    def __init__(self, state, critic=False):
        super().__init__()
        old=ActorCritic();old.load_state_dict(state)
        self.actor=old.actor
        self.critic=old.value if critic else None
    def forward(self,x):
        return self.actor(x), self.critic(x).squeeze(-1) if self.critic is not None else torch.zeros(x.shape[:-1])

def distribution(logits): return torch.distributions.Categorical(logits=logits)

def soften(model,x):
    with torch.no_grad():
        logits=model.actor(x);before=float(distribution(logits).entropy().mean());target=.8*np.log(3)
        alpha=1.
        if before<target:
            lo,hi=0.,1.
            for _ in range(64):
                mid=(lo+hi)/2
                if float(distribution(logits*mid).entropy().mean())>=target:lo=mid
                else:hi=mid
            alpha=lo
        assert alpha>0
        model.actor[-1].weight.mul_(alpha);model.actor[-1].bias.mul_(alpha)
        after=model.actor(x)
        assert torch.equal(logits.argmax(-1),after.argmax(-1))
        return dict(alpha=alpha,entropy_before=before,entropy_after=float(distribution(after).entropy().mean()),argmax_agreement=1.,target=target,iterations=64,reward_accesses=0)

def panel(model,x,initial,behavior=None):
    with torch.no_grad():
        p=distribution(model.actor(x));q=distribution(initial)
        top=p.probs.topk(2,-1).values
        out=dict(probabilities=p.probs.tolist(),entropy=float(p.entropy().mean()),kl_initial=float(torch.distributions.kl_divergence(q,p).mean()),argmax_agreement_initial=float((initial.argmax(-1)==p.probs.argmax(-1)).float().mean()),top2_margin=top[:,0].sub(top[:,1]).tolist())
        if behavior is not None:out['kl_behavior']=float(torch.distributions.kl_divergence(distribution(behavior),p).mean())
        return out

def group_advantage(rewards,kind,scale,permutation=None):
    r=np.asarray(rewards,dtype=np.float64)
    if permutation is not None:r=r[permutation]
    return (r-r.mean())/(r.std(ddof=0)+1e-8 if kind=='GRPO_STD' else scale)

def optimizers(model,lr=.0003):
    return torch.optim.Adam(model.actor.parameters(),lr=lr), (torch.optim.Adam(model.critic.parameters(),lr=lr) if model.critic is not None else None)

def update(model,opts,trajectories,terminal,kind,scale,seed,step,clip_low=.2,clip_high=.2):
    assert len(trajectories)==4 and len({len(t) for t in trajectories})==1
    assert all(t and all(not r.get('reference',False) for r in t) for t in trajectories)
    flat=[r for t in trajectories for r in t]
    x=torch.stack([r['x'].detach() for r in flat]);a=torch.tensor([r['action'] for r in flat])
    old=torch.tensor([r['logp'] for r in flat]);oldp=torch.tensor([r['probs'] for r in flat])
    rng=np.random.RandomState(seed);permutation=None;targets=None
    if kind in ('PPO_MATCHED','BANDIT_MATCHED'):
        pairs=[advantages([r['reward']/scale for r in t],[r['value'] for r in t],kind=='BANDIT_MATCHED') for t in trajectories]
        adv=np.concatenate([z[0] for z in pairs]);targets=torch.tensor(np.concatenate([z[1] for z in pairs]),dtype=torch.float32)
        if kind=='PPO_MATCHED':adv=(adv-adv.mean())/max(adv.std(ddof=0),1e-8)
    else:
        if kind=='GRPO_FS_SHUFFLE':permutation=rng.permutation(4)
        adv=np.repeat(group_advantage(terminal,kind,scale,permutation),len(trajectories[0]))
    adv=torch.tensor(adv,dtype=torch.float32)
    assert not any(v.requires_grad for v in (x,old,oldp,adv))
    records=[];n=len(flat);assert n%4==0
    for epoch in range(4):
        for batch,indices in enumerate(np.array_split(rng.permutation(n),4)):
            logits,values=model(x[indices]);dist=distribution(logits)
            ratio=(dist.log_prob(a[indices])-old[indices]).exp();aa=adv[indices]
            raw=ratio*aa;clipped=ratio.clamp(1-clip_low,1+clip_high)*aa
            actor_loss=-torch.minimum(raw,clipped).mean();entropy=dist.entropy().mean()
            loss=actor_loss-.01*entropy
            assert torch.isfinite(loss)
            opts[0].zero_grad(set_to_none=True);loss.backward()
            norm=float(nn.utils.clip_grad_norm_(model.actor.parameters(),1.,error_if_nonfinite=True))
            step(opts[0],'actor',epoch*4+batch)
            with torch.no_grad():
                pos=aa>0;neg=aa<0
                stats=dict(epoch=epoch,batch=batch,actor_loss=float(actor_loss.detach()),entropy=float(entropy.detach()),actor_raw_grad_norm=norm,actor_clip_coefficient=min(1.,1/(norm+1e-6)),ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),raw_outside_fraction=float(((ratio<1-clip_low)|(ratio>1+clip_high)).float().mean()),positive_count=int(pos.sum()),positive_active_count=int((pos&(ratio>1+clip_high)).sum()),negative_count=int(neg.sum()),negative_active_count=int((neg&(ratio<1-clip_low)).sum()),kl_behavior=float(torch.distributions.kl_divergence(torch.distributions.Categorical(probs=oldp[indices]),dist).mean()),critic_loss=None,critic_raw_grad_norm=None,critic_clip_coefficient=None,explained_variance=None,return_mse=None)
            if targets is not None:
                target=targets[indices];value_loss=.5*(values-target).square().mean()
                assert torch.isfinite(value_loss)
                opts[1].zero_grad(set_to_none=True);value_loss.backward()
                normv=float(nn.utils.clip_grad_norm_(model.critic.parameters(),1.,error_if_nonfinite=True));step(opts[1],'critic',epoch*4+batch)
                error=(target-values).detach();variance=float(target.var(unbiased=False))
                stats.update(critic_loss=float(value_loss.detach()),critic_raw_grad_norm=normv,critic_clip_coefficient=min(1.,1/(normv+1e-6)),return_mse=float(error.square().mean()),explained_variance=1-float(error.var(unbiased=False))/variance if variance>0 else None)
            records.append(stats)
    return dict(updates=records,permutation=permutation.tolist() if permutation is not None else None,advantage_mean=float(adv.mean()),advantage_std=float(adv.std(unbiased=False)),loss_denominator=n,positive_active_fraction=sum(r['positive_active_count'] for r in records)/max(1,sum(r['positive_count'] for r in records)))

def warmup(model,trajectories,kind,scale,step):
    x=torch.stack([r['x'] for t in trajectories for r in t]).detach()
    values=[]
    for t in trajectories:
        r=np.array([r['reward']/scale for r in t]);values.extend(r if kind=='BANDIT_MATCHED' else np.cumsum(r[::-1])[::-1].copy())
    targets=torch.tensor(values,dtype=torch.float32);opt=torch.optim.Adam(model.critic.parameters(),lr=.001)
    actor=copy.deepcopy(model.actor.state_dict());diag=[]
    for i in range(64):
        pred=model.critic(x).squeeze(-1);loss=.5*(pred-targets).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();norm=float(nn.utils.clip_grad_norm_(model.critic.parameters(),1.,error_if_nonfinite=True));step(opt,i)
        diag.append(dict(loss=float(loss.detach()),grad_norm=norm))
    assert all(torch.equal(v,actor[k]) for k,v in model.actor.state_dict().items())
    return diag
