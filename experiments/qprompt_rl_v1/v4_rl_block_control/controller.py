"""Small matched policies, frozen normalization and episodic PPO/GAE."""
import copy
import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(40, 32), nn.Tanh(), nn.Linear(32, 3))
        self.value = nn.Sequential(nn.Linear(40, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def forward(self, x):
        return self.actor(x), self.value(x).squeeze(-1)


def distribution(logits):
    # The same mixture is used for sampling, saved behavior probabilities and PPO ratios.
    return torch.distributions.Categorical(probs=.9 * logits.softmax(-1) + .1 / 3)


def scaler(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    return dict(mean=x.mean(0).tolist(), sd=np.maximum(sd, 1e-8).tolist(),
                constant=(sd < 1e-8).tolist(), minimum=x.min(0).tolist(), maximum=x.max(0).tolist())


def normalize(x, scale):
    x = np.asarray(x, dtype=np.float64)
    z = (x - scale['mean']) / scale['sd']
    z[..., scale['constant']] = 0
    assert np.isfinite(z).all()
    # Fixed clipping guards extrapolation; out-of-range counts remain in the report.
    return torch.tensor(np.clip(z, -10, 10), dtype=torch.float32)


def offline(x, rewards, seed, step):
    torch.manual_seed(seed)
    model = ActorCritic()
    opt = torch.optim.Adam(model.actor.parameters(), lr=.003)
    reward = torch.tensor(rewards, dtype=torch.float32)
    for i in range(256):
        logits, _ = model(x)
        logp = logits.log_softmax(-1)
        p = logp.exp()
        loss = -(p * reward).sum(-1).mean() + .001 * (p * (logp + np.log(3))).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 1., error_if_nonfinite=True)
        step(opt, i)
    return model


def advantages(rewards, values, bandit=False):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if bandit:
        return rewards - values, rewards
    # gamma=1 makes unscaled quality increments telescope to terminal quality.
    out = np.zeros_like(rewards)
    carry = 0.
    for i in reversed(range(len(rewards))):
        following = values[i + 1] if i + 1 < len(values) else 0.
        carry = rewards[i] + following - values[i] + .95 * carry
        out[i] = carry
    return out, out + values


def update(model, opt, trajectory, kind, seed, step):
    x = torch.stack([r['x'] for r in trajectory])
    actions = torch.tensor([r['action'] for r in trajectory], dtype=torch.long)
    old = torch.tensor([r['logp'] for r in trajectory], dtype=torch.float32)
    rewards = np.asarray([r['reward'] for r in trajectory])
    if kind == 'REWARD_SHUFFLE':
        rewards = rewards[np.random.RandomState(seed).permutation(len(rewards))]
    adv, returns = advantages(rewards, [r['value'] for r in trajectory], kind == 'BANDIT_25')
    adv = torch.tensor((adv - adv.mean()) / max(adv.std(), 1e-8), dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    diagnostics = []
    for epoch in range(4):
        logits, values = model(x)
        dist = distribution(logits)
        ratio = (dist.log_prob(actions) - old).exp()
        loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
        loss = loss + .5 * (values - returns).square().mean() - .01 * dist.entropy().mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        step(opt, epoch)
        diagnostics.append(dict(loss=float(loss.detach()), entropy=float(dist.entropy().mean().detach()),
                                clip_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean().detach())))
    return diagnostics


def self_check():
    # Known terminal boundary and bandit target; checkpointed actor/critic/Adam continuation.
    a, r = advantages([1., 2.], [0., 0.])
    assert np.allclose(a, [2.9, 2.]) and np.array_equal(a, r)
    a, r = advantages([1., 2.], [.4, .8], True)
    assert np.allclose(a, [.6, 1.2]) and np.array_equal(r, [1., 2.])
    torch.manual_seed(4)
    x = torch.randn(60, 40)
    target = (x[:, 0] > 0).long()
    reward = torch.nn.functional.one_hot(target, 3).float().numpy()
    model = offline(x, reward, 8, lambda opt, _: opt.step())
    assert (model(x)[0].argmax(-1) == target).float().mean() > .95
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    traj = []
    for z in x[:8]:
        logits, value = model(z)
        dist = distribution(logits)
        action = int(dist.sample())
        traj.append(dict(x=z, action=action, logp=float(dist.log_prob(torch.tensor(action)).detach()),
                         reward=float(action == 0), value=float(value.detach())))
    update(model, opt, traj, 'PPO_25', 10, lambda o, _: o.step())
    state, optim = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    update(model, opt, traj, 'PPO_25', 10, lambda o, _: o.step())
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(state); opt.load_state_dict(optim)
    update(model, opt, traj, 'PPO_25', 10, lambda o, _: o.step())
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    return dict(status='PASS', synthetic_only=True, gae_boundary=True, synthetic_policy_fit=True,
                actor_critic_adam_resume_exact=True, synthetic_controller_calls=268)
