"""A small set-conditioned policy; ordered sampling, unordered training subsets."""
import numpy as np
import torch
from torch import nn

FEATURES = 12


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(FEATURES + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x, embeddings, chosen, k):
        novelty = (1 - (embeddings @ embeddings[chosen].T).max(1).values).clamp(0, 2) if chosen else torch.ones(len(x))
        context = torch.stack((novelty, torch.full_like(novelty, len(chosen) / k)), -1)
        logits = self.score(torch.cat((x, context), -1)).squeeze(-1)
        mask = torch.zeros(len(x), dtype=torch.bool)
        mask[chosen] = True
        return logits.masked_fill(mask, -torch.inf)


def sequence(model, x, embeddings, k, generator=None, actions=None, greedy=False):
    assert 1 <= k <= len(x)
    chosen, logp, entropy = [], [], []
    for j in range(k):
        logits = model(x, embeddings, chosen, k)
        dist = torch.distributions.Categorical(logits=logits)
        action = int(actions[j]) if actions is not None else int(logits.argmax()) if greedy else int(torch.multinomial(dist.probs, 1, generator=generator))
        assert action not in chosen and 0 <= action < len(x)
        logp.append(dist.log_prob(torch.tensor(action)))
        entropy.append(dist.entropy())
        chosen.append(action)
    return chosen, torch.stack(logp), torch.stack(entropy)


def update(model, optimizer, x, embeddings, samples, rewards, k, seed, step):
    assert len(samples) == 4 and len(rewards) == 4
    rewards = np.asarray(rewards, dtype=np.float64)
    assert np.isfinite(rewards).all()
    sd = float(rewards.std(ddof=0))
    advantage = (rewards - rewards.mean()) / max(sd, 1e-6)
    rng = np.random.RandomState(seed)
    stats = []
    for epoch in range(4):
        for ordinal, i in enumerate(rng.permutation(4)):
            actions, old = samples[i]
            _, logp, entropy = sequence(model, x, embeddings, k, actions=actions)
            ratio = (logp - old).exp()
            a = float(advantage[i])
            surrogate = torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a).mean()
            loss = -surrogate - .01 * entropy.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            step(optimizer, epoch * 4 + ordinal)
            stats.append(dict(loss=float(loss.detach()), entropy=float(entropy.mean().detach()),
                              grad_norm=float(norm), clipped_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean())))
    return dict(rewards=rewards.tolist(), reward_mean=float(rewards.mean()), reward_std=sd,
                zero_reward_dispersion=sd < 1e-6, advantages=advantage.tolist(), actor_updates=len(stats), updates=stats)


def self_check():
    torch.manual_seed(771)
    model = Policy(); opt = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.randn(9, FEATURES); emb = torch.nn.functional.normalize(torch.randn(9, 5), dim=1)
    g = torch.Generator().manual_seed(772)
    samples = []
    for _ in range(4):
        with torch.no_grad(): chosen, logp, _ = sequence(model, x, emb, 4, generator=g)
        assert len(set(chosen)) == 4
        _, replay, _ = sequence(model, x, emb, 4, actions=chosen)
        assert torch.equal(logp, replay.detach())
        samples.append((chosen, logp))
    before = {k: v.clone() for k, v in model.state_dict().items()}; steps = []
    def apply(opt, key): steps.append(key); opt.step()
    d = update(model, opt, x, emb, samples, [-1., -.2, .3, 1.], 4, 773, apply)
    assert len(steps) == 16 and set(steps) == set(range(16))
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    # The policy is equivariant to a candidate permutation when the ordered set is relabelled.
    perm = torch.tensor([3, 8, 1, 6, 0, 2, 7, 4, 5]); inverse = torch.argsort(perm)
    actions = samples[0][0]; remapped = inverse[torch.tensor(actions)].tolist()
    _, a, _ = sequence(model, x, emb, 4, actions=actions)
    _, b, _ = sequence(model, x[perm], emb[perm], 4, actions=remapped)
    assert torch.allclose(a, b, atol=1e-6)
    assert d['reward_std'] > 0
    return dict(status='PASS', unique_sampling=True, behavior_replay=True,
                permutation_equivariant=True, nonzero_update=True, synthetic_actor_updates=16)


if __name__ == '__main__':
    print(self_check())
