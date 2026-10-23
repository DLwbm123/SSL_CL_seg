"""Factorial U targets with step-indexed, role-separated posterior streams."""
import torch
from ..paired_semantic_v19.method import balanced_risk


def rng(state, stream):
    return torch.Generator().manual_seed(state.config.seed * 1000003 + state.steps * 101 + stream)


def posterior_probability(state, x, tag=70):
    values = []
    for outer in range(state.config.global_mc):
        gen = rng(state, tag + outer)
        theta = state.q.rsample(gen)
        a = state.current.rsample(theta, gen)
        context = state.phi.context(x, theta, a)
        offsets = state.phi.initial_offsets(len(x), x)
        for local in range(state.config.local_mc):
            noise = state.phi.noise(len(x), x, rng(state, tag + 10 + outer * 3 + local))
            z, _ = state.phi.rsample_and_log_prob(context, noise, offsets, state.model, theta)
            values.append(state.model.class_probabilities(z, theta))
    return torch.stack(values).mean(0)


def strong_view(state, x):
    gen = rng(state, 90)
    gain = .9 + .2 * torch.rand((), generator=gen).item()
    bias = -.05 + .1 * torch.rand((), generator=gen).item()
    noise = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
    return gain * x + bias + .03 * noise


def consistency(teacher, student):
    teacher = teacher.detach()
    pixel = (teacher * (teacher.clamp_min(1e-12).log() - student.clamp_min(1e-12).log())).sum(1)
    groups = teacher.argmax(1)
    values, details = [], {}
    for label, name in [(0, 'background'), (1, 'foreground')]:
        mask = groups == label
        details[name + '_pixels'] = int(mask.sum())
        if mask.any():
            value = pixel[mask].mean()
            values.append(value)
            details[name] = value.detach()
        else:
            details[name] = pixel.new_zeros(())
    return torch.stack(values).mean(), details


def components(state, L, U, counts):
    """Shared draws for L across arms; U consumption cannot advance L/global RNG."""
    state.assert_active()
    result = {k: state.q.mean.sum() * 0 for k in ('task', 'image_L', 'image_U')}
    pixels = state.config.height * state.config.width
    for outer in range(state.config.global_mc):
        gen = rng(state, 1 + outer)
        theta = state.q.rsample(gen)
        a = state.current.rsample(theta, gen)
        for role, batch, tag in [('L', L, 10), ('U', U, 30)]:
            if batch is None:
                continue
            state.check_batch(batch, role)
            x = batch['x']
            context = state.phi.context(x, theta, a)
            offsets = state.phi.initial_offsets(len(x), x)
            for local in range(state.config.local_mc):
                noise = state.phi.noise(len(x), x, rng(state, tag + 3 * outer + local))
                z, logq = state.phi.rsample_and_log_prob(context, noise, offsets, state.model, theta)
                parts = state.model.log_joint(x, z, theta, a, **batch.get('observations', {}))
                scale = state.config.global_mc * state.config.local_mc
                result['image_' + role] = result['image_' + role] - .01 * counts[role] / counts['L'] * (parts.total - parts['y'] - logq).mean() / (pixels * scale)
                if role == 'L':
                    truth = batch['observations']['y_obs'].to(x.dtype)
                    result['task'] = result['task'] + balanced_risk(state.model.class_probabilities(z, theta), truth).mean() / scale
    result['kl'] = (state.q.kl_to(state.prior) + state.current.expected_kl(state.q)) / (counts['L'] * pixels)
    return result


class Objective:
    def __init__(self, state, refiner, image_u=False, semantic_u=False):
        self.state = state
        self.image_u, self.semantic_u = image_u, semantic_u

    def estimate(self, L, U, counts, generator):
        terms = components(self.state, L, U if self.image_u else None, counts)
        loss = terms['task'] + terms['image_L'] + terms['kl']
        if self.image_u:
            loss = loss + terms['image_U']
        details = {}
        if self.semantic_u:
            self.state.check_batch(U, 'U')
            with torch.no_grad():
                teacher = posterior_probability(self.state, U['x'])
            student = posterior_probability(self.state, strong_view(self.state, U['x']))
            sem, details = consistency(teacher, student)
            loss = loss + sem
            terms['semantic_U'] = sem
        if not torch.isfinite(loss):
            raise FloatingPointError('factorial U objective')
        return loss, {**{k: v.detach() for k, v in terms.items()}, **details, 'total': loss.detach()}
