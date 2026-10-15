"""Training-L-only summaries, fixed native/readout coordinates, no image replay."""
import copy
import math
import torch
from torch.nn import functional as F
from .protocol import MEMORY


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return not isinstance(value,float) or math.isfinite(value)


class SiteMemory:
    def __init__(self, classes, width, history=()):
        self.classes, self.width = classes, width
        self.history = copy.deepcopy(list(history))
        self.observations = [0]*classes
        self.sensitivity = [{} for _ in range(classes)]
        self.means = torch.zeros(classes, width)
        self.seconds = torch.zeros(classes, width)
        self.support = torch.zeros(classes, dtype=torch.long)
        self.correct = torch.zeros(classes)
        self.predicted = torch.zeros(classes)
        self.validate()

    def validate(self):
        if not finite_tree({n:getattr(self,n) for n in ('history','sensitivity','means','seconds','support','correct','predicted')}):
            raise ValueError('nonfinite memory')
        if any(type(n) is not int or n<0 for n in self.observations):
            raise ValueError('invalid observation count')
        if self.support.shape!=(self.classes,) or self.support.dtype!=torch.long or self.correct.shape!=(self.classes,) or self.predicted.shape!=(self.classes,):
            raise ValueError('invalid support/reliability schema')
        if (self.support<0).any() or (self.seconds<0).any():
            raise ValueError('negative current support/second moment')
        if any((v<0).any() for d in self.sensitivity for v in d.values()):
            raise ValueError('negative sensitivity')
        domains = [h['domain'] for h in self.history]
        if len(set(domains)) != len(domains):
            raise ValueError('duplicate historical domain')
        for h in self.history:
            if h['means'].shape != (self.classes, self.width) or h['seconds'].shape != h['means'].shape:
                raise ValueError('historical prototype coordinate mismatch')
            if h['support'].shape != (self.classes,) or len(h['importance']) != self.classes:
                raise ValueError('historical class schema mismatch')
            if (h['support'] < 0).any() or (h['seconds'] < 0).any():
                raise ValueError('negative historical support/second moment')
            for importance in h['importance']:
                for name, value in importance.items():
                    if name not in h['anchor'] or value.shape != h['anchor'][name].shape or (value < 0).any():
                        raise ValueError('native importance coordinate mismatch')

    def assert_coordinates(self, values):
        for weights in self.sensitivity+[d for h in self.history for d in h['importance']]:
            if any(n not in values or values[n].shape!=v.shape for n,v in weights.items()):
                raise ValueError('memory native key/shape mismatch')

    def penalty(self, values):
        zero = next(iter(values.values())).sum()*0
        sites = []
        for h in self.history:
            classes = []
            for importance in h['importance']:
                terms = []
                for name, weight in importance.items():
                    if name not in values or values[name].shape != weight.shape:
                        raise ValueError('native anchor key/shape mismatch')
                    if not weight.count_nonzero():
                        continue
                    current = values[name]; anchor = h['anchor'][name].to(current); w = weight.to(current)
                    # A dimensionless layer mean prevents tensor size from setting the memory weight.
                    terms.append((w*(current-anchor).square()).sum()/w.sum().clamp_min(1e-20)
                                 /anchor.square().mean().clamp_min(1e-8))
                if terms:
                    classes.append(torch.stack(terms).mean())
            if classes:
                sites.append(torch.stack(classes).mean())
        return torch.stack(sites).mean() if sites else zero

    @torch.no_grad()
    def evidence(self, features, labels):
        if not self.history:
            return torch.ones_like(labels, dtype=features.dtype)
        banks = []
        for h in self.history:
            mean = h['means'].to(features)
            sim = torch.einsum('bdhw,cd->bchw', F.normalize(features, dim=1), F.normalize(mean, dim=1))
            sim[:, h['support'].to(features.device) == 0] = -1e4
            banks.append(sim)
        scores = torch.stack(banks).max(0).values/MEMORY['prototype_temperature']
        probability = scores.softmax(1).gather(1, labels[:, None])[:, 0]
        return .5+.5*probability  # Historical evidence discounts; it never hard-rejects a conflict.

    def reliability(self, device):
        a, b = MEMORY['reliability_prior']
        return ((self.correct+a)/(self.predicted+a+b)).to(device)

    @torch.no_grad()
    def commit(self, draft):
        for c, gradients, per_image in draft['classes']:
            self.observations[c] += 1
            for name, gradient in gradients.items():
                squared = gradient.detach().cpu().float().square()
                if name not in self.sensitivity[c]:
                    self.sensitivity[c][name] = torch.zeros_like(squared)
                self.sensitivity[c][name].add_(squared)
            for v in per_image.detach().cpu().float():
                self.support[c] += 1
                n = int(self.support[c])
                self.means[c].add_((v-self.means[c])/n)
                self.seconds[c].add_((v.square()-self.seconds[c])/n)
        if draft['calibration'] is not None:
            predicted, correct = draft['calibration']
            self.predicted.add_(predicted.cpu()); self.correct.add_(correct.cpu())
        self.validate()

    def export_site(self, domain, values):
        if not all(self.observations[c] > 0 and self.support[c] > 0 for c in range(1, self.classes)):
            raise ValueError('missing training-L foreground memory; no silent synthetic support')
        if any(h['domain'] == domain for h in self.history):
            raise ValueError('domain memory already sealed')
        result = dict(domain=domain, coordinate='native parameters and post-F fixed source-readout features',
                      means=self.means.clone(), seconds=self.seconds.clone(), support=self.support.clone(),
                      observations=list(self.observations),
                      anchor={n:v.detach().cpu().clone() for n,v in values.items()},
                      importance=[{n:v/max(1,self.observations[c]) for n,v in d.items()}
                                  for c,d in enumerate(self.sensitivity)])
        return copy.deepcopy(self.history)+[result]

    def state(self):
        return {name:copy.deepcopy(getattr(self, name)) for name in
                ('history', 'observations', 'sensitivity', 'means', 'seconds', 'support', 'correct', 'predicted')}

    def load(self, state):
        candidate = SiteMemory(self.classes, self.width)
        if set(state) != set(candidate.state()):
            raise ValueError('memory state fields mismatch')
        for name, value in copy.deepcopy(state).items():
            setattr(candidate, name, value)
        if candidate.means.shape != (self.classes, self.width) or candidate.seconds.shape != candidate.means.shape:
            raise ValueError('current memory shape mismatch')
        if any(len(getattr(candidate, n)) != self.classes for n in ('observations','sensitivity','support','correct','predicted')):
            raise ValueError('current memory class mismatch')
        if (candidate.correct < 0).any() or (candidate.predicted < candidate.correct).any():
            raise ValueError('invalid training-L reliability counts')
        candidate.validate()
        self.__dict__.update(candidate.__dict__)
