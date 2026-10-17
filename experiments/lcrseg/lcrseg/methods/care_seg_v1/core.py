"""Decision-tree readouts and bounded conditional repair (no trainable heads)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F


@dataclass(frozen=True)
class Node:
    name: str
    children: tuple[tuple[int, ...], ...]

    @property
    def classes(self):
        return tuple(c for group in self.children for c in group)


class DecisionTree:
    def __init__(self, classes, specification):
        self.classes = int(classes)
        self.nodes = [Node(str(n['name']), tuple(tuple(g) for g in n['children'])) for n in specification]
        if not self.nodes or len({n.name for n in self.nodes}) != len(self.nodes):
            raise ValueError('tree needs unique named nodes')
        by_set = {}
        for node in self.nodes:
            if len(node.children) < 2 or any(not g for g in node.children):
                raise ValueError('each node needs at least two nonempty children')
            if len(set(node.classes)) != len(node.classes) or any(c not in range(classes) for c in node.classes):
                raise ValueError('children must be disjoint valid classes')
            group = frozenset(node.classes)
            if group in by_set:
                raise ValueError('duplicate decision group')
            by_set[group] = node
        root = frozenset(range(classes))
        if frozenset(self.nodes[0].classes) != root:
            raise ValueError('root must partition all classes')
        seen = {root}
        for node in self.nodes:
            if frozenset(node.classes) not in seen:
                raise ValueError('tree must be connected and parent-first')
            for group in node.children:
                if len(group) > 1:
                    if frozenset(group) not in by_set:
                        raise ValueError('all non-leaf groups need a decision node')
                    seen.add(frozenset(group))
        if seen != set(by_set):
            raise ValueError('tree contains unreachable nodes')

    @staticmethod
    def read(probability, node):
        """Channel-first C,... input; return conditional probabilities and parent mass."""
        child = torch.stack([probability[list(g)].sum(0) for g in node.children])
        mass = child.sum(0)
        return child / mass.clamp_min(1e-8), mass

    @staticmethod
    def labels(label, node):
        target = torch.full_like(label, -1)
        for k, group in enumerate(node.children):
            for c in group:
                target[label == c] = k
        return target

    def loss(self, logits, entries, valid):
        """Partial conditional CE; missing children are not filled with historical targets."""
        probability = logits.float().softmax(1)
        per_case = []
        for i, decisions in enumerate(entries):
            numerator = logits[i].sum() * 0
            for node, entry in zip(self.nodes, decisions, strict=True):
                q, _ = self.read(probability[i], node)
                target, weight = entry['target'].detach(), entry['weight'].detach()
                numerator = numerator + (-(target * q.clamp_min(1e-8).log()).sum(0) * weight * valid[i, 0]).sum()
            per_case.append(numerator / valid[i, 0].sum().clamp_min(1))
        return torch.stack(per_case).mean()


def evidence(first, second, similarities):
    q = .5 * (first + second)
    logq = q.clamp_min(1e-8).log()
    js = .5 * ((first * (first.clamp_min(1e-8).log()-logq)).sum(0) +
               (second * (second.clamp_min(1e-8).log()-logq)).sum(0)) / np.log(2)
    score, pred = q.max(0)
    stability = (score * (1-js)).clamp(1e-6, 1-1e-6)
    chosen = similarities.gather(0, pred.unsqueeze(0))[0]
    competing = similarities.clone().scatter_(0, pred.unsqueeze(0), -torch.inf).max(0).values
    vector = torch.stack([torch.logit(stability), chosen-competing], -1)
    return q, pred, similarities.argmax(0), vector


def geometry_basis(mask, *, spacing, radius):
    """True volume or 2D grid; never silently treat MRI slices as independent volumes."""
    if mask.ndim not in (2, 3) or len(spacing) != mask.ndim or radius <= 0:
        raise ValueError('invalid geometry dimension/radius')
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('spacing must be finite and positive')
    axes = [np.linspace(-1, 1, size) if size > 1 else np.zeros(1) for size in mask.shape]
    coordinates = np.meshgrid(*axes, indexing='ij')
    basis = np.stack([np.ones(mask.shape), *coordinates])
    if not mask.any() or mask.all():
        return np.zeros(mask.shape), basis
    distance = distance_transform_edt(mask, sampling=spacing)-distance_transform_edt(~mask, sampling=spacing)
    return np.maximum(0, 1-np.abs(distance)/radius), basis


class Repair:
    """Binary conditional log-odds repair; fixed projected iterations are accounted."""
    def __init__(self, dimension, bound, steps, lr):
        if dimension not in (3, 4) or bound <= 0 or steps < 0 or lr <= 0:
            raise ValueError('invalid repair configuration')
        self.coefficient = torch.zeros(dimension)
        self.bound, self.steps, self.lr = bound, steps, lr
        self.updates = 0

    def fit(self, probability, basis, target, weights):
        if probability.numel() == 0:
            return
        a = torch.zeros(basis.shape[1], dtype=torch.float64, requires_grad=True)
        z = torch.logit(probability.detach().double().clamp(1e-6, 1-1e-6))
        x, y, w = basis.detach().double(), target.detach().double(), weights.detach().double()
        if not all(torch.isfinite(t).all() for t in (z, x, y, w)) or w.sum() <= 0:
            raise ValueError('nonfinite/empty repair support')
        for _ in range(self.steps):
            loss = (F.binary_cross_entropy_with_logits(z+x@a, y, reduction='none')*w).sum()/w.sum()+.01*a.square().sum()
            gradient, = torch.autograd.grad(loss, a)
            with torch.no_grad():
                a -= self.lr*gradient
                a *= min(1., self.bound / max(float(a.abs().sum()), 1e-12))
            self.updates += 1
        self.coefficient = a.detach().float()

    def apply(self, probability, basis):
        offset = torch.einsum('d,d...->...', self.coefficient.to(basis), basis)
        one = torch.sigmoid(torch.logit(probability[1].clamp(1e-6, 1-1e-6))+offset)
        return torch.stack([1-one, one]).detach()
