"""Bounded cumulative transforms and post-Adam displacement constraints."""
import torch
from ..five_frameworks_v1.kernels import StageSubspaceAdapter
from .protocol import GUARD


def spectral_stats(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError('finite square transform required')
    s = torch.linalg.svdvals(matrix.detach().double())
    return dict(min=float(s.min()), max=float(s.max()), condition=float(s.max()/s.min()))


def bounded(matrix):
    s = spectral_stats(matrix)
    return s['min'] >= GUARD['singular_min'] and s['max'] <= GUARD['singular_max']


class BoundedSidecar(StageSubspaceAdapter):
    def __init__(self, q, previous=None):
        super().__init__(q.contiguous(), previous)
        self.capture = False
        self.views = []
        self.ema_guard = {}
        if not bounded(self.previous):
            raise ValueError('inherited F outside V2 bounds: collapsed V1 targets cannot be repaired by import')

    def matrices(self):
        g = torch.eye(len(self.q), device=self.q.device, dtype=self.q.dtype)+self.q@self.r@self.q.T
        return g, self.previous@g

    def valid(self):
        return all(bounded(m) for m in self.matrices())

    def forward(self, h, *, detach_parent=False, backward_scale=None, return_pre_previous=False):
        if backward_scale is not None:
            raise ValueError('V2 has no feature-backward scaling')
        x = h.detach() if detach_parent else h
        g, f = self.matrices()
        before = torch.einsum('ij,bjhw->bihw', g, x)
        out = torch.einsum('ij,bjhw->bihw', f, x)
        if self.capture:
            self.views.append(f)
        return (out, before) if return_pre_previous else out

    @torch.no_grad()
    def update_teacher(self, teacher, decay=.99):
        before = teacher.r.clone()
        delta = (self.r-before)*(1-decay)
        alpha = 1.
        for _ in range(GUARD['backtracks']+1):
            teacher.r.copy_(before+alpha*delta)
            if teacher.valid():
                break
            alpha *= .5
        else:
            teacher.r.copy_(before)
            alpha = 0.
        if not teacher.valid():
            raise FloatingPointError('invalid previous EMA transform')
        teacher.ema_guard = dict(scale=alpha, G=spectral_stats(teacher.matrices()[0]),
                                 F=spectral_stats(teacher.matrices()[1]))

    @torch.no_grad()
    def seal(self):
        if not self.valid():
            raise FloatingPointError('cannot seal unbounded transform')
        return super().seal()


def dot(a, b):
    return sum((x.double()*y.double()).sum() for x, y in zip(a, b))


@torch.no_grad()
def constrain_displacement(params, before, references, valid):
    """References are loss gradients: non-increasing loss requires g dot delta <= 0.

    Adam moments retain the one computed gradient observation. Parameter damping
    is explicit; we do not falsify optimizer state or add optimizer invocations.
    """
    start = [before[id(p)] for p in params]
    proposal = [p.detach()-v for p, v in zip(params, start)]
    if not all(torch.isfinite(d).all() for d in proposal):
        raise FloatingPointError('nonfinite Adam proposal')
    refs = [r for r in references if float(dot(r, r)) > 0]
    delta = [d.clone() for d in proposal]
    def acceptable(ds):
        return all(float(dot(g, ds)) <= GUARD['dot_tolerance']*max(1e-20,
                   float(dot(g, g).sqrt()*dot(ds, ds).sqrt())) for g in refs)
    for _ in range(GUARD['cone_passes']):
        if acceptable(delta):
            break
        for g in refs:
            violation = dot(g, delta)
            if violation > 0:
                coefficient = violation/dot(g, g)
                delta = [(d.double()-coefficient*v.double()).to(d) for d, v in zip(delta, g)]
    cone_zero = not acceptable(delta)
    if cone_zero:
        delta = [torch.zeros_like(d) for d in delta]
    alpha = 1.
    for _ in range(GUARD['backtracks']+1):
        for p, v, d in zip(params, start, delta):
            p.copy_(v+alpha*d)
        actual = [p.detach()-v for p, v in zip(params, start)]
        if acceptable(actual) and valid():
            break
        alpha *= .5
    else:
        for p, v in zip(params, start):
            p.copy_(v)
        alpha = 0.
        actual = [torch.zeros_like(d) for d in delta]
    if not valid() or not acceptable(actual):
        raise FloatingPointError('invalid start state or infeasible guard')
    return dict(scale=alpha, cone_zero=cone_zero,
                proposal_norm=float(dot(proposal, proposal).sqrt()),
                actual_norm=float(dot(actual, actual).sqrt()),
                zero_displacement=all(not d.count_nonzero() for d in actual),
                dots_before=[float(dot(g, proposal)) for g in refs],
                dots_after=[float(dot(g, actual)) for g in refs])
