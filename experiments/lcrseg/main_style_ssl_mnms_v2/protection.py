"""Structural spectral feasibility and elastic actual-displacement protection."""
import math
import torch
from ..five_frameworks_v1.kernels import StageSubspaceAdapter
from .protocol import ORTHOGONAL_TOLERANCE, ELASTIC_STRENGTH, DUAL_SWEEPS


def spectrum(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError('finite square transform required')
    s = torch.linalg.svdvals(matrix.detach().double())
    return dict(min=float(s.min()), max=float(s.max()), condition=float(s.max()/s.min()))


def require_rotation(matrix):
    spectrum(matrix)
    m = matrix.detach().double()
    if float((m.T@m-torch.eye(len(m), device=m.device)).abs().max()) > ORTHOGONAL_TOLERANCE or float(torch.linalg.det(m)) <= 0:
        raise ValueError('proper orthogonal transform required; no clipping of old targets')


class OrthogonalSidecar(StageSubspaceAdapter):
    def __init__(self, q, previous=None):
        super().__init__(q.contiguous(), previous)
        require_rotation(self.previous)
        self.capture = False
        self.views = []

    def matrices(self):
        # FP64 solve is differentiable; all model activations remain FP32.
        q = self.q.double(); r = self.r.double(); k = (r-r.T)/2
        eye = torch.eye(len(r), device=r.device, dtype=r.dtype)
        rotation = torch.linalg.solve(eye-k/2, eye+k/2)
        g = torch.eye(len(q), device=q.device, dtype=q.dtype)+q@(rotation-eye)@q.T
        g = g.to(self.q)
        return g, self.previous@g

    def effective(self):
        return self.matrices()[1]

    def valid(self):
        for matrix in self.matrices():
            require_rotation(matrix)
        return True

    def forward(self, h, *, detach_parent=False, backward_scale=None, return_pre_previous=False):
        if backward_scale is not None:
            raise ValueError('no feature-backward scaling in this protocol')
        x = h.detach() if detach_parent else h
        g, f = self.matrices()
        before = torch.einsum('ij,bjhw->bihw', g, x)
        out = torch.einsum('ij,bjhw->bihw', f, x)
        if self.capture:
            self.views.append(f)
        return (out, before) if return_pre_previous else out

    @torch.no_grad()
    def update_teacher(self, teacher, decay=.99):
        teacher.r.mul_(decay).add_(self.r, alpha=1-decay)
        teacher.valid()

    @torch.no_grad()
    def seal(self):
        self.valid()
        return self.effective().detach().clone()


@torch.no_grad()
def elastic_displacement(params, before, references):
    """references = [(name, gradients, positive_weight)]; no spectral callback.

    Minimize .5||d-p||² + .5 sum(w_i relu(u_i dot d)²), sum(w)=1.
    The continuous solution retains at least half the proposal norm; FP32
    quantization can still erase sub-ULP displacements and is reported explicitly.
    """
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError('nonempty unique parameter list required')
    start = [before[id(p)] for p in params]
    if any(v.shape != p.shape or not torch.isfinite(v).all() for p,v in zip(params,start)):
        raise ValueError('invalid before state')
    proposal = torch.cat([(p.detach()-v).reshape(-1).double() for p,v in zip(params,start)])
    if not torch.isfinite(proposal).all():
        raise FloatingPointError('nonfinite proposal; state not committed')
    names=[]; units=[]; weights=[]; seen=set()
    for name, gradients, weight in references:
        if name in seen or len(gradients)!=len(params) or not math.isfinite(weight) or weight<=0:
            raise ValueError('invalid reference identity/weight')
        seen.add(name)
        if any(g.shape!=p.shape for g,p in zip(gradients,params)):
            raise ValueError('reference shape mismatch')
        vector=torch.cat([g.detach().reshape(-1).double() for g in gradients])
        if not torch.isfinite(vector).all():
            raise ValueError('nonfinite reference')
        norm=vector.norm()
        if norm>0:
            names.append(name); units.append(vector/norm); weights.append(weight)
    delta=proposal.clone(); converged=True; sweeps=0
    if units:
        u=torch.stack(units); w=proposal.new_tensor(weights)
        w=w/w.max(); w=ELASTIC_STRENGTH*w/w.sum()
        gram=u@u.T+torch.diag(1/w); b=u@proposal; alpha=torch.zeros_like(b)
        converged=False
        for sweeps in range(1, DUAL_SWEEPS+1):
            old=alpha.clone()
            for i in range(len(alpha)):
                alpha[i]=(alpha[i]+(b[i]-gram[i]@alpha)/gram[i,i]).clamp_min(0)
            if float((alpha-old).abs().max())<=1e-12*max(float(b.abs().max()),1e-30):
                converged=True
                break
        if not converged:
            raise FloatingPointError('elastic solver did not converge; no silent zero fallback')
        delta=proposal-u.T@alpha
        residual=delta-proposal+u.T@(w*(u@delta).clamp_min(0))
        if float(residual.norm())>1e-9*max(float(proposal.norm()),1e-30):
            raise FloatingPointError('elastic stationarity failure')
    if not torch.isfinite(delta).all():
        raise FloatingPointError('nonfinite corrected displacement')
    offset=0
    for p,v in zip(params,start):
        p.copy_(v+delta[offset:offset+p.numel()].reshape_as(p).to(p)); offset+=p.numel()
    actual=torch.cat([(p.detach()-v).reshape(-1).double() for p,v in zip(params,start)])
    return dict(proposal_norm=float(proposal.norm()), continuous_norm=float(delta.norm()),
                actual_norm=float(actual.norm()), zero_displacement=bool(not actual.count_nonzero()),
                continuous_retention=float(delta.norm()/proposal.norm()) if proposal.norm()>0 else None,
                actual_retention=float(actual.norm()/proposal.norm()) if proposal.norm()>0 else None,
                rounding_error=float((actual-delta).norm()), reference_names=names,
                dots_before=[float(v@proposal) for v in units], dots_after=[float(v@actual) for v in units],
                solver_sweeps=sweeps, converged=converged, spectral_backtracks=0,
                residual_conflicts=sum(bool(v@actual>0) for v in units))
