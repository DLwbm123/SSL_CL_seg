"""Independent mathematical kernels for the five-framework handoff.

Adapted from the user-supplied V1 reference kernels. Integration and verified
external backends live in the adjacent modules. All teacher targets, masks and spectral scales are stop-gradient.
The feature-space adapter is deliberately distinct from a weight-space projector.
"""
from __future__ import annotations

from typing import Callable, Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _finite(x: Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite")


def _prob(x: Tensor, name: str) -> None:
    _finite(x, name)
    if bool(((x < 0) | (x > 1)).any()):
        raise ValueError(f"{name} must be in [0,1]")


def _mask(mask: Tensor, p: Tensor) -> Tensor:
    if p.ndim != 4 or mask.shape != (p.shape[0], *p.shape[2:]):
        raise ValueError("expected BCHW probabilities and BHW mask")
    if mask.dtype != torch.bool:
        raise TypeError("valid mask must be boolean")
    return mask.detach()


def masked_kl(logits: Tensor, target: Tensor, valid: Tensor,
              temperature: float = 1.0) -> Tensor:
    """Forward KL; target already represents the SAME temperature distribution.

    Reduction: sum over classes, average over all accepted pixels. Empty support
    yields a graph-connected zero. Teacher entropy changes the value, not the
    student gradient, relative to identically reduced soft cross entropy.
    """
    if logits.shape != target.shape or temperature <= 0:
        raise ValueError("shape mismatch or invalid temperature")
    _finite(logits, "logits"); _prob(target, "target")
    valid = _mask(valid, logits)
    q = target.detach().to(logits)
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("target must sum to one across classes")
    logp = F.log_softmax(logits / temperature, dim=1)
    terms = torch.xlogy(q, q) - q * logp
    return temperature**2 * (terms.sum(1) * valid).sum() / valid.sum().clamp_min(1)


def masked_jml1(student: Tensor, teacher: Tensor, valid: Tensor,
                foreground: Sequence[int] = (1, 2), eps: float = 1e-12) -> Tensor:
    """Masked, equal-image/equal-class JML1 adaptation.

    Implements 1-(||p+q||_1-||p-q||_1)/(||p+q||_1+||p-q||_1).
    Both-zero class => zero, all-invalid image excluded from image denominator.
    No thresholding of student probabilities, no Dice substitution.
    """
    if student.shape != teacher.shape or not foreground:
        raise ValueError("shape mismatch or empty class set")
    _prob(student, "student"); _prob(teacher, "teacher")
    valid = _mask(valid, student)
    classes = tuple(foreground)
    if min(classes) < 0 or max(classes) >= student.shape[1]:
        raise ValueError("invalid foreground index")
    p = student[:, classes].double()
    q = teacher.detach()[:, classes].double()
    m = valid[:, None]
    plus = ((p + q).abs() * m).sum((-2, -1))
    diff = ((p - q).abs() * m).sum((-2, -1))
    den = plus + diff
    per = torch.where(den > eps, 2 * diff / den.clamp_min(eps), den * 0)
    active = valid.flatten(1).any(1)
    return ((per.mean(1) * active).sum() / active.sum().clamp_min(1)).to(student.dtype)


def collect_sources(a: Tensor, b: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Select rather than average complementary-view predictions."""
    if a.shape != b.shape:
        raise ValueError("view shapes differ")
    mask = _mask(mask, a)[:, None]
    return torch.where(mask, a, b), torch.where(mask, b, a)


def channel_basis(kernel: Tensor, rank: int, previous_transform: Tensor | None = None
                  ) -> tuple[Tensor, Tensor]:
    """GOLD-inspired channel proxy, not the full convolution operator row space.

    Native readout C maps F_prev @ h to logits. Thus C_eff_delta=C_delta@F_prev
    and H=sum_delta C_eff_delta.T Pi C_eff_delta. Q acts on raw h, before F_prev.
    No hard-coded 3x3 kernel, class count, channel count or padding assumption.
    """
    if kernel.ndim != 4:
        raise ValueError("expected K,D,Kh,Kw readout")
    _finite(kernel, "kernel")
    c, d = kernel.shape[:2]
    if c < 2 or not 1 <= rank <= d:
        raise ValueError("invalid rank or class count")
    w = kernel.detach().to(device="cpu", dtype=torch.float64)
    if previous_transform is not None:
        if previous_transform.shape != (d, d):
            raise ValueError("previous transform shape mismatch")
        _finite(previous_transform, "previous_transform")
        w = torch.einsum('ocij,cd->odij', w, previous_transform.detach().to(w))
    # Pi-centering avoids the softmax common-logit direction.
    w = w - w.mean(0, keepdim=True)
    rows = w.permute(2, 3, 0, 1).reshape(-1, d)
    gram = rows.T @ rows
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    values = values.flip(0).clamp_min(0)
    q = (torch.eye(d, dtype=w.dtype)[:, :rank] if values.max() == 0
         else vectors.flip(1)[:, :rank])
    # Canonical signs only; repeated eigenvalues still permit basis rotations.
    pivots = q.abs().argmax(0)
    signs = q[pivots, torch.arange(rank, device=q.device)].sign()
    q = q * torch.where(signs == 0, torch.ones_like(signs), signs)
    return q.to(kernel), values[:rank].to(kernel)


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, scale: Tensor) -> Tensor:
        ctx.save_for_backward(scale.detach())
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (scale,) = ctx.saved_tensors
        return gradient * scale, None


def gradient_scale_identity(value: Tensor, scale: Tensor) -> Tensor:
    """An explicit custom update rule: identity forward, scaled backward.

    NOT the derivative of the identity function and not a scalar-loss gradient.
    Test forward identity and backward against its stated rule, not gradcheck.
    """
    try:
        torch.broadcast_shapes(value.shape, scale.shape)
    except RuntimeError as e:
        raise ValueError("scale cannot broadcast") from e
    _finite(scale, "scale")
    if bool((scale < 0).any()):
        raise ValueError("scale must be nonnegative")
    return _GradientScale.apply(value, scale.detach().to(value))


class StageSubspaceAdapter(nn.Module):
    """Current-stage G=I+Q R Q^T, with constant-size learned transform F_prev.

    Forward: F_prev @ G @ h. At boundary: F_next=F_prev@G. This keeps
    classifier parameters untouched and introduces ONE declared D-by-D learned
    parameter memory, not one adapter per task. Q is ephemeral between stages.
    Parent isolation and the legacy trainer are not implemented by this class.
    """
    def __init__(self, q: Tensor, previous: Tensor | None = None):
        super().__init__()
        if q.ndim != 2:
            raise ValueError("Q must be D by k")
        d, k = q.shape
        if not torch.allclose(q.T @ q, torch.eye(k, device=q.device, dtype=q.dtype),
                              atol=2e-5, rtol=2e-5):
            raise ValueError("Q must have orthonormal columns")
        previous = torch.eye(d, device=q.device, dtype=q.dtype) if previous is None else previous
        if previous.shape != (d, d):
            raise ValueError("F_prev shape mismatch")
        self.register_buffer("q", q.detach().clone())
        # Frozen learned parameter, not a data-derived historic statistic.
        self.previous = nn.Parameter(previous.detach().clone(), requires_grad=False)
        self.r = nn.Parameter(q.new_zeros(k, k))

    def forward(self, h: Tensor, *, detach_parent: bool = False,
                backward_scale: Tensor | None = None,
                return_pre_previous: bool = False):
        if h.ndim != 4 or h.shape[1] != self.q.shape[0]:
            raise ValueError("feature channel mismatch")
        x = h.detach() if detach_parent else h
        z = torch.einsum('dk,bdhw->bkhw', self.q, x)
        shift = torch.einsum('ij,bjhw->bihw', self.r, z)
        if backward_scale is not None:
            shift = gradient_scale_identity(shift, backward_scale)
        before = x + torch.einsum('dk,bkhw->bdhw', self.q, shift)
        output = torch.einsum('de,behw->bdhw', self.previous, before)
        return (output, before) if return_pre_previous else output

    def effective(self) -> Tensor:
        d = self.q.shape[0]
        return self.previous @ (torch.eye(d, device=self.q.device, dtype=self.q.dtype)
                                + self.q @ self.r @ self.q.T)

    @torch.no_grad()
    def seal(self) -> Tensor:
        return self.effective().detach().clone()


def uncertainty_scales(uncertainty: Tensor, eigenvalues: Tensor, kappa: float,
                       eps: float = 1e-6) -> Tensor:
    """BHW uncertainty -> BkHW fixed spectral preconditioner."""
    _prob(uncertainty, "uncertainty")
    if uncertainty.ndim != 3 or eigenvalues.ndim != 1 or kappa < 0:
        raise ValueError("invalid spectral scale arguments")
    s = eigenvalues.detach().to(uncertainty)
    _finite(s, "eigenvalues")
    if bool((s < 0).any()):
        raise ValueError("negative eigenvalues")
    s = s / s.max().clamp_min(eps)
    return (1 + kappa * uncertainty.detach()[:, None] /
            (s + eps)[None, :, None, None]).reciprocal()


def gradient_seed_basis(gradients: Sequence[Tensor], rank: int,
                        free_projector: Tensor | None = None) -> Tensor:
    """Top right singular directions of CURRENT-L gradients, without dense G^T G.

    Caller freezes sample order, scale calibration and parent constraint semantics.
    Empty/zero probes raise; production caller must explicitly log and keep the
    parent's legal initialization, never invent a successful structural basis.
    """
    if not gradients:
        raise ValueError("no current-L probes")
    rows = torch.cat([g.detach().double().flatten(1) for g in gradients], dim=0)
    _finite(rows, "gradient rows")
    d = rows.shape[1]
    if free_projector is not None:
        if free_projector.shape != (d, d):
            raise ValueError("input projector dimension mismatch")
        p = free_projector.detach().to(rows)
        if not torch.allclose(p, p.T, atol=1e-8, rtol=1e-8) or not torch.allclose(p@p, p, atol=1e-8, rtol=1e-8):
            raise ValueError("not an orthogonal projector")
        rows = rows @ p
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    if not 1 <= rank <= len(s) or s[0] <= 0:
        raise ValueError("insufficient probe rank")
    numerical_rank = int((s > s[0] * max(rows.shape) * torch.finfo(s.dtype).eps).sum())
    if numerical_rank < rank:
        raise ValueError("insufficient probe rank")
    return vh[:rank].T.to(gradients[0])


def sliced_wasserstein_equal(a: Tensor, b: Tensor, directions: Tensor) -> Tensor:
    """Exact empirical 1-D W2 average for equal-cardinality sampled sets.

    Caller performs class conditioning and current-stage sampling. No persistent
    queue, no Gaussian shortcut, no uniformity loss. b is stop-gradient.
    """
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] == 0:
        raise ValueError("nonempty equal N-by-D sets required")
    if directions.ndim != 2 or directions.shape[0] != a.shape[1] or directions.shape[1] == 0:
        raise ValueError("direction shape mismatch")
    for name, value in (("a", a), ("b", b), ("directions", directions)):
        _finite(value, name)
    norms = directions.norm(dim=0)
    if bool((norms == 0).any()):
        raise ValueError("zero slicing direction")
    v = (directions / norms).detach().to(a)
    av = (a @ v).sort(dim=0).values
    bv = (b.detach().to(a) @ v).sort(dim=0).values
    return (av - bv).square().mean()


def current_prototypes(features: Tensor, labels: Tensor, classes: int) -> tuple[Tensor, Tensor]:
    """Equal-present-image means in a caller-verified common feature coordinate.

    This computes a CURRENT batch estimate, not a historical bank. ignore=255 is
    excluded. Production code may maintain a current-stage EMA of these means.
    """
    if features.ndim != 4 or labels.shape != (features.shape[0], *features.shape[2:]):
        raise ValueError("features/labels must already be geometrically aligned")
    if labels.dtype != torch.long or classes < 2:
        raise ValueError("integer labels and >=2 classes required")
    if bool((~(((labels >= 0) & (labels < classes)) | (labels == 255))).any()):
        raise ValueError("invalid label")
    h = features.detach()
    proto = h.new_zeros(classes, h.shape[1]); support = torch.zeros(classes, device=h.device, dtype=torch.bool)
    for c in range(classes):
        mask = labels == c
        den = mask.sum((-2, -1))
        present = den > 0
        if bool(present.any()):
            per_image = (h * mask[:, None]).sum((-2, -1)) / den.clamp_min(1)[:, None]
            mean = per_image[present].mean(0)
            if mean.norm() > 1e-12:
                proto[c] = F.normalize(mean, dim=0)
                support[c] = True
    return proto, support


def teacher_pas(probabilities: Tensor, features: Tensor, prototypes: Tensor,
                support: Tensor, geometry: Tensor, *, confidence: float = .7,
                similarity: float = .5) -> tuple[Tensor, Tensor]:
    """Current-domain, teacher-only PAS-inspired check; NOT full JASCL.

    Missing prototype => confidence-only admission, explicitly reported by caller.
    No student-confidence condition and no teacher-target refinement are hidden.
    """
    _prob(probabilities, "teacher probabilities")
    geometry = _mask(geometry, probabilities)
    b, c, h, w = probabilities.shape
    if features.shape[0] != b or features.shape[2:] != (h,w):
        raise ValueError("teacher feature and output coordinates are not aligned")
    if prototypes.shape != (c,features.shape[1]) or support.shape != (c,) or support.dtype != torch.bool:
        raise ValueError("prototype/support shape mismatch")
    if not 0 <= confidence <= 1 or not -1 <= similarity <= 1:
        raise ValueError("threshold out of range")
    q = probabilities.detach(); f = features.detach()
    conf, labels = q.max(1)
    selected = prototypes.detach()[labels].permute(0,3,1,2).to(f)
    sim = F.cosine_similarity(f, selected, dim=1)
    supported = support.detach()[labels]
    keep = geometry & (conf > confidence) & ((sim > similarity) | ~supported)
    return keep.detach(), sim.detach()
