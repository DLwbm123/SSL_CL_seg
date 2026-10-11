"""GUA direction component only; no optimizer-state alignment or model mutation."""
import torch
from ..f5_class_cone_v15.model import class_cone


@torch.no_grad()
def align_displacement(proposal, references, metric):
    """Project a descent displacement in a supplied positive diagonal metric.

    proposal is theta_before - theta_proposed, including the actual group rates.
    metric is supplied explicitly; the GUA Adam choice is sqrt(v_hat) + eps.
    References are gradients at theta_before. Caller owns all optimizer state.
    """
    grids = [metric] + list(references)
    if len(references) > 3 or any(len(g) != len(proposal) for g in grids):
        raise ValueError('at most three references; identical parameter grids required')
    if any(a.shape != b.shape or a.device != b.device
           for g in grids for a, b in zip(proposal, g)):
        raise ValueError('parameter shape/device mismatch')
    values = list(proposal) + [x for g in grids for x in g]
    if any(not x.is_floating_point() for x in values):
        raise ValueError('floating tensors required')
    if any(not torch.isfinite(x).all() for x in values):
        raise FloatingPointError('nonfinite displacement/reference/metric')
    if any((m <= 0).any() for m in metric):
        raise ValueError('strictly positive diagonal metric required')

    def dots(direction):
        out = [sum((a.double() * b.double()).sum() for a, b in zip(direction, r))
               for r in references]
        if any(not torch.isfinite(x) for x in out):
            raise FloatingPointError('overflow reference dot')
        return [float(x) for x in out]

    before = dots(proposal) if proposal else [0.] * len(references)
    if all(x >= 0 for x in before):
        return proposal, dict(projected=False, dots_before=before, dots_after=before,
                              active_constraints=0)
    roots = [m.double().sqrt() for m in metric]
    transformed = [a.double() * root for a, root in zip(proposal, roots)]
    scale = sum(x.square().sum() for x in transformed).sqrt()
    if not torch.isfinite(scale) or scale <= 0:
        raise FloatingPointError('invalid transformed displacement norm')
    normals = [[h.double() / root for h, root in zip(r, roots)] for r in references]
    projected, stats = class_cone([x / scale for x in transformed], normals)
    corrected = [(z * scale / root).to(a.dtype)
                 for z, root, a in zip(projected, roots, proposal)]
    if any(not torch.isfinite(x).all() for x in corrected):
        raise FloatingPointError('nonfinite aligned displacement')
    return corrected, dict(projected=stats['projected'], dots_before=before,
                           dots_after=dots(corrected),
                           active_constraints=stats['active_constraints'])
