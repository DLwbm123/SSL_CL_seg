"""Exact four-halfspace Euclidean projection for a source-patient gradient cone."""
from itertools import combinations

import numpy as np
import torch


class PatientCone:
    def __init__(self, anchors):
        if anchors.ndim != 2 or anchors.shape[0] != 4:
            raise ValueError('expected four flattened patient gradients')
        norms = anchors.norm(dim=1)
        if not torch.isfinite(anchors).all() or bool((norms - 1).abs().max() > 1e-8):
            raise ValueError('source anchors must be finite unit vectors')
        self.anchors = anchors.detach()
        self.gram = (anchors @ anchors.T).cpu().numpy()
        if np.linalg.matrix_rank(self.gram, tol=1e-10) != 4:
            raise ValueError('source gradient anchors are linearly dependent')
        self.subsets = [tuple()] + [s for n in range(1, 5)
                                     for s in combinations(range(4), n)]

    def project(self, gradient):
        """Return min-norm correction subject to all four nonnegative dot products."""
        if gradient.ndim != 1 or gradient.numel() != self.anchors.shape[1]:
            raise ValueError('gradient dimension mismatch')
        if not torch.isfinite(gradient).all():
            raise ValueError('nonfinite generator gradient')
        before = (self.anchors @ gradient).detach().cpu().numpy()
        tol = 1e-10 * max(1., float(gradient.norm()))
        candidate = None
        for subset in self.subsets:
            multipliers = np.zeros(4, dtype=np.float64)
            if subset:
                s = np.asarray(subset, dtype=int)
                try:
                    multipliers[s] = np.linalg.solve(self.gram[np.ix_(s, s)], -before[s])
                except np.linalg.LinAlgError:
                    continue
                if np.min(multipliers[s]) < -tol:
                    continue
            after = before + self.gram @ multipliers
            if np.min(after) < -tol:
                continue
            cost = float(multipliers @ self.gram @ multipliers)
            if candidate is None or cost < candidate[0]:
                candidate = cost, multipliers
        if candidate is None:
            raise RuntimeError('no feasible four-patient gradient projection')
        multipliers = candidate[1]
        correction = torch.as_tensor(multipliers, device=gradient.device,
                                     dtype=gradient.dtype) @ self.anchors
        projected = gradient + correction
        after = (self.anchors @ projected).detach().cpu().numpy()
        if not torch.isfinite(projected).all() or np.min(after) < -2 * tol:
            raise RuntimeError('projected gradient violates source-patient cone')
        return projected, dict(before=before.tolist(),after=after.tolist(),
            correction_norm=float(correction.norm()),
            changed=bool(float(correction.norm()) > tol),
            generator_norm=float(gradient.norm()),
            active_multipliers=[float(v) for v in multipliers])
