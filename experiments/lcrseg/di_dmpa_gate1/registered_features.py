"""Scope-only recovery: full-map finite guard, strict registered-vector norms.

No sampling, feature transformation, epsilon, prototype fitting or model calls.
"""
from __future__ import annotations

import numpy as np

from .binding import H, NumericalError, require


class RegisteredFeatureNumericalError(NumericalError):
    def __init__(self, message, provenance, case_audits=None):
        self.provenance = provenance
        self.case_audits = case_audits or []
        super().__init__(f"{message}; provenance={provenance}")


class FullMapNumericalError(RegisteredFeatureNumericalError):
    """A complete-map NaN/Inf blocks even when no selected vector is invalid."""


def inspect_registered_case(feature, case, context):
    """Inspect all classes, then fail closed; never return an invalid cache.

    Coordinates retain their exact input order. Error diagnostics may be used
    by the authorized localization audit, but invalid features are never used.
    """
    feature = np.asarray(feature, dtype=np.float64)
    require(feature.ndim == 3 and feature.shape[0] == 16, "expected 16-D feature map")
    finite = np.isfinite(feature).all(axis=0)
    norms = np.full(finite.shape, np.nan, dtype=np.float64)
    norms[finite] = np.linalg.norm(feature[:, finite], axis=0)
    # Nonfinite vectors have no norm statistic and are never replaced.
    exact = finite & np.all(feature == 0, axis=0)
    near = finite & (norms <= 1e-12)
    zeros = np.argwhere(exact).tolist()
    positive = norms[finite & (norms > 0)]
    audit = dict(context, case_id=case.get('case_id', 'synthetic'),
        full_map_vector_count=int(finite.size), full_map_nonfinite_count=int((~finite).sum()),
        full_map_exact_zero_count=int(exact.sum()), full_map_norm_le_1e12_count=int(near.sum()),
        full_map_minimum_positive_norm=float(positive.min()) if len(positive) else None,
        full_map_zero_coordinate_sha256=H(zeros), first_zero_coordinates=zeros[:32])
    fields = ('registered_count', 'registered_nonfinite_count', 'registered_zero_count',
              'registered_exact_zero_count', 'registered_minimum_norm', 'registered_p01_norm',
              'registered_median_norm', 'registered_maximum_norm',
              'normalized_norm_max_abs_error', 'full_zero_registered_intersection')
    audit.update({f + '_by_class': {} for f in fields})
    selected_arrays, invalid = {}, []
    for c in range(3):
        coords = np.asarray(case['classes'][c]['coordinates'], dtype=np.int64).reshape(-1, 2)
        require(len(coords) == len(np.unique(coords, axis=0)), 'duplicate registered coordinates')
        require(not len(coords) or ((coords >= 0).all() and (coords[:, 0] < feature.shape[1]).all()
                and (coords[:, 1] < feature.shape[2]).all()), 'registered coordinate out of bounds')
        selected = feature[:, coords[:, 0], coords[:, 1]].T
        sf = np.isfinite(selected).all(axis=1)
        # Compute norms on original vectors, without epsilon or replacement.
        sn = np.linalg.norm(selected, axis=1)
        bad = ~sf | ~np.isfinite(sn) | (sn <= 1e-12)
        valid_norms = sn[sf & np.isfinite(sn)]
        stats = dict(registered_count=len(coords), registered_nonfinite_count=int((~sf).sum()),
            registered_zero_count=int((sf & (sn <= 1e-12)).sum()),
            registered_exact_zero_count=int((sf & np.all(selected == 0, axis=1)).sum()),
            registered_minimum_norm=float(valid_norms.min()) if len(valid_norms) else None,
            registered_p01_norm=float(np.quantile(valid_norms, .01)) if len(valid_norms) else None,
            registered_median_norm=float(np.median(valid_norms)) if len(valid_norms) else None,
            registered_maximum_norm=float(valid_norms.max()) if len(valid_norms) else None,
            normalized_norm_max_abs_error=None,
            full_zero_registered_intersection=int(exact[coords[:, 0], coords[:, 1]].sum()))
        if not bad.any():
            selected_arrays[c] = selected / sn[:, None]
            stats['normalized_norm_max_abs_error'] = (float(np.max(np.abs(
                np.linalg.norm(selected_arrays[c], axis=1) - 1))) if len(coords) else None)
        else:
            first = int(np.flatnonzero(bad)[0])
            invalid.append(dict(class_id=c, coordinate_y=int(coords[first, 0]),
                coordinate_x=int(coords[first, 1]), invalid_count=int(bad.sum()),
                minimum_selected_norm=stats['registered_minimum_norm']))
        for field, value in stats.items():
            audit[field + '_by_class'][str(c)] = value
    if invalid:
        provenance = dict(context, case_id=audit['case_id'], **invalid[0],
                          total_invalid_count=sum(x['invalid_count'] for x in invalid))
        raise RegisteredFeatureNumericalError('registered feature nonfinite or norm <=1e-12', provenance, [audit])
    if not finite.all():
        y, x = np.argwhere(~finite)[0]
        provenance = dict(context, case_id=audit['case_id'], class_id=None,
            coordinate_y=int(y), coordinate_x=int(x), total_invalid_count=int((~finite).sum()),
            minimum_selected_norm=min((x for x in audit['registered_minimum_norm_by_class'].values()
                                       if x is not None), default=None))
        raise FullMapNumericalError('nonfinite full feature map (including unsampled positions)', provenance, [audit])
    return selected_arrays, audit


def summarize_cases(cases):
    result = {'case_count': len(cases)}
    for field in ('full_map_vector_count', 'full_map_nonfinite_count', 'full_map_exact_zero_count',
                  'full_map_norm_le_1e12_count'):
        result[field] = sum(c[field] for c in cases)
    result['full_map_zero_case_count'] = sum(c['full_map_exact_zero_count'] > 0 for c in cases)
    result['full_map_zero_coordinate_sha256'] = H([[c['case_id'], c['full_map_zero_coordinate_sha256']] for c in cases])
    for field in ('registered_count', 'registered_nonfinite_count', 'registered_zero_count',
                  'registered_exact_zero_count', 'full_zero_registered_intersection'):
        result[field + '_by_class'] = {str(k): sum(c[field + '_by_class'][str(k)] for c in cases) for k in range(3)}
    result['registered_minimum_norm_by_class'] = {str(k): min((c['registered_minimum_norm_by_class'][str(k)]
        for c in cases if c['registered_minimum_norm_by_class'][str(k)] is not None), default=None) for k in range(3)}
    return result
