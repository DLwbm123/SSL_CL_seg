"""Validation evaluator only: case-balanced, positive-support risk curves."""
import hashlib
import json

import numpy as np

from .binding import require, finite, array_hash
from .reliability import CANDIDATES, poe_target

POINTS = (.05, .10, .20, .30, .40, .50)
STRATA = ('overall', 0, 1, 2)


def tie_keys(seed, stage, cases, height=384, width=384):
    """Full SHA-256, not a truncated random key; independent of all GT."""
    keys = np.empty((len(cases)*height*width, 4), dtype='>u8')
    for ci, case in enumerate(cases):
        prefix = json.dumps(['reliability-tie-v1', seed, stage, case], ensure_ascii=False, separators=(',', ':'))[:-1]+','
        for y in range(height):
            base = hashlib.sha256((prefix+str(y)+',').encode())
            for x in range(width):
                h = base.copy(); h.update((str(x)+']').encode())
                keys[ci*height*width+y*width+x] = np.frombuffer(h.digest(), dtype='>u8')
    return keys


def case_weights(case_index, mask):
    case_index = np.asarray(case_index, np.int64); mask = np.asarray(mask, bool)
    require(case_index.shape == mask.shape and np.all(case_index >= 0), 'case weights shape/index')
    weights = np.zeros(len(mask), np.float64)
    if mask.any():
        counts = np.bincount(case_index[mask]); ncase = np.count_nonzero(counts)
        weights[mask] = 1.0/ncase/counts[case_index[mask]]
    return weights


def ranked_curve(weights, correct, mass, ties):
    weights = np.asarray(weights, np.float64); correct = np.asarray(correct, bool); mass = np.asarray(mass, np.float64)
    finite(weights, mass); require(weights.shape == correct.shape == mass.shape, 'curve shape')
    require(ties.shape == (len(weights), 4) and np.all(weights >= 0) and np.all(mass >= 0), 'curve input')
    pos = np.flatnonzero((weights > 0) & (mass > 0))
    keys = ties[pos]
    order = np.lexsort((keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0], -weights[pos]))
    pos = pos[order]
    if not len(pos):
        return dict(pos=pos, x=np.empty(0), risk=np.empty(0), integral=np.empty(0), maximum=0.0,
                    full_mass=float(mass.sum()), positive_count=0, full_aurc=None)
    w = mass[pos]/mass.sum(); x = np.cumsum(w); risk = 1-np.cumsum(w*correct[pos])/x
    integral = np.cumsum(np.diff(np.r_[0., x])*risk)
    curve = dict(pos=pos, x=x, risk=risk, integral=integral, maximum=min(1., float(x[-1])),
                 full_mass=float(mass.sum()), positive_count=len(pos))
    curve['full_aurc'] = aurc(curve, curve['maximum'])
    return curve


def point(curve, coverage):
    if coverage <= 0 or not len(curve['pos']):
        return dict(requested=coverage, precision=None, achieved=None, prefix_count=0, reason='EMPTY_POSITIVE_SUPPORT')
    if coverage > curve['maximum']:
        return dict(requested=coverage, precision=None, achieved=None, prefix_count=0, reason='OUTSIDE_POSITIVE_SUPPORT')
    index = int(np.searchsorted(curve['x'], coverage, side='left'))
    require(index < len(curve['x']), 'coverage prefix out of range')
    return dict(requested=coverage, precision=float(1-curve['risk'][index]), achieved=float(curve['x'][index]),
                prefix_count=index+1, reason=None)


def aurc(curve, upper):
    if upper <= 0 or upper > curve['maximum'] or not len(curve['pos']):
        return None
    i = int(np.searchsorted(curve['x'], upper, side='left'))
    previous = float(curve['x'][i-1]) if i else 0.
    area = float(curve['integral'][i-1]) if i else 0.
    return float((area+(upper-previous)*curve['risk'][i])/upper)


def calibration(confidence, correct, mass):
    confidence = np.asarray(confidence, np.float64); finite(confidence, mass)
    require(np.all((confidence >= 0) & (confidence <= 1)), 'calibration scores outside [0,1]')
    bins = np.minimum((confidence*15).astype(np.int64), 14); total = float(mass.sum()); rows = []; ece = 0.
    for b in range(15):
        mask = (bins == b) & (mass > 0); w = mass[mask]; amount = float(w.sum())
        accuracy = float(np.sum(w*correct[mask])/amount) if amount else None
        mean = float(np.sum(w*confidence[mask])/amount) if amount else None
        fraction = amount/total if total else 0.
        rows.append(dict(bin=b, lower=b/15, upper=(b+1)/15, count=int(mask.sum()), mass=fraction,
            mean_confidence=mean, accuracy=accuracy, empty=not bool(amount)))
        if amount:
            ece += fraction*abs(accuracy-mean)
    return (float(ece) if total else None), rows


def brier(probability, labels, mass):
    finite(probability)
    selected = mass > 0
    if not selected.any():
        return None
    p = probability[selected]; y = labels[selected]
    require(np.all((y >= 0) & (y < 3)), 'Brier evaluator label mapping')
    squared = (p-np.eye(3)[y])**2
    return float(np.sum(squared.sum(1)*mass[selected])/mass[selected].sum())


def composition(curve, coverage, prediction, labels, mass):
    p = point(curve, coverage); chosen = curve['pos'][:p['prefix_count']]
    amount = float(mass[chosen].sum()); rows = []
    for c in range(3):
        pred_mass = float(mass[prediction == c].sum()); true_mass = float(mass[labels == c].sum())
        pred_kept = float(mass[chosen][prediction[chosen] == c].sum())
        true_kept = float(mass[chosen][labels[chosen] == c].sum())
        correct_kept = float(mass[chosen][(labels[chosen] == c) & (prediction[chosen] == c)].sum())
        rows.append(dict(class_id=c, requested_coverage=coverage, achieved_coverage=p['achieved'], reason=p['reason'],
            predicted_retained_fraction=pred_kept/pred_mass if pred_mass and p['reason'] is None else None,
            true_retained_fraction=true_kept/true_mass if true_mass and p['reason'] is None else None,
            true_class_recall=correct_kept/true_mass if true_mass and p['reason'] is None else None,
            predicted_composition=pred_kept/amount if amount else None,
            true_composition=true_kept/amount if amount else None, predicted_full_mass=pred_mass, true_full_mass=true_mass))
    return rows


def evaluate(seed, stage, cases, caches, labels, *, include_poe=False, height=384, width=384):
    """GT crosses this evaluator boundary only, after caches are sealed."""
    require(len(cases) == len(caches) == len(labels), 'evaluator case count mismatch')
    y = np.concatenate([np.asarray(a).reshape(-1) for a in labels]).astype(np.int64)
    require(set(np.unique(y)).issubset({0, 1, 2, 255}), 'invalid evaluator GT mapping')
    n = height*width; require(len(y) == len(cases)*n, 'evaluator geometry')
    ci = np.repeat(np.arange(len(cases)), n); ties = tie_keys(seed, stage, cases, height, width)
    probability = np.concatenate([a['teacher_probability'] for a in caches]).astype(np.float64)
    active = np.concatenate([a['active_mask'] for a in caches]); valid = y != 255
    prediction = probability.argmax(1); finite(probability)
    targets = {c: dict(probability=probability, prediction=prediction, available=np.ones(len(y), bool),
                      weights=np.concatenate([a[c] for a in caches]).astype(np.float64)) for c in CANDIDATES}
    if include_poe:
        controls = [poe_target(a) for a in caches]
        pp = np.concatenate([a['probability'] for a in controls]); available = np.concatenate([a['valid'] for a in controls])
        targets['PoE'] = dict(probability=pp, prediction=np.where(available, pp.argmax(1), -1), available=available,
                              weights=np.concatenate([a['weights'] for a in controls]))
    curves = {}; masses = {}; classwise = []; points = []; calibrations = []; compositions = []; guards = []
    for candidate, t in targets.items():
        for cls in STRATA:
            mask = valid if cls == 'overall' else valid & (t['prediction'] == cls)
            mass = case_weights(ci, mask); masses[candidate, cls] = mass
            correct = t['prediction'] == y
            curves[candidate, cls] = ranked_curve(t['weights'], correct, mass, ties)
    for candidate, t in targets.items():
        for cls in STRATA:
            curve = curves[candidate, cls]; ref = curves['R1', cls]; mass = masses[candidate, cls]
            common = min(curve['maximum'], ref['maximum']); correct = t['prediction'] == y
            ece, bins = calibration(t['weights'], correct, mass)
            confidence_mass = mass*t['available']
            teacher_ece, _ = calibration(t['probability'].max(1), correct, confidence_mass)
            pred_points = [point(curve, x) for x in POINTS]; ref_points = [point(ref, x) for x in POINTS]
            row = dict(candidate=candidate, seed=seed, stage_index=stage, class_id=cls,
                maximum_supported_coverage=curve['maximum'], common_upper_bound=common,
                common_support_AURC=aurc(curve, common), reference_common_support_AURC=aurc(ref, common),
                full_support_AURC=curve['full_aurc'], precision_points=pred_points, reference_points=ref_points,
                actual_shared_points=[x for x, a, b in zip(POINTS, pred_points, ref_points) if a['precision'] is not None and b['precision'] is not None],
                reliability_ECE_15=ece, probability_ECE_15=teacher_ece, multiclass_Brier=brier(t['probability'], y, confidence_mass),
                calibration_probability_available_mass=float(confidence_mass.sum()), positive_pixel_count=curve['positive_count'],
                full_valid_pixel_count=int(np.count_nonzero(mass)), null_count=int(np.count_nonzero((mass > 0) & ~active)),
                null_case_balanced_mass=float(mass[~active].sum()), case_count=int(len(np.unique(ci[mass > 0]))),
                ranked_coverage_sha256=array_hash(curve['x']), ranked_risk_sha256=array_hash(curve['risk']),
                full_curve_storage='exact ranked steps reconstructible from immutable per-pixel caches and full SHA tie rule')
            classwise.append(row)
            for pt in pred_points+[point(curve, curve['maximum'])]:
                points.append(dict(candidate=candidate, seed=seed, stage_index=stage, class_id=cls,
                    requested_coverage=pt['requested'], precision=pt['precision'], achieved_coverage=pt['achieved'],
                    reason=pt['reason'], point_kind='registered' if pt['requested'] in POINTS else 'maximum_support'))
            for b in bins:
                calibrations.append(dict(candidate=candidate, seed=seed, stage_index=stage, class_id=cls, **b))
        fg = [x for x in classwise if x['candidate'] == candidate and x['class_id'] in (1, 2)]
        macro = dict(fg[0], class_id='foreground_macro', common_upper_bound=None,
            maximum_supported_coverage=min(x['maximum_supported_coverage'] for x in fg),
            ranked_coverage_sha256=None, ranked_risk_sha256=None,
            full_curve_storage='equal rim/cup macro; class-specific common-support bounds and raw curve hashes are reported separately')
        for field in ('common_support_AURC', 'reference_common_support_AURC', 'full_support_AURC', 'reliability_ECE_15',
                      'probability_ECE_15', 'multiclass_Brier', 'calibration_probability_available_mass', 'null_case_balanced_mass'):
            values = [x[field] for x in fg]
            macro[field] = float(np.mean(values)) if all(x is not None for x in values) else None
        for field in ('positive_pixel_count', 'full_valid_pixel_count', 'null_count'):
            macro[field] = sum(x[field] for x in fg)
        macro['case_count'] = int(len(np.unique(ci[valid & np.isin(t['prediction'], [1, 2])])))
        macro['actual_shared_points'] = [x for x in POINTS if all(x in y['actual_shared_points'] for y in fg)]
        for field in ('precision_points', 'reference_points'):
            macro[field] = []
            for index, x in enumerate(POINTS):
                pair = [y[field][index] for y in fg]; supported = all(y['precision'] is not None for y in pair)
                macro[field].append(dict(requested=x, precision=float(np.mean([y['precision'] for y in pair])) if supported else None,
                    achieved=float(np.mean([y['achieved'] for y in pair])) if supported else None,
                    reason=None if supported else 'FOREGROUND_CLASS_UNSUPPORTED'))
        classwise.append(macro)
        for pt in macro['precision_points']:
            points.append(dict(candidate=candidate, seed=seed, stage_index=stage, class_id='foreground_macro',
                requested_coverage=pt['requested'], precision=pt['precision'], achieved_coverage=pt['achieved'], reason=pt['reason'], point_kind='registered'))
        curve = curves[candidate, 'overall']; ref = curves['R1', 'overall']
        operating = min(.50, curve['maximum'], ref['maximum'])
        a = composition(curve, operating, t['prediction'], y, masses[candidate, 'overall'])
        b = composition(ref, operating, prediction, y, masses['R1', 'overall'])
        for c in (1, 2):
            av = a[c]['predicted_retained_fraction']; bv = b[c]['predicted_retained_fraction']
            guards.append(dict(candidate=candidate, seed=seed, stage_index=stage, class_id=c, global_operating_point=operating,
                candidate_retained_fraction=av, reference_retained_fraction=bv,
                candidate_achieved_coverage=a[c]['achieved_coverage'], reference_achieved_coverage=b[c]['achieved_coverage'],
                comparison_rhs=None if bv is None else .8*bv, pass_=bool(operating > 0 and av is not None and bv is not None and av >= .8*bv)))
        for kind, x in [('matched_global', operating)]+[('registered', x) for x in POINTS]:
            for r in composition(curve, x, t['prediction'], y, masses[candidate, 'overall']):
                compositions.append(dict(candidate=candidate, seed=seed, stage_index=stage, point_kind=kind, **r))
    poe = None
    if include_poe:
        t = targets['PoE']; mass = case_weights(ci, valid); available = t['available'] & valid
        amount = float(mass[available].sum())
        poe = dict(seed=seed, stage_index=stage, available_mass=amount, null_count=int(np.count_nonzero(~active & valid)),
            changed_prediction_rate=float(mass[available & (t['prediction'] != prediction)].sum()/amount) if amount else None,
            same_R3_weights=True, own_predicted_class_strata=True)
    return dict(seed=seed, stage_index=stage, case_count=len(cases), pixel_count=len(y), non_ignore_count=int(valid.sum()),
        classwise=classwise, precision_coverage=points, calibration=calibrations, composition=compositions,
        coverage_guard=guards, poe=poe, hidden_gt_training_usage='none', test_gt_usage='none', labels_consumer='diagnostic_evaluator_only')
