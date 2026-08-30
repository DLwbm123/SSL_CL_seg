"""GT-free, detached K2 identity-history scoring and frozen legacy PAS."""
import numpy as np
import torch
from scipy.special import expit, logsumexp, softmax

from di_dmpa_gate1_v2.features import split_support
from di_dmpa_gate1_v2.binding import NonfiniteFeature
from di_dmpa_jascl.modeling import compute_pas_validity
from .binding import require, finite, NonfiniteEvidence, DOMAINS, H

CANDIDATES = ('R0', 'R1', 'R2', 'R3')


def banks(freeze, seed, stage, *, transform='identity'):
    require(transform == 'identity', 'T1/T2/transformed bank forbidden')
    require(freeze['selected_K'] == 2 and freeze['primary_panel'] == 'B0-EMA', 'wrong K/panel')
    require(seed in range(3) and stage in range(3), 'unknown seed/stage')
    selected = []
    for t in range(stage+1):
        group = []
        for c in range(3):
            match = [r for r in freeze['prototype_records'] if (r['seed'], r['stage_index'], r['class_id']) == (seed, t, c)]
            require(len(match) == 1, 'missing/duplicate prototype record')
            r = match[0]
            require(r['domain'] == DOMAINS[t] and r['panel'] == 'B0-EMA' and r['feature_source'] == 'ema_teacher' and
                    r['baseline'] == 'B0' and r['K'] == 2 and r['training_source'] == 'train_labeled' and
                    r['converged'] and r['active_mask'] == [True, True] and not r['operational_refit_allowed'], 'invalid operational prototype provenance')
            a = np.asarray(r['centers'], dtype=np.float64); finite(a)
            require(a.shape == (2, 16) and np.all(np.abs(np.linalg.norm(a, axis=1)-1) <= 1e-12), 'invalid prototype center')
            group.append(a)
        selected.append(np.stack(group))
    current = selected[-1]; history = np.concatenate(selected[:-1], axis=1) if stage else np.empty((3, 0, 16), np.float64)
    current.setflags(write=False); history.setflags(write=False)
    return current, history


def bank_identity(freeze, seed, stage):
    records = [r for t in range(stage+1) for c in range(3) for r in freeze['prototype_records']
               if (r['seed'], r['stage_index'], r['class_id']) == (seed, t, c)]
    return dict(seed=seed, stage_index=stage, current=[DOMAINS[stage]], history=list(DOMAINS[:stage]),
                historical_transform='identity', source='B0-EMA train_labeled original fits', K=2, ordered_records_sha256=H(records))


def class_scores(directions, centers, valid=None):
    """Mask missing centers without treating a zero placeholder as a center."""
    centers = np.asarray(centers, dtype=np.float64)
    require(centers.ndim == 3 and centers.shape[0] == 3 and centers.shape[2] == 16, 'bank shape')
    valid = np.ones(centers.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    require(valid.shape == centers.shape[:2], 'bank mask shape'); finite(directions, centers)
    require(np.all(np.abs(np.linalg.norm(centers[valid], axis=1)-1) <= 1e-12), 'non-unit active center')
    scores = np.full((len(directions), 3), -np.inf); maximum = np.full_like(scores, -np.inf)
    for c in range(3):
        if valid[c].any():
            cos = np.clip(directions @ centers[c, valid[c]].T, -1, 1)
            scores[:, c] = logsumexp(cos/.07, axis=1)-np.log(valid[c].sum())
            maximum[:, c] = cos.max(axis=1)
    return scores, maximum


def margin(scores, predicted):
    rows = np.arange(len(predicted)); same = scores[rows, predicted]
    competitors = scores.copy(); competitors[rows, predicted] = -np.inf
    other = competitors.max(axis=1); valid = np.isfinite(same) & np.isfinite(other)
    result = np.full(len(predicted), np.nan)
    result[valid] = same[valid]-other[valid]
    return result, valid


def score_arrays(probability, raw_features, joint_pas, current, history, *, current_valid=None, history_valid=None):
    """No labels, cases or evaluator state enter this API."""
    p = np.asarray(probability, dtype=np.float64); z = np.asarray(raw_features)
    require(p.ndim == 2 and p.shape[1] == 3 and z.shape == (len(p), 16), 'probability/feature shape')
    finite(p, z)
    require(np.all(p >= 0) and np.all(p <= 1) and np.allclose(p.sum(axis=1), 1, atol=2e-7, rtol=0), 'invalid probability')
    joint = np.asarray(joint_pas)
    require(joint.shape == (len(p),) and np.all((joint == 0) | (joint == 1)), 'PAS must be binary')
    try:
        support = split_support(z)
    except NonfiniteFeature as error:
        raise NonfiniteEvidence(str(error)) from error
    active = support['active_mask']; predicted = p.argmax(axis=1); q = p.max(axis=1)
    ac = np.full((len(p), 3), np.nan); ah = np.full_like(ac, np.nan)
    mc = np.full(len(p), np.nan); mh = np.full(len(p), np.nan)
    sh = np.full(len(p), np.nan); gate = np.full(len(p), np.nan)
    r2 = np.zeros(len(p)); r3 = np.zeros(len(p)); proto_valid = np.zeros(len(p), bool)
    idx = np.flatnonzero(active); u = support['directions'][active]
    cur, _ = class_scores(u, current, current_valid)
    mcur, valid = margin(cur, predicted[active]); ac[active] = cur; mc[active] = mcur; proto_valid[active] = valid
    r2[idx[valid]] = q[idx[valid]]*expit(mcur[valid]/.10)
    r3[:] = r2
    gate[active] = 0.0
    if np.asarray(history).shape[1]:
        hist, hmax = class_scores(u, history, history_valid)
        m, hv = margin(hist, predicted[active]); ah[active] = hist; mh[active] = m
        sim = hmax[np.arange(len(idx)), predicted[active]]
        sh[idx[hv]] = sim[hv]; g = np.zeros(len(idx)); g[hv] = expit((sim[hv]-.30)/.10)
        gate[active] = g
        r3[idx[hv]] *= (1-g[hv])+g[hv]*expit(m[hv]/.10)
    result = dict(teacher_probability=p.astype(np.float32), R0=q.astype(np.float32), R1=joint.astype(bool), R2=r2, R3=r3,
        raw_norms=support['raw_norms'], active_mask=active, prototype_valid=proto_valid,
        current_scores=ac, history_scores=ah, current_margin=mc, history_margin=mh, history_similarity=sh, history_gate=gate)
    finite(*(result[k] for k in ('teacher_probability', 'R0', 'R1', 'R2', 'R3', 'raw_norms')))
    require(np.all((r3 >= 0) & (r3 <= r2)) and np.all(r2 <= q), 'invalid reliability bounds')
    require(np.all(r2[~active] == 0) and np.all(r3[~active] == 0), 'null feature received prototype weight')
    return result


@torch.no_grad()
def legacy_pas(student_logits, student_features, teacher_logits, teacher_features, legacy_prototypes):
    finite(student_logits, student_features, teacher_logits, teacher_features, legacy_prototypes)
    s = compute_pas_validity(student_logits, student_features, legacy_prototypes, .7, .7)
    t = compute_pas_validity(teacher_logits, teacher_features, legacy_prototypes, .7, .7)
    return s.valid_mask & t.valid_mask


@torch.no_grad()
def build(student_logits, student_features, teacher_logits, teacher_features, legacy_prototypes, current, history):
    joint = legacy_pas(student_logits, student_features, teacher_logits, teacher_features, legacy_prototypes)
    finite(teacher_logits, teacher_features)
    p = teacher_logits.float().softmax(1).permute(0, 2, 3, 1).cpu().numpy().reshape(-1, 3)
    z = teacher_features.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, 16)
    return score_arrays(p, z, joint.cpu().numpy().reshape(-1), current, history)


def poe_target(scores):
    """Control changes only the target; fixed R3 weights and null exclusions."""
    p = np.asarray(scores['teacher_probability'], np.float64)
    valid = scores['active_mask'] & scores['prototype_valid']
    target = np.zeros_like(p)
    for_slice = np.flatnonzero(valid)
    if len(for_slice):
        logp = np.full_like(p[valid], -np.inf)
        np.log(p[valid], out=logp, where=p[valid] > 0)
        cur = scores['current_scores'][valid]
        logcur = cur-logsumexp(cur, axis=1, keepdims=True)
        fused = logp+.5*logcur
        hist = scores['history_scores'][valid]; g = scores['history_gate'][valid]
        usable = np.isfinite(hist).any(axis=1) & (g > 0)
        if usable.any():
            loghist = hist[usable]-logsumexp(hist[usable], axis=1, keepdims=True)
            fused[usable] += (.25*g[usable, None])*loghist
        target[valid] = softmax(fused, axis=1)
    finite(target)
    weights = np.where(valid, scores['R3'], 0.)
    return dict(probability=target, weights=weights, valid=valid, prediction=target.argmax(axis=1))
