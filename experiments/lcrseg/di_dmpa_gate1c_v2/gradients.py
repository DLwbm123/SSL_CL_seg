"""autograd.grad-only probes; no parameter.grad writes and no optimizer."""
import numpy as np
import torch
from torch.nn import functional as F

from .binding import require, finite, GradientPartitionError, array_hash

BLOCKS = {'encoder': ('enc1.', 'enc2.', 'enc3.'), 'bottleneck': ('bottleneck.',),
          'dec3': ('decoder.dec3.',), 'dec2': ('decoder.dec2.',), 'dec1': ('decoder.dec1.',),
          'classifier.mu': ('decoder.conv_logit.mu.',)}
INACTIVE = {'decoder.conv_logit.sigma.weight', 'decoder.conv_logit.grad_update'}
NORMALIZATIONS = ('pixel_normalized', 'class_balanced')


def partition(student):
    names = []; params = []; active = []; blocks = {k: [] for k in BLOCKS}; inventory = []
    for i, (name, parameter) in enumerate(student.named_parameters()):
        names.append(name); params.append(parameter)
        matches = [b for b, prefixes in BLOCKS.items() if name.startswith(prefixes)]
        inactive = name in INACTIVE
        if (inactive and matches) or (not inactive and len(matches) != 1):
            raise GradientPartitionError(f'unpartitioned/overlapping active parameter: {name}')
        inventory.append(dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
            active=not inactive, block=matches[0] if matches else None, expected_gradient='None' if inactive else 'Tensor'))
        if not inactive:
            blocks[matches[0]].append(i); active.append(i)
    if not INACTIVE.issubset(names) or any(not indices for indices in blocks.values()):
        raise GradientPartitionError('missing official inactive inventory or active block')
    require(all(p.requires_grad for p in params), 'student autograd must include inactive parameters to verify None')
    require(all(p.grad is None for p in params), 'parameter.grad was populated')
    return dict(names=names, params=params, active=active, blocks=blocks, inventory=inventory)


def objective(probability, target, weights, predicted, normalization, *, class_component=None):
    """Per-class components add as vectors, not as scalar gradient norms."""
    require(normalization in NORMALIZATIONS, 'unknown normalization')
    target = target.detach(); weights = weights.detach(); predicted = predicted.detach()
    finite(probability, target, weights)
    require(probability.shape == target.shape and weights.shape == predicted.shape == probability.shape[:1]+probability.shape[2:], 'consistency shapes')
    require(not target.requires_grad and not weights.requires_grad and bool((weights >= 0).all()), 'target/weight detach')
    loss = (probability-target).square().sum(1)
    if float(weights.sum()) == 0:
        return probability.sum()*0.0
    if normalization == 'pixel_normalized':
        numerator = weights*loss
        if class_component is not None:
            numerator = numerator*(predicted == class_component)
        return numerator.sum()/(weights.sum()+1e-12)
    classes = [c for c in range(3) if float(weights[predicted == c].sum()) > 0]
    if not classes:
        return probability.sum()*0.0
    selected = classes if class_component is None else [c for c in classes if c == class_component]
    terms = [(weights[predicted == c]*loss[predicted == c]).sum()/(weights[predicted == c].sum()+1e-12) for c in selected]
    return sum(terms)/len(classes) if terms else probability.sum()*0.0


def grad(loss, parts, *, retain=True):
    finite(loss)
    values = torch.autograd.grad(loss, parts['params'], retain_graph=retain, create_graph=False, allow_unused=True)
    for name, value in zip(parts['names'], values):
        if (value is None) != (name in INACTIVE):
            raise GradientPartitionError(f'unexpected active/None gradient: {name}')
        if value is not None:
            finite(value)
    require(all(p.grad is None for p in parts['params']), 'autograd wrote parameter.grad')
    return tuple(None if g is None else g.detach() for g in values)


def vectors(values, parts):
    flat = {b: torch.cat([values[i].reshape(-1) for i in indices]).double().cpu().numpy()
            for b, indices in parts['blocks'].items()}
    flat['global'] = np.concatenate([flat[b] for b in BLOCKS])
    return flat


def alignment(supervised, unsupervised):
    finite(supervised, unsupervised)
    sn = float(np.linalg.norm(supervised)); un = float(np.linalg.norm(unsupervised))
    zero = sn == 0 or un == 0
    cosine = None if zero else float(np.clip(np.dot(supervised, unsupervised)/(sn*un), -1, 1))
    return dict(cosine=cosine, zero_gradient=zero, supervised_zero=sn == 0, unsupervised_zero=un == 0,
        supervised_norm=sn, unsupervised_norm=un, norm_ratio=None if sn == 0 else .5*un/sn,
        negative_cosine=None if zero else cosine < 0,
        undefined_reason='ZERO_GRADIENT_NORM' if zero else None)


def consistency_gradients(student_probability, target_probability, scores, parts, supervised_vectors,
                          *, candidates=('R0', 'R1', 'R2', 'R3'), draw=0, teacher_kind='stochastic', decompose=False, context=None):
    context = dict(context or {}); device = student_probability.device
    shape = student_probability.shape[:1]+student_probability.shape[2:]
    target = torch.as_tensor(target_probability, device=device, dtype=student_probability.dtype)
    require(target.shape == student_probability.shape, 'target probability geometry')
    predicted = target.detach().argmax(1)
    rows = []; components = []; hashes = {}
    for candidate in candidates:
        require(candidate in ('R0', 'R1', 'R2', 'R3', 'PoE'), 'unknown/R4 candidate')
        weights = torch.as_tensor(np.array(scores[candidate], copy=True), device=device, dtype=torch.float64).reshape(shape).detach()
        for normalization in NORMALIZATIONS:
            loss = objective(student_probability, target, weights, predicted, normalization)
            gradient = grad(loss, parts); vector = vectors(gradient, parts)
            hashes[f'{candidate}/{normalization}'] = array_hash(vector['global'])
            identity = dict(context, candidate=candidate, normalization=normalization, draw_index=draw, teacher_kind=teacher_kind)
            for block in ('global', *BLOCKS):
                rows.append(dict(identity, block=block, loss=float(loss.detach()), **alignment(supervised_vectors[block], vector[block])))
            if decompose:
                class_vectors = [vectors(grad(objective(student_probability, target, weights, predicted, normalization, class_component=c), parts), parts) for c in range(3)]
                for block in ('global', *BLOCKS):
                    total = vector[block]; partsum = sum(v[block] for v in class_vectors)
                    residual = float(np.max(np.abs(total-partsum))) if len(total) else 0.
                    require(np.allclose(total, partsum, atol=1e-6, rtol=1e-4), 'class gradient decomposition outside preregistered tolerance')
                    dots = [[float(np.dot(class_vectors[c][block], class_vectors[k][block])) for k in range(3)] for c in range(3)]
                    total_norm2 = float(np.dot(total, total))
                    for c in range(3):
                        v = class_vectors[c][block]; dot_total = float(np.dot(v, total))
                        components.append(dict(identity, block=block, class_id=c,
                            **alignment(supervised_vectors[block], v), dot_background=dots[c][0], dot_rim=dots[c][1], dot_cup=dots[c][2],
                            dot_total=dot_total, projection_fraction=dot_total/total_norm2 if total_norm2 else None,
                            component_vector_sha256=array_hash(v), total_vector_sha256=array_hash(total),
                            component_sum_max_abs_error=residual, component_sum_pass=True,
                            nonadditive_norm_sum=float(sum(np.linalg.norm(x[block]) for x in class_vectors)), total_norm=float(np.linalg.norm(total))))
                del class_vectors
            del gradient, vector, loss
    return rows, components, hashes


def supervised_gradient(student_logits, labels, parts):
    require(labels.shape == student_logits.shape[:1]+student_logits.shape[2:] and bool((labels != 255).any()), 'labeled reference shape/support')
    finite(student_logits)
    # PyTorch 2.2 CUDA NLLLoss2d's fused mean uses atomicAdd; keep strict determinism.
    per_pixel = F.cross_entropy(student_logits, labels, ignore_index=255, reduction='none')
    loss = per_pixel.sum() / (labels != 255).sum()
    values = grad(loss, parts, retain=False)
    return float(loss.detach()), vectors(values, parts)


def isolation(models, legacy, banks_before, current, history):
    require(all(p.grad is None for m in models.values() for p in m.parameters()), 'parameter.grad must remain None')
    require(all(not p.requires_grad for p in models['ema_teacher'].parameters()), 'teacher autograd enabled')
    require(not legacy.requires_grad and legacy.grad is None, 'legacy prototype gradient')
    require(banks_before == (array_hash(current), array_hash(history)), 'prototype/history bank changed')
    return dict(teacher_gradients='None', prototype_gradients='None', history_bank_gradients='None',
                student_parameter_grad_fields='None', optimizer_constructed=False, backward_called=False,
                model_optimizer_steps=0, transport_optimizer_steps_this_gate=0)


def summary(values):
    defined = [x for x in values if x is not None]
    finite(np.asarray(defined))
    # Required comparisons never use this descriptive subset if any row is undefined.
    return dict(count=len(values), defined_count=len(defined), undefined_count=len(values)-len(defined),
        median=float(np.median(defined)) if defined else None,
        p10=float(np.quantile(defined, .1, method='linear')) if defined else None,
        p90=float(np.quantile(defined, .9, method='linear')) if defined else None,
        population_variance=float(np.var(values, ddof=0)) if values and len(defined) == len(values) else None,
        required_comparison_defined=len(defined) == len(values), values=values)
