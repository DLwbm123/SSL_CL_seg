"""Frozen image-only forwards and fixed-pair execution, with evaluator GT separate."""
from pathlib import Path

import h5py
import numpy as np
import torch

from di_dmpa_gate1.feature_extraction import _images, seed_after_load
from di_dmpa_gate1_v2.features import ImmutableModels
from .binding import (require, finite, complete, H, sha256, check_hash, write_json, read_json, safe_asset,
    checkpoint, load_b0, records, image_only, no_updates, tensor_hash, array_hash, save_arrays, read_arrays)
from .reliability import banks, bank_identity, build, score_arrays, poe_target, CANDIDATES
from .gradients import partition, supervised_gradient, consistency_gradients, isolation
from .metrics import evaluate
from .precision import attach_gradient_student, student_forward, compare, comparable

CACHE_FIELDS = ('teacher_probability', 'R0', 'R1', 'R2', 'R3', 'raw_norms', 'active_mask',
                'prototype_valid', 'current_scores', 'history_scores', 'history_gate')


def unit_name(seed, stage):
    return f'seed{seed}_stage{stage}'


def pair_name(pair):
    return f'seed{pair["seed"]}_stage{pair["stage_index"]}_pair{pair["pair_index"]:02d}'


def visible_labels(rows, data_root, *, role):
    require(role in ('train_labeled', 'val'), 'hidden/test GT access forbidden')
    values = []
    for row in rows:
        require(row['primary_20pct_split'] == role and row['dataset'] == 'fundus', 'visible label role mismatch')
        path = safe_asset(data_root, row['label_h5_relpath']); check_hash(path, row['label_sha256'])
        with h5py.File(path, 'r') as handle:
            label = handle['label'][...]
        require(label.shape == (384, 384) and set(np.unique(label)).issubset({0, 1, 2, 255}), 'label geometry/mapping changed')
        values.append(label.astype(np.int64))
    return values


def validate_scores(a, stage, expected_count):
    require(set(a) == set(CACHE_FIELDS), 'reliability cache fields changed')
    n = expected_count
    require(a['teacher_probability'].shape == (n, 3) and a['current_scores'].shape == (n, 3) and a['history_scores'].shape == (n, 3), 'cache geometry')
    require(all(a[k].shape == (n,) for k in CACHE_FIELDS if k not in ('teacher_probability', 'current_scores', 'history_scores')), 'missing scalar cache row')
    finite(*(a[k] for k in ('teacher_probability', 'R0', 'R1', 'R2', 'R3', 'raw_norms')))
    active = a['active_mask']; require(active.dtype == bool and a['prototype_valid'].dtype == bool and a['R1'].dtype == bool, 'cache boolean dtype')
    require(np.array_equal(active, a['raw_norms'] > 1e-12) and np.all(a['raw_norms'] >= 0), 'null/support mismatch')
    require(np.all(a['R2'][~active] == 0) and np.all(a['R3'][~active] == 0) and not a['prototype_valid'][~active].any(), 'null dropped/weighted')
    for k in ('current_scores', 'history_scores', 'history_gate'):
        require(np.isnan(a[k][~active]).all(), 'null score/gate must be structurally unavailable')
    finite(a['current_scores'][active], a['history_gate'][active])
    if stage:
        finite(a['history_scores'][active])
    else:
        require(np.isnan(a['history_scores']).all() and np.array_equal(a['R3'], a['R2']) and np.all(a['history_gate'][active] == 0), 'stage0 identity violated')
    require(np.all(a['R3'] >= 0) and np.all(a['R3'] <= a['R2']) and np.all(a['R2'] <= a['R0'].astype(np.float64)), 'cache reliability bound')
    p = a['teacher_probability']; finite(p)
    require(np.all(p >= 0) and np.all(p <= 1) and np.allclose(p.sum(1), 1, atol=2e-7, rtol=0), 'cache probability')
    require(np.array_equal(a['R0'], p.max(1)), 'confidence formula changed')
    return dict(rows=n, active=int(active.sum()), null=int((~active).sum()), all_null_rows_preserved=True)


def validation_unit(root, data_root, p, freeze, metadata, seed, stage, output, device, *, case_ids=None):
    output = Path(output); cp = checkpoint(p, seed, stage)
    rows = records(data_root, p, seed, stage, 'val')
    plan = next(u for u in p['validation']['plans'] if (u['seed'], u['stage_index']) == (seed, stage))
    if case_ids is not None:
        rows = [r for r in rows if r['case_id'] in case_ids]
        require({r['case_id'] for r in rows} == set(case_ids), 'integration case outside current val')
    models, legacy = load_b0(root, p, seed, stage, device); current, history = banks(freeze, seed, stage)
    bank_before = (array_hash(current), array_hash(history)); legacy_before = tensor_hash(legacy)
    context = dict(metadata, seed=seed, stage_index=stage, role='val', bank=bank_identity(freeze, seed, stage), legacy_prototypes_sha256=legacy_before)
    cases = []
    with no_updates(), ImmutableModels(models, cp, output/'validation_models'/unit_name(seed, stage), context), torch.no_grad():
        for case_index, row in enumerate(rows):
            item = next(x for x in plan['cases'] if x['case_id'] == row['case_id'])
            images = _images([image_only(row)], data_root).to(device)
            seed_after_load(item['teacher_draw0_seed']); tl, tf = models['ema_teacher'](images, stochastic_classifier=True)
            seed_after_load(item['student_seed']); sl, sf = models['student'](images, stochastic_classifier=True)
            scores = build(sl, sf, tl, tf, legacy, current, history)
            arrays = {k: scores[k] for k in CACHE_FIELDS}; support = validate_scores(arrays, stage, 384*384)
            desc = save_arrays(output/'validation_cache'/unit_name(seed, stage)/(row['case_id']+'.npz'), arrays)
            null_xy = np.argwhere(~scores['active_mask'].reshape(384, 384))
            cases.append(dict(case_id=row['case_id'], teacher_draw0_seed=item['teacher_draw0_seed'], student_seed=item['student_seed'],
                image_sha256=row['image_sha256'], arrays=desc, support=support, first_null_coordinates=null_xy[:32].tolist(),
                student_logits_sha256=tensor_hash(sl), student_features_sha256=tensor_hash(sf),
                teacher_logits_sha256=tensor_hash(tl), teacher_features_sha256=tensor_hash(tf), GT_received_by_builder=False))
            if (case_index+1) % 10 == 0 or case_index+1 == len(rows):
                print(f'validation cached {unit_name(seed, stage)} {case_index+1}/{len(rows)}', flush=True)
        iso = isolation(models, legacy, bank_before, current, history)
        require(tensor_hash(legacy) == legacy_before, 'legacy PAS prototypes changed')
    result = dict(metadata=context, seed=seed, stage_index=stage, cases=cases, case_count=len(cases),
        read_only=True, all_scores_finite_or_structural_null=True, isolation=iso, current_domain_only=True,
        checkpoint_sha256=cp['sha256'], legacy_prototypes_sha256_after=tensor_hash(legacy))
    write_json(output/'validation_units'/(unit_name(seed, stage)+'.json'), result)
    del models, legacy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_unit(task):
    data_root, p, output, seed, stage, include_poe = task
    output = Path(output); entry = read_json(output/'validation_units'/(unit_name(seed, stage)+'.json'))
    rows = records(data_root, p, seed, stage, 'val')
    require([r['case_id'] for r in rows] == [c['case_id'] for c in entry['cases']], 'evaluator/cache case mismatch')
    caches = [read_arrays(c['arrays']) for c in entry['cases']]
    for a in caches:
        validate_scores(a, stage, 384*384)
    labels = visible_labels(rows, data_root, role='val')
    result = evaluate(seed, stage, [r['case_id'] for r in rows], caches, labels, include_poe=include_poe)
    result['metadata'] = entry['metadata']; result['validation_unit_sha256'] = sha256(output/'validation_units'/(unit_name(seed, stage)+'.json'))
    folder = 'poe_validation' if include_poe else 'reliability_units'
    write_json(output/folder/(unit_name(seed, stage)+'.json'), result)
    print(f'validation evaluator complete {unit_name(seed, stage)} poe={include_poe}', flush=True)
    return dict(seed=seed, stage_index=stage, complete=True)


def pair_inputs(data_root, p, pair):
    seed, stage = pair['seed'], pair['stage_index']
    require(pair in p['gradient_diagnostic']['batch_pairs'], 'unregistered gradient pair')
    labeled = {r['case_id']: r for r in records(data_root, p, seed, stage, 'train_labeled')}
    unlabeled = {r['case_id']: r for r in records(data_root, p, seed, stage, 'train_unlabeled')}
    lr = [labeled[k] for k in pair['labeled_case_ids']]; ur = [unlabeled[k] for k in pair['unlabeled_case_ids']]
    require(len(lr) == len(ur) == 2, 'fixed batch size changed')
    images_u = _images([image_only(r) for r in ur], data_root)
    images_l = _images([image_only(r) for r in lr], data_root)
    labels = torch.from_numpy(np.stack(visible_labels(lr, data_root, role='train_labeled')))
    return images_u, images_l, labels


def _prob_nchw(flat, shape):
    b, _, h, w = shape
    return flat.reshape(b, h, w, 3).transpose(0, 3, 1, 2)


def gradient_pair(models, legacy, current, history, p, pair, data_root, output, metadata, *, phase, device):
    require(phase in ('draw0', 'noise', 'posterior', 'poe'), 'unknown probe phase')
    output = Path(output); name = pair_name(pair); directory = output/'probes'/phase/name
    directory.mkdir(parents=True, exist_ok=False)
    models['student'].requires_grad_(True)
    shadow = models.get('gradient_student')
    require((shadow is not None) == (p.get('diagnostic_precision') == 'float64_shadow'), 'gradient receiver/mode mismatch')
    parts = partition(models['student'] if shadow is None else shadow)
    xu, xl, labels = pair_inputs(data_root, p, pair); xu, xl, labels = xu.to(device), xl.to(device), labels.to(device)
    sl, sf, gradient_sl, unlabeled_draw = student_forward(models, xu, pair['forward_seeds']['student_unlabeled'])
    finite(sl, sf); native_probability = sl.float().softmax(1)
    probability = native_probability if shadow is None else gradient_sl.softmax(1)
    primary = None; cached = None
    if phase == 'draw0':
        seed_after_load(pair['teacher_draw_seeds'][0])
        with torch.no_grad():
            tl, tf = models['ema_teacher'](xu, stochastic_classifier=True)
        finite(tl, tf)
    else:
        primary = read_json(output/'probes/draw0'/name/'result.json'); cached = read_arrays(primary['primary_cache'])
        require(primary['pair'] == pair and primary['metadata'] == metadata, 'primary probe provenance mismatch')
        require(tensor_hash(sl) == primary['student_logits_sha256'] and tensor_hash(sf) == primary['student_features_sha256'], 'fixed student forward changed')
        require(np.array_equal(sl.detach().cpu().numpy(), cached['student_logits']), 'student logits not bitwise equal to cached forward')
        tl = torch.as_tensor(cached['teacher_logits_draw0'], device=device)
        tf = torch.as_tensor(cached['teacher_features'], device=device)
    ll, lf, gradient_ll, labeled_draw = student_forward(models, xl, pair['forward_seeds']['student_labeled'])
    finite(ll, lf); supervised_loss, supervised = supervised_gradient(gradient_ll, labels, parts)
    native_reference = None; supervised_comparisons = None
    if shadow is not None:
        native_parts = partition(models['student'])
        native_supervised_loss, native_supervised = supervised_gradient(ll, labels, native_parts)
        native_reference = dict(probability=native_probability, parts=native_parts, supervised=native_supervised, comparisons=[])
        supervised_comparisons = {block: compare(native_supervised[block], supervised[block]) for block in native_supervised}
    sup_sha = array_hash(supervised['global'])
    if primary is not None:
        require(sup_sha == primary['supervised_gradient_sha256'] and tensor_hash(ll) == primary['labeled_logits_sha256'], 'fixed supervised forward/gradient changed')
    context = {k: pair[k] for k in ('batch_id', 'seed', 'stage_index', 'domain', 'pair_index')}
    rows = []; components = []; gradient_hashes = {}; weight_cache = {k: [] for k in CANDIDATES}; changed = []
    original_teacher_feature_sha = tensor_hash(tf)
    result = dict(metadata=metadata, pair=pair, phase=phase, student_logits_sha256=tensor_hash(sl),
        student_features_sha256=tensor_hash(sf), labeled_logits_sha256=tensor_hash(ll),
        teacher_features_sha256=original_teacher_feature_sha, supervised_loss=supervised_loss, supervised_gradient_sha256=sup_sha,
        parameter_inventory=parts['inventory'], fixed_student_cache_verified=primary is not None,
        no_optimizer=True, no_backward=True, no_parameter_grad_writes=True)
    if native_reference is not None:
        result.update(diagnostic_precision='float64_shadow', student_draw_replay=dict(unlabeled=unlabeled_draw, labeled=labeled_draw),
            native_student_probability_sha256=tensor_hash(native_probability),
            native_supervised_loss=native_supervised_loss, native_supervised_gradient_sha256=array_hash(native_supervised['global']),
            supervised_precision_comparisons=supervised_comparisons,
            supervised_precision_comparable=comparable(supervised_comparisons['global']))
        if primary is not None:
            require(primary['student_draw_replay'] == result['student_draw_replay'] and
                    primary['native_supervised_gradient_sha256'] == result['native_supervised_gradient_sha256'], 'fixed native/shadow forward changed')
    if phase == 'poe':
        noise = read_json(output/'probes/noise'/name/'result.json')
        teacher_draws = read_arrays(noise['teacher_cache'])['teacher_probabilities']
        require(teacher_draws.shape[0] == 8 and noise['teacher_draw_seeds'] == pair['teacher_draw_seeds'], 'teacher draw cache mismatch')
        raw = cached['teacher_features'].transpose(0, 2, 3, 1).reshape(-1, 16)
        targets = []; available_for_draws = None
        for draw in range(8):
            pflat = teacher_draws[draw].transpose(0, 2, 3, 1).reshape(-1, 3)
            scores = score_arrays(pflat, raw, np.zeros(len(raw), bool), current, history)
            control = poe_target(scores); target = _prob_nchw(control['probability'], sl.shape)
            if available_for_draws is None:
                available_for_draws = control['valid']
            else:
                require(np.array_equal(available_for_draws, control['valid']), 'PoE directional support changed across draws')
            rr, cc, hh = consistency_gradients(probability, target, {'PoE': control['weights']}, parts, supervised,
                candidates=('PoE',), draw=draw, teacher_kind='stochastic', decompose=draw == 0, context=context, native_reference=native_reference)
            rows.extend(rr); components.extend(cc); gradient_hashes[str(draw)] = hh; targets.append(target)
            available = control['valid']; pred_t = pflat.argmax(1)
            changed.append(dict(draw_index=draw, available_count=int(available.sum()), null_count=int((~scores['active_mask']).sum()),
                changed_prediction_rate=float(np.mean(control['prediction'][available] != pred_t[available])) if available.any() else None,
                target_sha256=array_hash(target)))
        own_predictions = np.stack(targets).argmax(2).reshape(8, -1)[:, available_for_draws]
        result.update(teacher_draw_seeds=pair['teacher_draw_seeds'], teacher_forwards=0, cached_teacher_draws_used=8,
            target_probability_variance=float(np.var(np.stack(targets).astype(np.float64), axis=0, ddof=0).mean()),
            predicted_class_change_rate=float(np.mean(own_predictions != own_predictions[0])) if own_predictions.size else None,
            any_draw_class_change_rate=float(np.mean(np.any(own_predictions != own_predictions[0], axis=0))) if own_predictions.size else None,
            changed_predictions=changed, same_detached_R3_weight=True, own_predicted_class_strata=True)
    else:
        draws = range(8) if phase == 'noise' else [0]
        teacher_probabilities = []
        for draw in draws:
            if phase == 'posterior':
                with torch.no_grad():
                    tl, tf = models['ema_teacher'](xu, stochastic_classifier=False)
            elif phase == 'noise' and draw:
                seed_after_load(pair['teacher_draw_seeds'][draw])
                with torch.no_grad():
                    tl, tf = models['ema_teacher'](xu, stochastic_classifier=True)
            finite(tl, tf)
            require(tensor_hash(tf) == original_teacher_feature_sha, 'classifier draw changed EMA feature')
            scores = build(sl.detach(), sf.detach(), tl, tf, legacy, current, history)
            target = tl.float().softmax(1).detach(); target_np = target.cpu().numpy()
            teacher_probabilities.append(target_np)
            for k in CANDIDATES:
                weight_cache[k].append(scores[k])
            if phase == 'noise' and draw == 0:
                rr, cc, hh = primary['alignment'], [], primary['gradient_hashes']['0']
                require(array_hash(target_np) == primary['teacher_probability_sha256'], 'draw0 cached target changed')
                if native_reference is not None:
                    native_reference['comparisons'].extend(primary['native_precision_comparisons'])
            else:
                rr, cc, hh = consistency_gradients(probability, target, scores, parts, supervised, draw=draw,
                    teacher_kind='posterior_mean' if phase == 'posterior' else 'stochastic', decompose=phase == 'draw0', context=context, native_reference=native_reference)
            rows.extend(rr); components.extend(cc); gradient_hashes[str(draw)] = hh
            if phase == 'draw0':
                result['primary_cache'] = save_arrays(directory/'primary_cache.npz', dict(student_logits=sl.detach().cpu().numpy(),
                    teacher_logits_draw0=tl.cpu().numpy(), teacher_features=tf.cpu().numpy()))
                result['teacher_probability_sha256'] = array_hash(target_np)
                result['R1_validity_sha256'] = array_hash(scores['R1'])
                result['null_ema_count'] = int((~scores['active_mask']).sum())
                result['R2_R3_exact_equal'] = bool(np.array_equal(scores['R2'], scores['R3']))
                result['teacher_draw_seed'] = pair['teacher_draw_seeds'][0]
        if phase == 'noise':
            stack = np.stack(teacher_probabilities); preds = stack.argmax(2)
            result.update(teacher_draw_seeds=pair['teacher_draw_seeds'], primary_draw0_reused=True, teacher_forwards=7,
                teacher_cache=save_arrays(directory/'teacher_cache.npz', dict(teacher_probabilities=stack)),
                target_probability_variance=float(np.var(stack.astype(np.float64), axis=0, ddof=0).mean()),
                predicted_class_change_rate=float(np.mean(preds != preds[0])),
                any_draw_class_change_rate=float(np.mean(np.any(preds != preds[0], axis=0))),
                weight_variance={k: float(np.var(np.stack(v).astype(np.float64), axis=0, ddof=0).mean()) for k, v in weight_cache.items()})
        elif phase == 'posterior':
            result.update(teacher_forwards=1, teacher_stochastic_classifier=False, baseline_replacement=False,
                teacher_probability_sha256=array_hash(teacher_probabilities[0]),
                target_difference_from_draw0=float(np.mean((teacher_probabilities[0]-torch.as_tensor(cached['teacher_logits_draw0']).softmax(1).numpy())**2)))
        else:
            result['teacher_forwards'] = 1
    result.update(alignment=rows, class_contribution=components, gradient_hashes=gradient_hashes)
    if native_reference is not None:
        result['native_precision_comparisons'] = native_reference['comparisons']
    require(all(q.grad is None for m in models.values() for q in m.parameters()), 'parameter.grad mutated')
    write_json(directory/'result.json', result)
    return result


def probe_unit(root, data_root, p, freeze, metadata, seed, stage, output, device, phase, *, pair_indices=None):
    output = Path(output); cp = checkpoint(p, seed, stage)
    models, legacy = load_b0(root, p, seed, stage, device); current, history = banks(freeze, seed, stage)
    before = (array_hash(current), array_hash(history)); legacy_before = tensor_hash(legacy)
    pairs = [q for q in p['gradient_diagnostic']['batch_pairs'] if (q['seed'], q['stage_index']) == (seed, stage)]
    if pair_indices is not None:
        pairs = [q for q in pairs if q['pair_index'] in pair_indices]
    models['student'].requires_grad_(True)
    attach_gradient_student(models, p)
    with no_updates():
        for pair in pairs:
            context = dict(metadata, phase=phase, pair_id=pair['batch_id'], bank=bank_identity(freeze, seed, stage), legacy_prototypes_sha256=legacy_before)
            guard_path = output/'probe_models'/phase/pair_name(pair)
            with ImmutableModels(models, cp, guard_path, context):
                try:
                    result = gradient_pair(models, legacy, current, history, p, pair, data_root, output, metadata, phase=phase, device=device)
                finally:
                    iso = isolation(models, legacy, before, current, history)
                    require(tensor_hash(legacy) == legacy_before, 'legacy PAS prototypes changed')
                    write_json(output/'probes'/phase/pair_name(pair)/'isolation.json', dict(metadata=context, **iso,
                        legacy_prototypes_unchanged=True, current_history_banks_unchanged=True))
            print(f'probe complete {phase} {pair_name(pair)}', flush=True)
    models['student'].requires_grad_(False)
    del models, legacy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
