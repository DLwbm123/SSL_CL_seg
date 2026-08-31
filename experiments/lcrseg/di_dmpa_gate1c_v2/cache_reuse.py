"""Read-only v2.1 validation reuse, with original provenance kept verbatim."""
from pathlib import Path

import numpy as np

from . import binding as b, execution as e
from .reliability import bank_identity


def checked_file(path, digest, size=None):
    path = Path(path)
    b.check_hash(path, digest)
    b.require(size is None or path.stat().st_size == size, 'reference byte length changed')
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def recheck_references(receipt):
    b.require(receipt['status'] == 'PASS' and receipt['cache_reuse_approved'], 'unapproved cache reference')
    refs = receipt['references']
    b.require(len(refs) == len({r['path'] for r in refs}), 'duplicate external reference')
    for row in refs:
        checked_file(row['path'], row['sha256'], row['bytes'])


def derived_unit(original, source, old_meta, new_meta):
    """Only the new wrapper is new; its cases and source metadata are unchanged."""
    extra = {k: v for k, v in original['metadata'].items() if k not in old_meta}
    b.require(set(extra) == {'seed', 'stage_index', 'role', 'bank', 'legacy_prototypes_sha256'}, 'unknown source context')
    b.require(original['metadata'] == dict(old_meta, **extra), 'mixed original metadata')
    return dict(original, metadata=dict(new_meta, **extra), source_validation_unit=dict(
        source, original_metadata=original['metadata'], reused=True, new_validation_forwards=0))


def validate_derived(unit, original, source, old_meta, new_meta):
    b.require(unit == derived_unit(original, source, old_meta, new_meta), 'derived cache relabeled or changed')


def validate_case(case, row, plan_case, manifest_row, source_root, stage, *, pixels=384*384):
    b.require(case['case_id'] == row['case_id'] == plan_case['case_id'], 'cache case identity changed')
    b.require(case['image_sha256'] == row['image_sha256'] and not case['GT_received_by_builder'], 'cache image/GT source changed')
    b.require(all(case[k] == plan_case[k] for k in ('teacher_draw0_seed', 'student_seed')), 'cache forward seed changed')
    desc = case['arrays']; expected = Path(source_root)/manifest_row['path']
    b.require(Path(desc['path']) == expected and expected.resolve().is_relative_to(Path(source_root).resolve()), 'cache path escaped source')
    b.require(desc['sha256'] == manifest_row['sha256'] and desc['bytes'] == manifest_row['bytes'], 'cache descriptor/manifest mismatch')
    checked_file(expected, desc['sha256'], desc['bytes'])
    arrays = b.read_arrays(desc)
    expected_types = {k: 'float64' for k in e.CACHE_FIELDS}
    expected_types.update(teacher_probability='float32', R0='float32', R1='bool', active_mask='bool', prototype_valid='bool')
    b.require({k: str(a.dtype) for k, a in arrays.items()} == expected_types, 'cache dtype changed')
    support = e.validate_scores(arrays, stage, pixels)
    coordinates = np.argwhere(~arrays['active_mask'].reshape(384, 384)).tolist()[:32] if pixels == 384*384 else None
    b.require(support == case['support'] and (coordinates is None or coordinates == case['first_null_coordinates']), 'cache support/null census changed')
    return support


def validate_unit(original, guard, old_meta, seed, stage, cp, legacy_sha, bank):
    context = dict(old_meta, seed=seed, stage_index=stage, role='val', bank=bank, legacy_prototypes_sha256=legacy_sha)
    b.require(original['metadata'] == context and original['checkpoint_sha256'] == cp['sha256'] and
        original['legacy_prototypes_sha256_after'] == legacy_sha and original['seed'] == seed and
        original['stage_index'] == stage and original['read_only'] and original['current_domain_only'] and
        original['all_scores_finite_or_structural_null'], 'validation model/bank/context mismatch')
    iso = original['isolation']
    b.require(all(iso[k] == 'None' for k in ('teacher_gradients', 'prototype_gradients', 'history_bank_gradients', 'student_parameter_grad_fields')) and
        iso['model_optimizer_steps'] == iso['transport_optimizer_steps_this_gate'] == 0 and
        not iso['optimizer_constructed'] and not iso['backward_called'], 'original validation isolation failed')
    b.require(guard['metadata'] == context and guard['status'] == 'PASS' and guard['extraction_completed'] and
        guard['bitwise_unchanged'] and guard['before'] == guard['after'] and set(guard['before']) == {'student', 'ema_teacher'} and
        guard['checkpoint_id'] == cp['checkpoint_id'] and guard['checkpoint_sha256_before'] == guard['checkpoint_sha256_after'] == cp['sha256'],
        'original validation guard incomplete')
    return context


def audit_sources(root, data_root, spec, p, freeze, fresh_input):
    """Runs after published-code synthetic gates; zero forwards and no GT arrays."""
    root = Path(root); data_root = Path(data_root); repo = root.parents[1]
    reuse = spec['validation_reuse']; source = Path(reuse['source_root']); refs = {}

    def add(path, digest, size=None, *, retain=False):
        row = checked_file(path, digest, size)
        row['retain_private_copy'] = retain
        prior = refs.get(row['path'])
        b.require(prior is None or prior['sha256'] == row['sha256'], 'conflicting reference hashes')
        if prior is not None:
            row['retain_private_copy'] |= prior['retain_private_copy']
        refs[row['path']] = row
        return {k: row[k] for k in ('path', 'sha256', 'bytes')}

    manifest_path = source/reuse['source_artifact_manifest']
    add(manifest_path, reuse['source_artifact_manifest_sha256'], retain=True)
    manifest = b.read_json(manifest_path); index = {r['path']: r for r in manifest['artifacts']}
    b.require(len(index) == len(manifest['artifacts']), 'duplicate historical artifact')

    def original_json(relative):
        b.require(relative in index, 'unindexed original evidence')
        row = index[relative]; path = source/relative
        b.require(path.resolve().is_relative_to(source.resolve()), 'original path escaped source')
        desc = add(path, row['sha256'], row['bytes'], retain=True)
        return b.read_json(path), desc

    old_meta, _ = original_json('GATE1C_V2_RUN_METADATA.json')
    b.require(old_meta['diagnostic_code_commit'] == old_meta['remote_verified_code_commit'] == reuse['source_code_commit'] and
        old_meta['preregistration_commit'] == spec['input_amendment']['commit'] and
        old_meta['input_contract_version'] == 'v2.1' and old_meta['method_flags'] == p['method_flags'] and
        old_meta['primary_panel'] == 'B0-EMA' and old_meta['selected_K'] == 2 and
        old_meta['hidden_gt_training_usage'] == old_meta['test_gt_usage'] == 'none', 'wrong validation source provenance')
    originals = {}
    for name, key in (('RELIABILITY_CACHE_MANIFEST.json', 'cache_manifest_sha256'),
                      ('RELIABILITY_SUPPORT_CENSUS.json', 'support_census_sha256'),
                      ('VALIDATION_CACHE_BARRIER.json', 'validation_barrier_sha256'),
                      ('VALIDATION_MODEL_IMMUTABILITY_AUDIT.json', 'validation_model_audit_sha256')):
        value, desc = original_json(name)
        b.require(desc['sha256'] == reuse[key] and value['metadata'] == old_meta and value['status'] == 'PASS', 'historical validation barrier changed')
        originals[name] = value
    old_input, _ = original_json('GATE1C_V2_INPUT_AUDIT.json')
    b.require(old_input['metadata'] == old_meta and old_input['status'] == fresh_input['status'] == 'PASS' and
        old_input['legacy_payload_readiness'] == fresh_input['legacy_payload_readiness'] and
        old_input['checkpoints'] == fresh_input['checkpoints'] and old_input['units'] == fresh_input['units'] and
        old_input['hidden_gt_training_usage'] == old_input['test_gt_usage'] == 'none' and old_input['test_role_constructions'] == 0,
        'original/new input payload or roles differ')
    legacy = {r['checkpoint_id']: r['tensor_sha256'] for r in fresh_input['legacy_payload_readiness']['checkpoints']}
    expected_paths = {f'validation_cache/seed{u["seed"]}_stage{u["stage_index"]}/{c["case_id"]}.npz'
                      for u in p['validation']['plans'] for c in u['cases']}
    b.require(expected_paths == {name for name in index if name.startswith('validation_cache/')} and len(expected_paths) == 495,
        'missing/extra validation cache cases')
    for prefix, expected in (
        ('validation_units/', {f'validation_units/{e.unit_name(u["seed"],u["stage_index"])}.json' for u in p['validation']['plans']}),
        ('validation_models/', {f'validation_models/{e.unit_name(u["seed"],u["stage_index"])}/immutability/B0_seed{u["seed"]}_stage{u["stage_index"]}.json' for u in p['validation']['plans']})):
        b.require({n for n in index if n.startswith(prefix)} == expected, 'missing/extra original unit/guard')
    for cp in p['immutable_baseline']['checkpoint_inputs']:
        add(cp['path'], cp['sha256'], retain=True)
    for asset in p['benchmark']['manifest_assets']:
        add(asset['runtime_path'], asset['sha256'], retain=True)
        add(data_root/f'splits/fundus_seed{asset["seed"]}.json', asset['fundus_split_sha256'], retain=True)
    for record in freeze['prototype_records']:
        b.require(record['panel'] == 'B0-EMA' and record['K'] == 2, 'wrong prototype source')
        add(record['source_geometry_unit_remote_path'], record['source_file_sha256'], retain=True)
    recovered = p['legacy_prototype_reconstruction']
    add(recovered['bank_path'], recovered['bank_sha256'], retain=True)
    add(repo/recovered['recovery_comparison_path'], recovered['recovery_comparison_sha256'], retain=True)
    for record in (p['benchmark']['domain_order_source'], p['immutable_baseline']['configs']['B0']):
        add(repo/record['path'], record.get('sha256', record.get('file_sha256')), retain=True)
    add(repo/p['immutable_baseline']['freeze_path'], p['immutable_baseline']['freeze_sha256'], retain=True)
    units = []; guards = []; census = []; cache_bytes = 0; known_null = None
    for plan in p['validation']['plans']:
        seed, stage = plan['seed'], plan['stage_index']; name = e.unit_name(seed, stage)
        cp = b.checkpoint(p, seed, stage)
        rows_by_role = {role: b.records(data_root, p, seed, stage, role) for role in ('train_labeled', 'train_unlabeled', 'val')}
        for role, rows in rows_by_role.items():
            for row in rows:
                add(b.safe_asset(data_root, row['image_h5_relpath']), row['image_sha256'])
                if role != 'train_unlabeled':
                    add(b.safe_asset(data_root, row['label_h5_relpath']), row['label_sha256'])
        original, desc = original_json('validation_units/'+name+'.json')
        guard, guard_desc = original_json(f'validation_models/{name}/immutability/B0_seed{seed}_stage{stage}.json')
        context = validate_unit(original, guard, old_meta, seed, stage, cp, legacy[cp['checkpoint_id']], bank_identity(freeze, seed, stage))
        guards.append(dict(guard_desc, checkpoint_id=cp['checkpoint_id'], before=guard['before'], after=guard['after'],
                           original_metadata=context, reused=True, newly_executed=False))
        rows = rows_by_role['val']; cases = original['cases']
        b.require(len(rows) == len(cases) == original['case_count'] == len(plan['cases']) and
            [c['case_id'] for c in cases] == [c['case_id'] for c in plan['cases']], 'validation unit incomplete')
        historical = next(u for u in originals['RELIABILITY_CACHE_MANIFEST.json']['units'] if (u['seed'], u['stage_index']) == (seed, stage))
        b.require(historical['path'] == desc['path'] and historical['sha256'] == desc['sha256'] and historical['cases'] == cases,
            'cache index disagrees with original unit')
        for case, row, item in zip(cases, rows, plan['cases']):
            relative = f'validation_cache/{name}/{case["case_id"]}.npz'
            support = validate_case(case, row, item, index[relative], source, stage)
            add(case['arrays']['path'], case['arrays']['sha256'], case['arrays']['bytes'], retain=True)
            cache_bytes += case['arrays']['bytes']
            census.append(dict(seed=seed, stage_index=stage, case_id=case['case_id'], **support,
                first_null_coordinates=case['first_null_coordinates'], raw_all_pixel_null_fraction=support['null']/support['rows']))
            if (seed, stage, case['case_id']) == (2, 0, 'REFUGE_train_n0038'):
                b.require(support['null'] == 1 and case['first_null_coordinates'] == [[185, 180]], 'known real null changed')
                known_null = census[-1]
        units.append(dict(original=original, source=desc))
        print('verified original validation', name, len(cases), flush=True)
    old_census = originals['RELIABILITY_SUPPORT_CENSUS.json']
    b.require(len(units) == len(guards) == 9 and len(census) == 495 and census == old_census['cases'] and known_null is not None,
        'source support census changed')
    b.require(sum(c['rows'] for c in census) == reuse['pixels'] and cache_bytes == reuse['raw_cache_file_bytes'], 'source pixel/byte budget mismatch')
    receipt = dict(status='PASS', cache_reuse_approved=True, source_root=str(source), source_metadata=old_meta,
        source_code_commit=reuse['source_code_commit'], source_manifest_sha256=reuse['source_artifact_manifest_sha256'],
        units=[u['source'] for u in units], original_validation_guards=guards, references=list(refs.values()),
        cases=495, pixels=sum(c['rows'] for c in census), cache_bytes=cache_bytes, known_real_null=known_null,
        original_validation_forwards=990, new_validation_forwards=0, reused_validation_guard_count=9,
        cache_arrays_loaded=True, new_model_forwards=0, optimizer_updates=0, old_partial_gradients_reused=False,
        labels_loaded_as_arrays=False, hidden_gt_training_usage='none', test_gt_usage='none')
    return receipt, units, census


def write_references(output, metadata, receipt, units, census):
    output = Path(output); indexed = []
    b.require(receipt['status'] == 'PASS' and receipt['cache_reuse_approved'], 'unapproved reuse')
    for unit in units:
        original, source = unit['original'], unit['source']
        wrapper = derived_unit(original, source, receipt['source_metadata'], metadata)
        validate_derived(wrapper, original, source, receipt['source_metadata'], metadata)
        path = output/'validation_units'/(e.unit_name(original['seed'], original['stage_index'])+'.json')
        digest = b.write_json(path, wrapper)
        indexed.append(dict(seed=original['seed'], stage_index=original['stage_index'], path=str(path), sha256=digest,
            cases=wrapper['cases'], source_validation_unit=wrapper['source_validation_unit']))
    b.write_json(output/'RELIABILITY_CACHE_MANIFEST.json', dict(metadata=metadata, status='PASS', unit_count=9, case_count=495,
        units=indexed, reused_validation_forwards=990, new_validation_forwards=0))
    b.write_json(output/'RELIABILITY_SUPPORT_CENSUS.json', dict(metadata=metadata, status='PASS', cases=census,
        rows=sum(c['rows'] for c in census), active=sum(c['active'] for c in census), null=sum(c['null'] for c in census),
        null_UIDs_preserved=True, GT_received_by_builder=False, recalculated_from_verified_original_arrays=True))
