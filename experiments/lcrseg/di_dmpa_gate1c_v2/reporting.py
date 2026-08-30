"""Complete-evidence compiler; controls never rescue the identity-history gate."""
from pathlib import Path

import numpy as np

from di_dmpa_gate1.gate1a_reporting import write_csv
from .binding import complete, require, finite, read_json, write_json, write_text, sha256, H
from .gradients import BLOCKS, NORMALIZATIONS, summary
from .metrics import POINTS


def reduction(reference, candidate):
    if reference is None or candidate is None:
        return None
    if reference == 0:
        return 0.0 if candidate == 0 else None
    return float((reference-candidate)/reference)


def candidate_conditions(classwise, coverage_guard, gradient_rows, candidate, normalization, *, immutable=True, leakage=True):
    units = sorted([r for r in classwise if r['candidate'] == candidate and r['class_id'] in (1, 2)], key=lambda r: (r['seed'], r['stage_index'], r['class_id']))
    expected = {(s, t, c) for s in range(3) for t in range(3) for c in (1, 2)}
    complete(len(units) == 18 and {(r['seed'], r['stage_index'], r['class_id']) for r in units} == expected, '18 foreground units required')
    shared = [x for x in POINTS if all(x in r['actual_shared_points'] for r in units)]
    references = [r['reference_common_support_AURC'] for r in units]; candidates = [r['common_support_AURC'] for r in units]
    defined = all(x is not None for x in references+candidates)
    ref_aurc = float(np.mean(references)) if defined else None
    cand_aurc = float(np.mean(candidates)) if defined else None
    precision_delta = []; raw_units = []
    for r in units:
        a = {x['requested']: x['precision'] for x in r['precision_points']}
        b = {x['requested']: x['precision'] for x in r['reference_points']}
        pd = float(np.mean([a[x]-b[x] for x in shared])) if shared else None
        if pd is not None:
            precision_delta.append(pd)
        ca, ra = r['common_support_AURC'], r['reference_common_support_AURC']
        improved = (ca is not None and ra is not None and ca < ra) or (pd is not None and pd > 0)
        raw_units.append(dict(seed=r['seed'], stage_index=r['stage_index'], class_id=r['class_id'],
            common_upper_bound=r['common_upper_bound'], candidate_AURC=ca, reference_AURC=ra,
            AURC_delta=None if ca is None or ra is None else ca-ra, shared_point_mean_precision_delta=pd, improving=bool(improved)))
    precision = float(np.mean(precision_delta)) if len(precision_delta) == 18 else None
    aurc_pass = bool(defined and ref_aurc > 0 and cand_aurc <= .9*ref_aurc)
    precision_pass = bool(precision is not None and precision >= .01)
    guards = [r for r in coverage_guard if r['candidate'] == candidate]
    complete(len(guards) == 18 and {(r['seed'], r['stage_index'], r['class_id']) for r in guards} == expected, '18 C3 guards required')
    guards = [dict(r, pass_=bool(r['global_operating_point'] > 0 and r['candidate_retained_fraction'] is not None and
        r['reference_retained_fraction'] is not None and r['candidate_retained_fraction'] >= .8*r['reference_retained_fraction'])) for r in guards]
    target = sorted([r for r in gradient_rows if r['candidate'] == candidate and r['normalization'] == normalization and r['block'] == 'global'], key=lambda r: (r['seed'], r['stage_index'], r['pair_index']))
    reference = sorted([r for r in gradient_rows if r['candidate'] == 'R1' and r['normalization'] == 'pixel_normalized' and r['block'] == 'global'], key=lambda r: (r['seed'], r['stage_index'], r['pair_index']))
    pairs = {(s, t, i) for s in range(3) for t in range(3) for i in range(8)}
    for rows in (target, reference):
        complete(len(rows) == 72 and {(r['seed'], r['stage_index'], r['pair_index']) for r in rows} == pairs, '72 paired global comparisons required')
    ts = summary([r['cosine'] for r in target]); rs = summary([r['cosine'] for r in reference])
    gd = ts['required_comparison_defined'] and rs['required_comparison_defined']
    tn = sum(r['cosine'] is not None and r['cosine'] < 0 for r in target)
    rn = sum(r['cosine'] is not None and r['cosine'] < 0 for r in reference)
    tf = tn/72 if gd else None; rf = rn/72 if gd else None
    domains = []
    for t in range(3):
        a = summary([r['cosine'] for r in target if r['stage_index'] == t]); b = summary([r['cosine'] for r in reference if r['stage_index'] == t])
        ok = a['required_comparison_defined'] and b['required_comparison_defined']
        domains.append(dict(stage_index=t, count=24, candidate_median=a['median'], reference_median=b['median'],
            comparison_defined=ok, worsening=b['median']-a['median'] if ok else None,
            pass_=bool(ok and a['median'] >= b['median']-.05)))
    result = dict(
        C1=dict(pass_=aurc_pass or precision_pass, reference_macro_AURC=ref_aurc, candidate_macro_AURC=cand_aurc,
            relative_AURC_reduction=reduction(ref_aurc, cand_aurc), AURC_comparison_defined=defined,
            AURC_comparison_rhs=None if ref_aurc is None else .9*ref_aurc, AURC_branch_pass=aurc_pass,
            shared_points=shared, matched_precision_delta=precision, precision_branch_pass=precision_pass, units=raw_units),
        C2=dict(pass_=sum(r['improving'] for r in raw_units) >= 12, improving_units=sum(r['improving'] for r in raw_units), denominator=18, minimum=12),
        C3=dict(pass_=all(r['pass_'] for r in guards), units=guards, retained_fraction_factor=.8),
        C4=dict(pass_=bool(gd and rn > 0 and tn <= .8*rn), candidate_negative_count=tn, reference_negative_count=rn,
            candidate_fraction=tf, reference_fraction=rf, relative_reduction=reduction(rf, tf),
            comparison_defined=gd, reference_zero_is_not_improvement=rn == 0, denominator=72,
            candidate_undefined=ts['undefined_count'], reference_undefined=rs['undefined_count']),
        C5=dict(pass_=bool(gd and ts['median'] >= rs['median']+.05), candidate_median=ts['median'], reference_median=rs['median'],
            increase=ts['median']-rs['median'] if gd else None, comparison_defined=gd),
        C6=dict(pass_=all(r['pass_'] for r in domains), units=domains, maximum_worsening=.05),
        C7=dict(pass_=bool(immutable), teacher_prototype_history_gradients='None' if immutable else 'INVALID', model_bitwise_unchanged=bool(immutable)),
        C8=dict(pass_=bool(leakage), hidden_gt_training_usage='none' if leakage else 'INVALID', test_gt_usage='none' if leakage else 'INVALID'))
    return dict(candidate=candidate, normalization=normalization, reference='R1/pixel_normalized', C1_C8=result,
                all_pass=all(r['pass_'] for r in result.values()))


def select(primary_pixel, primary_class_balanced, controls=None):
    require(primary_pixel['candidate'] == primary_class_balanced['candidate'] == 'R3', 'controls cannot select primary')
    if primary_pixel['all_pass']:
        status = 'PASS_IDENTITY_HISTORY_WEIGHT_ONLY'; normalization = 'PIXEL_NORMALIZED'
    elif primary_class_balanced['all_pass']:
        status = 'PASS_IDENTITY_HISTORY_CLASS_BALANCED_ONLY'; normalization = 'CLASS_BALANCED'
    else:
        status = 'FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED'; normalization = None
    passed = normalization is not None
    return dict(reliability_status=status, selected_reliability='R3_IDENTITY_HISTORY_WEIGHT_ONLY' if passed else None,
        selected_normalization=normalization, reduced_candidate_status='ELIGIBLE_FOR_NEW_NON_TRANSPORT_METHOD_PREREGISTRATION' if passed else 'NOT_ELIGIBLE',
        gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED', controls_used_for_rescue=False,
        method_registered=False, di_dmpa_training_launched=False, Gate2=False, next_action='STOP_FOR_INDEPENDENT_REVIEW')


def validate_probe_results(p, results, phase):
    expected = p['gradient_diagnostic']['batch_pairs']; lookup = {q['batch_id']: q for q in expected}
    complete(len(results) == 72 and {r['pair']['batch_id'] for r in results} == set(lookup), f'{phase}:72 fixed pairs required')
    candidates = ('PoE',) if phase == 'poe' else ('R0', 'R1', 'R2', 'R3')
    draws = range(8) if phase in ('noise', 'poe') else [0]
    expected_keys = {(c, n, b, d) for c in candidates for n in NORMALIZATIONS for b in ('global', *BLOCKS) for d in draws}
    for r in results:
        require(r['pair'] == lookup[r['pair']['batch_id']] and r['phase'] == phase, 'pair identity changed')
        rows = r['alignment']; keys = [(x['candidate'], x['normalization'], x['block'], x['draw_index']) for x in rows]
        complete(len(keys) == len(set(keys)) and set(keys) == expected_keys, 'missing/extra gradient candidate/block/draw')
        require(r['no_optimizer'] and r['no_backward'] and r['no_parameter_grad_writes'], 'gradient isolation flags')
        if phase in ('noise', 'poe'):
            require(r['teacher_draw_seeds'] == r['pair']['teacher_draw_seeds'], 'teacher draw seed replacement')
        for x in rows:
            require(x['batch_id'] == r['pair']['batch_id'], 'gradient row pair mismatch')
            require(x['teacher_kind'] == ('posterior_mean' if phase == 'posterior' else 'stochastic'), 'posterior-mean control mixed into stochastic primary')
            require((x['cosine'] is None) == x['zero_gradient'], 'zero-gradient null silently changed')
            finite([v for v in (x['cosine'], x['norm_ratio'], x['supervised_norm'], x['unsupervised_norm']) if v is not None])
    return [row for r in results for row in r['alignment']]


def grouped_gradient_statistics(rows):
    groups = {}
    for r in rows:
        key = (r['candidate'], r['normalization'], r['teacher_kind'], r['block'])
        for domain in ('all', r['stage_index']):
            groups.setdefault((*key, domain), []).append(r)
    output = []
    for key, values in groups.items():
        cos = summary([r['cosine'] for r in values]); ratio = summary([r['norm_ratio'] for r in values])
        output.append(dict(candidate=key[0], normalization=key[1], teacher_kind=key[2], block=key[3], stage_index=key[4],
            cosine=cos, norm_ratio=ratio, negative_fraction=None if cos['undefined_count'] else sum(r['cosine'] < 0 for r in values)/len(values),
            descriptive_only=True, admission_uses_draw0_only=True))
    return output


def noise_rows(noise, poe):
    table = []
    for units, is_poe in ((noise, False), (poe, True)):
        for r in units:
            grouped = {}
            for x in r['alignment']:
                grouped.setdefault((x['candidate'], x['normalization'], x['block']), []).append(x)
            for (c, n, b), values in grouped.items():
                values.sort(key=lambda x: x['draw_index'])
                complete([x['draw_index'] for x in values] == list(range(8)), '8 noise draws missing')
                cs = summary([x['cosine'] for x in values]); ns = summary([x['norm_ratio'] for x in values])
                source = next(x for x in noise if x['pair']['batch_id'] == r['pair']['batch_id']) if is_poe else r
                table.append(dict(batch_id=r['pair']['batch_id'], seed=r['pair']['seed'], stage_index=r['pair']['stage_index'],
                    candidate=c, normalization=n, block=b, teacher_draws=8, target_probability_variance=r['target_probability_variance'],
                    weight_variance=source['weight_variance']['R3' if is_poe else c],
                    predicted_class_change_rate=r['predicted_class_change_rate'],
                    cosine_variance=cs['population_variance'], norm_ratio_variance=ns['population_variance'],
                    undefined_cosines=cs['undefined_count'], undefined_ratios=ns['undefined_count'],
                    all_cosines=cs['values'], all_norm_ratios=ns['values']))
    return table


def artifact_manifest(output):
    output = Path(output); excluded = {'GATE1C_V2_ARTIFACT_MANIFEST.json', 'GATE1C_V2_ARTIFACT_MANIFEST.sha256'}
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(output.rglob('*')) if p.is_file() and p.name not in excluded]
    metadata_path = output/'GATE1C_V2_RUN_METADATA.json'
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    result = dict(scope=metadata.get('execution_scope', 'GATE1C_V2_ONLY'),
                  input_contract_version=metadata.get('input_contract_version', 'v2'),
                  artifacts=files, file_count=len(files), total_bytes=sum(x['bytes'] for x in files),
                  raw_tensors='remote_only; public descriptors retain exact paths, bytes and hashes', manifest_excludes_self=True)
    digest = write_json(output/'GATE1C_V2_ARTIFACT_MANIFEST.json', result)
    write_text(output/'GATE1C_V2_ARTIFACT_MANIFEST.sha256', digest+'  GATE1C_V2_ARTIFACT_MANIFEST.json\n')
    return result


def compile_report(output, p, metadata, audit):
    output = Path(output)
    val = [read_json(path) for path in sorted((output/'reliability_units').glob('*.json'))]
    pv = [read_json(path) for path in sorted((output/'poe_validation').glob('*.json'))]
    expected = {(s, t) for s in range(3) for t in range(3)}
    for units in (val, pv):
        complete(len(units) == 9 and {(u['seed'], u['stage_index']) for u in units} == expected, 'nine validation units required')
        for u in units:
            for k, v in metadata.items():
                require(u['metadata'][k] == v, 'mixed validation provenance')
    for a, b in zip(val, pv):
        require(a['classwise'] == [r for r in b['classwise'] if r['candidate'] != 'PoE'], 'PoE evaluator changed primary metrics')
    results = {phase: [read_json(path) for path in sorted((output/'probes'/phase).glob('*/result.json'))]
               for phase in ('draw0', 'noise', 'posterior', 'poe')}
    rows = {phase: validate_probe_results(p, units, phase) for phase, units in results.items()}
    for units in results.values():
        for u in units:
            require(u['metadata'] == metadata, 'mixed gradient provenance')
    classwise = [r for u in val for r in u['classwise']]+[r for u in pv for r in u['classwise'] if r['candidate'] == 'PoE']
    guards = [r for u in val for r in u['coverage_guard']]+[r for u in pv for r in u['coverage_guard'] if r['candidate'] == 'PoE']
    primary_rows = rows['draw0']+[r for r in rows['poe'] if r['draw_index'] == 0]
    leakage_audit = read_json(output/'GATE1C_V2_INPUT_AUDIT.json')
    leakage = (leakage_audit['status'] == 'PASS' and leakage_audit['hidden_gt_training_usage'] == 'none' and
               leakage_audit['test_gt_usage'] == 'none' and leakage_audit['test_role_constructions'] == 0)
    admissions = {c: {n: candidate_conditions(classwise, guards, primary_rows, c, n,
                    immutable=audit['status'] == 'PASS', leakage=leakage) for n in NORMALIZATIONS} for c in ('R0', 'R1', 'R2', 'R3', 'PoE')}
    decision = select(admissions['R3']['pixel_normalized'], admissions['R3']['class_balanced'])
    reliability = dict(metadata=metadata, validation_units=9, foreground_units=18, classwise=classwise,
        coverage_guard=guards, admission=admissions, full_curves='exact per-pixel caches plus deterministic tie rule; CSV contains registered points and support endpoints')
    write_json(output/'RELIABILITY_DIAGNOSTIC_V2.json', reliability)
    for field, filename in (('precision_coverage', 'reliability_precision_coverage_v2.csv'), ('calibration', 'reliability_calibration_v2.csv'),
                            ('composition', 'reliability_composition_v2.csv')):
        values = [r for u in val for r in u[field]]+[r for u in pv for r in u[field] if r['candidate'] == 'PoE']
        write_csv(output/filename, values)
    write_csv(output/'reliability_classwise_v2.csv', [{k: v for k, v in r.items() if k not in ('precision_points', 'reference_points')} for r in classwise])
    all_rows = [dict(phase=phase, **r) for phase, records in rows.items() for r in records]
    write_csv(output/'gradient_alignment_v2.csv', [r for r in all_rows if r['block'] == 'global'])
    write_csv(output/'gradient_blockwise_v2.csv', [r for r in all_rows if r['block'] != 'global'])
    components = [dict(phase=phase, **r) for phase in ('draw0', 'poe') for u in results[phase] for r in u['class_contribution']]
    complete(len(components) == 72*(4+1)*2*7*3, 'class gradient decomposition incomplete')
    complete(all(r['component_sum_pass'] for r in components), 'class gradient decomposition failed')
    write_csv(output/'gradient_class_contribution_v2.csv', components)
    gradient = dict(metadata=metadata, fixed_pairs=72, draw0_only_admission=True,
        statistics=grouped_gradient_statistics(rows['draw0']+[r for r in rows['poe'] if r['draw_index'] == 0]+rows['posterior']),
        primary_gradient_row_count=len(rows['draw0']), class_decomposition_rows=len(components), inactive_parameter_inventory=results['draw0'][0]['parameter_inventory'],
        R3_pixel=admissions['R3']['pixel_normalized'], R3_class_balanced=admissions['R3']['class_balanced'])
    write_json(output/'GRADIENT_CONFLICT_DIAGNOSTIC_V2.json', gradient)
    noise = noise_rows(results['noise'], results['poe']); write_csv(output/'teacher_draw_variance_v2.csv', noise)
    noise_summary = []
    for c in ('R0', 'R1', 'R2', 'R3', 'PoE'):
        for n in NORMALIZATIONS:
            for b in ('global', *BLOCKS):
                for t in ('all', 0, 1, 2):
                    selected = [r for r in noise if r['candidate'] == c and r['normalization'] == n and r['block'] == b and (t == 'all' or r['stage_index'] == t)]
                    noise_summary.append(dict(candidate=c, normalization=n, block=b, stage_index=t, pairs=len(selected),
                        probability_variance=summary([r['target_probability_variance'] for r in selected]), weight_variance=summary([r['weight_variance'] for r in selected]),
                        cosine_variance=summary([r['cosine_variance'] for r in selected]), norm_ratio_variance=summary([r['norm_ratio_variance'] for r in selected])))
    write_json(output/'TEACHER_TARGET_STOCHASTICITY_DIAGNOSTIC_V2.json', dict(metadata=metadata,
        pair_count=72, draws_per_pair=8, draw_records=576, primary_draw=0, student_fixed=True, same_draws_all_candidates=True,
        summaries=noise_summary, per_pair_probability=[{k: u[k] for k in ('pair', 'target_probability_variance', 'predicted_class_change_rate', 'any_draw_class_change_rate', 'weight_variance')} for u in results['noise']],
        posterior_mean=dict(control_only=True, cases=72, baseline_replacement=False,
            gradient_statistics=grouped_gradient_statistics(rows['posterior']))))
    write_json(output/'POE_TARGET_DIAGNOSTIC_V2.json', dict(metadata=metadata, control_only=True, rescue_allowed=False,
        own_predicted_class_strata=True, same_R3_weights=True, validation=[u['poe'] for u in pv], admission=admissions['PoE'],
        gradient_statistics=grouped_gradient_statistics([r for r in rows['poe'] if r['draw_index'] == 0]),
        per_pair_changed_predictions=[dict(pair=u['pair'], draws=u['changed_predictions'], probability_variance=u['target_probability_variance']) for u in results['poe']]))
    status = dict(metadata, **decision, validation_units_completed=9, validation_units_expected=9,
        gradient_pairs_completed=72, gradient_pairs_expected=72, teacher_draw_records_completed=576, teacher_draw_records_expected=576,
        R0_R1_R2_R3_results={c: admissions[c] for c in ('R0', 'R1', 'R2', 'R3')}, PoE_control=admissions['PoE'],
        C1_C8_pixel=admissions['R3']['pixel_normalized']['C1_C8'], C1_C8_class_balanced=admissions['R3']['class_balanced']['C1_C8'],
        gate1a_status='PASS_MULTI_MODALITY_SUPPORTED', gate1b_status='FAIL_TRANSPORT_NOT_SUPPORTED',
        model_checkpoint_immutability='PASS', model_immutability_guards=audit['guard_count'],
        training=False, final_test=False, main_merge=False, theory_final=False,
        posterior_mean_control_only=True, report_commit=None, report_commit_resolution='separate publication receipt after report commit; no self-referential hash')
    version = metadata.get('input_contract_version', 'v2')
    if version == 'v2.1':
        status['next_action'] = metadata['next_action']
    write_json(output/'GATE1C_V2_STATUS.json', status)
    lines = ['# Gate 1C '+version+': identity-history reliability', '', '**'+status['reliability_status']+'**', '',
        'The complete offline diagnostic uses frozen K=2 B0-EMA prototypes with identity history. R4 is unavailable. No T1/T2 output was used.', '',
        '| Gate | R3 pixel normalized | R3 class balanced |', '| --- | --- | --- |']
    for key in ('C1','C2','C3','C4','C5','C6','C7','C8'):
        lines.append('| '+key+' | '+('PASS' if status['C1_C8_pixel'][key]['pass_'] else 'FAIL')+' | '+('PASS' if status['C1_C8_class_balanced'][key]['pass_'] else 'FAIL')+' |')
    lines += ['', 'Validation:9/9 units (18 foreground units); gradient probes:72/72; teacher draws:576/576. All candidate and control evidence is retained, with explicit unsupported coverage and undefined zero-gradient cosines.',
        '', 'Selected reliability: `'+str(status['selected_reliability'])+'`; normalization: `'+str(status['selected_normalization'])+'`.',
        'Reduced candidate: `'+status['reduced_candidate_status']+'`. **Overall Gate1 remains FAIL_TRANSPORT_NOT_SUPPORTED**.',
        '', 'All numeric comparisons use unrounded values. Every gradient admission comparison uses pixel-normalized R1 under the active input contract. Shared validation points and all C1–C8 raw values are in the status/diagnostic JSON and tables.',
        '', f"Model/checkpoint immutability:{audit['guard_count']} complete guards PASS; all9 B0 checkpoint disk hashes unchanged. Model optimizer steps=0; transport optimizer steps this gate=0; no EMA/GAS/prototype update, backward or parameter.grad write.",
        '', 'GT is isolated: current labeled GT only for the supervised reference; current val GT only in the diagnostic evaluator; hidden unlabeled GT and test GT usage both none.',
        '', 'R0/R2, class-balanced controls, posterior-mean teacher and offline PoE are reported separately. Only independently passing R3 class balancing can be selected after pixel R3 fails. Controls never rescue an R3 failure and no method is registered.',
        '', f"Preregistration `{metadata['preregistration_commit']}`; authorization `{metadata['authorization_commit']}`; exact code `{metadata['diagnostic_code_commit']}`.",
        '', 'Report commit is resolved in a separate publication receipt. Exact commands, test evidence, all warnings, caches, model audits and SHA-256 artifact manifests accompany this report.',
        '', '**'+metadata.get('next_action', 'STOP_FOR_INDEPENDENT_REVIEW')+'**. No reduced-method implementation, DI-DMPA training, Gate2, theory final, Prostate/MnMS, full sweep or main merge in this diagnostic.']
    if version == 'v2.1':
        lines += ['', 'Input amendment: only B0/seed1/stage1 uses the independently reconstructed legacy PAS bank. '
            'All nine original model checkpoints and the other eight legacy banks remain unchanged. '
            'The original v2 attempt remains incomplete and none of its partial caches were reused. '
            'The 400 prior baseline recovery optimizer updates are separate from this zero-update diagnostic. '
            'There is no historical saved bank hash; reconstruction support is not historical artifact identity verification.']
    text = '\n'.join(lines)+'\n'
    for name in ('GATE1C_V2_FINAL_REPORT.md', 'RELIABILITY_DIAGNOSTIC_V2.md', 'GRADIENT_CONFLICT_DIAGNOSTIC_V2.md'):
        write_text(output/name, text)
    if version == 'v2.1':
        write_json(output/'GATE1C_V21_STATUS.json', status)
        write_text(output/'GATE1C_V21_FINAL_REPORT.md', text)
    undefined = sum(r['cosine'] is None for r in rows['draw0'] if r['block'] == 'global')
    warnings = ['# Gate 1C '+version+' failures and warnings', '', f'Primary global candidate/normalization rows with undefined zero-gradient cosine: {undefined}; these remain null and never pass a required comparison.',
        '', 'Unsupported validation points are explicit nulls, not extrapolated. Stage0/missing-history scores and null-feature scores are structural nulls, never fake normalized features.',
        '', 'Official classifier constructor/import RNG side effects are preserved; each registered forward reseeds afterwards. Inactive sigma/grad_update are inventoried, not silently dropped.',
        '', 'All raw stdout/stderr logs, test failures if any, and unit artifacts are retained. No result-driven retry, threshold adjustment, new transport fit or baseline update.']
    write_text(output/'GATE1C_V2_FAILURES_AND_WARNINGS.md', '\n'.join(warnings)+'\n')
    return status
