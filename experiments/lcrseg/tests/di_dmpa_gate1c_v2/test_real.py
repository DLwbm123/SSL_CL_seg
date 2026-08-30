"""Opt-in only AFTER exact code publication; never part of pre-publication tests."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_jascl.modeling import pas_probability_objective
from di_dmpa_gate1c_v2 import binding as b, execution as e, reliability as r
from .test_core import Cached
from .test_pipeline import check_supervised_mean_reference

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not os.environ.get('GATE1C_V2_CODE_COMMIT'), reason='published-code opt-in real integration only')
def test_real_three_stages_null_pas_and_registered_pair(monkeypatch):
    code = os.environ['GATE1C_V2_CODE_COMMIT']; output = Path(os.environ['GATE1C_V2_INTEGRATION_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    version = os.environ.get('GATE1C_INPUT_CONTRACT', 'v2')
    p, freeze, meta = b.verify(ROOT, code, input_contract=version)
    readiness = b.legacy_input_audit(p) if version == 'v2.1' else None
    source = ROOT/'docs/di_dmpa_jascl/gate1a_v2_results/gate1a_v2_8ae5d7532f90aee5d53c0d966706ef64c18a19ac_attempt1/feature_units/B0-EMA_seed2_stage0_val.json'
    b.check_hash(source, '7a36401d2dabc2e58b7f3a5dcee344e1eed1944184bc9b5c75b5d24667d35700')
    known = next(c for c in b.read_json(source)['case_support'] if c['case_id'] == 'REFUGE_train_n0038' and c['null_count'])
    assert known['first_null_coordinates'] == [[185, 180]]
    deterministic_ce = [check_supervised_mean_reference(f'cuda:{gpu}', pattern)
                        for gpu in (0, 1) for pattern in ('mixed', 'single_valid')]
    parity = []; native = e.build
    def checked(sl, sf, tl, tf, legacy, current, history):
        result = native(sl, sf, tl, tf, legacy, current, history)
        _, _, _, original = pas_probability_objective(Cached((sl, sf)), Cached((tl, tf)), None, legacy)
        expected = original.cpu().numpy().reshape(-1)
        assert np.array_equal(result['R1'], expected)
        parity.append(dict(pixels=len(expected), joint_valid=int(expected.sum()), exact_equal=True))
        return result
    monkeypatch.setattr(e, 'build', checked)
    torch.set_num_threads(1); units = []
    with b.no_updates():
        for stage in range(3):
            plan = next(x for x in p['validation']['plans'] if (x['seed'], x['stage_index']) == (0, stage))
            entry = e.validation_unit(ROOT, '/root/LCRSeg', p, freeze, meta, 0, stage, output, 'cuda:0', case_ids=[plan['cases'][0]['case_id']])
            cache = b.read_arrays(entry['cases'][0]['arrays'])
            assert entry['metadata']['bank']['history'] == list(b.DOMAINS[:stage])
            if stage == 0:
                assert np.array_equal(cache['R3'], cache['R2'])
            else:
                assert r.banks(freeze, 0, stage)[1].shape[1] == 2*stage
                assert np.all(cache['R3'] <= cache['R2'])
            units.append(dict(seed=0, stage_index=stage, case_id=entry['cases'][0]['case_id'], passed=True))
        entry = e.validation_unit(ROOT, '/root/LCRSeg', p, freeze, meta, 2, 0, output, 'cuda:0', case_ids=['REFUGE_train_n0038'])
        cache = b.read_arrays(entry['cases'][0]['arrays']); i = 185*384+180
        assert not cache['active_mask'][i] and cache['raw_norms'][i] <= 1e-12
        assert cache['R2'][i] == cache['R3'][i] == 0 and not cache['prototype_valid'][i]
        assert np.isnan(cache['current_scores'][i]).all() and np.isnan(cache['history_gate'][i])
        e.probe_unit(ROOT, '/root/LCRSeg', p, freeze, meta, 0, 0, output, 'cuda:0', 'draw0', pair_indices=[0])
        if version == 'v2.1':
            assert readiness['reconstructed_inputs'] == 1 and len(readiness['checkpoints']) == 9
            entry = e.validation_unit(ROOT, '/root/LCRSeg', p, freeze, meta, 1, 1, output, 'cuda:1',
                                      case_ids=['RIM_ONE_r3_test_G-24-L'])
            units.append(dict(seed=1, stage_index=1, case_id=entry['cases'][0]['case_id'], passed=True))
            e.probe_unit(ROOT, '/root/LCRSeg', p, freeze, meta, 1, 1, output, 'cuda:1', 'draw0', pair_indices=[0])
            affected = b.read_json(output/'probes/draw0/seed1_stage1_pair00/result.json')
            assert affected['pair']['batch_id'] == 'B0/seed1/stage1/RIM_ONE_r3/pair00'
            assert len(affected['alignment']) == 56 and all(x['component_sum_pass'] for x in affected['class_contribution'])
    probe = b.read_json(output/'probes/draw0/seed0_stage0_pair00/result.json')
    assert probe['pair'] == p['gradient_diagnostic']['batch_pairs'][0] and len(probe['alignment']) == 56
    assert probe['R2_R3_exact_equal'] and probe['no_optimizer'] and probe['no_backward']
    assert all(x['component_sum_pass'] for x in probe['class_contribution'])
    audits = [b.read_json(path) for folder in ('validation_models', 'probe_models') for path in (output/folder).rglob('*.json')]
    assert len(audits) == (7 if version == 'v2.1' else 5) and all(x['bitwise_unchanged'] and x['extraction_completed'] for x in audits)
    for cp in p['immutable_baseline']['checkpoint_inputs']:
        b.check_hash(cp['path'], cp['sha256'])
    report = dict(metadata=meta, status='PASS', validation_cases=units, null_ema_case='REFUGE_train_n0038', null_coordinate=[185, 180],
        legacy_payload_readiness=readiness,
        null_available_and_tested=True, source_frozen_feature_unit_sha256=b.sha256(source),
        registered_full_pair=probe['pair'], alignment_rows=56, class_decomposition_rows=len(probe['class_contribution']),
        R1_exact_Gate0_parity=parity, R1_parity_all_exact=all(x['exact_equal'] for x in parity),
        model_immutability_guards=len(audits), all_model_states_bitwise_unchanged=True, all9_B0_checkpoint_hashes_unchanged=True,
        model_optimizer_steps=0, transport_optimizer_steps_this_gate=0, no_optimizer=True, hidden_gt_training_usage='none', test_gt_usage='none',
        deterministic_mean_ce_regression=deterministic_ce)
    if version == 'v2.1':
        report.update(additional_registered_full_pair=affected['pair'], additional_alignment_rows=56,
            additional_class_decomposition_rows=len(affected['class_contribution']),
            reconstructed_bank_sha256=b.check_hash(p['legacy_prototype_reconstruction']['bank_path'],
                                                  p['legacy_prototype_reconstruction']['bank_sha256']))
    b.write_json(output.parent/'GATE1C_V2_REAL_INTEGRATION.json', report)
