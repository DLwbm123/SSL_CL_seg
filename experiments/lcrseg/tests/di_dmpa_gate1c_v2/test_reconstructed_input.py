"""Synthetic-only v2.1 binding guards; no real checkpoint, bank or GT reads."""
import copy

import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b
from .test_core import ROOT, DOCS, contract


def amendment():
    return b.read_json(DOCS/'DI_DMPA_GATE1C_V21_PREREGISTRATION.json')


def synthetic_input(tmp_path, bank=None):
    spec = copy.deepcopy(amendment()['legacy_prototype_reconstruction'])
    path = tmp_path/'synthetic_bank.pt'
    torch.save(torch.arange(48, dtype=torch.float32).reshape(3, 16) if bank is None else bank, path)
    spec.update(bank_path=str(path), bank_sha256=b.sha256(path))
    p = dict(input_contract_version='v2.1', _legacy_input_contract_verified=True, legacy_prototype_reconstruction=spec)
    cp = dict(checkpoint_id=spec['checkpoint_id'], sha256=spec['checkpoint_sha256'])
    return p, cp


def test_v21_registration_and_recovery_proof_are_frozen():
    for suffix, digest in b.PREREG_V21_HASHES.items():
        b.check_hash(DOCS/f'DI_DMPA_GATE1C_V21_PREREGISTRATION.{suffix}', digest)
    spec = amendment()['legacy_prototype_reconstruction']
    b.check_hash(ROOT.parents[1]/spec['recovery_comparison_path'], spec['recovery_comparison_sha256'])
    proof = b.read_json(ROOT.parents[1]/spec['recovery_comparison_path'])
    b.validate_recovery_proof(spec, proof)
    for change in ('missing_replica', 'different_bank', 'short_trace', 'wrong_helper', 'hidden_gt', 'claimed_historical_hash'):
        bad = copy.deepcopy(proof)
        if change == 'missing_replica': bad['replicas'].pop()
        elif change == 'different_bank': bad['replicas'][1]['candidate_sha256'] = '0'*64
        elif change == 'short_trace': bad['replicas'][0]['original_trace_rows_matched'] = 199
        elif change == 'wrong_helper': bad['replicas'][0]['metadata']['recovery_helper_commit'] = '0'*40
        elif change == 'hidden_gt': bad['replicas'][0]['hidden_gt_training_usage'] = 'used'
        else: bad['historical_bank_hash_verified'] = True
        with pytest.raises(b.ProtocolError): b.validate_recovery_proof(spec, bad)


def test_original_v2_still_rejects_missing_bank():
    with pytest.raises(b.ProtocolError, match='missing frozen legacy'):
        b.legacy_input({'prototypes': None}, {'checkpoint_id': 'B0/seed1/stage1'}, {}, 'cpu')


def test_amendment_preserves_every_other_parent_field(monkeypatch):
    parent = contract()[0]
    spec = amendment()['legacy_prototype_reconstruction']
    native_hash = b.check_hash
    bank_checks = []
    def no_real_bank(path, expected):
        if str(path) == spec['bank_path']:
            bank_checks.append(expected)  # Only synthetic binding; do not open the real artifact.
            return expected
        return native_hash(path, expected)
    monkeypatch.setattr(b, 'check_hash', no_real_bank)
    bound, meta = b.bind_v21(ROOT, b.PREREG_V21, copy.deepcopy(parent),
        {'publication_file_sha256': {}, 'model_optimizer_steps': 0, 'transport_optimizer_steps_this_gate': 0})
    assert bank_checks == [spec['bank_sha256']]
    assert meta['model_optimizer_steps'] == meta['transport_optimizer_steps_this_gate'] == 0
    assert meta['original_gate1c_v2_completed'] is False and meta['prior_baseline_recovery_optimizer_steps'] == 400
    for key in ('input_contract_version', 'legacy_prototype_reconstruction', '_legacy_input_contract_verified'):
        bound.pop(key)
    bound['registration_id'] = parent['registration_id']
    bound['gate1c']['candidates']['R1'] = parent['gate1c']['candidates']['R1']
    assert bound == parent


def test_valid_bound_reconstruction_and_other_banks_unchanged(tmp_path):
    p, cp = synthetic_input(tmp_path)
    legacy = b.legacy_input({'prototypes': None}, cp, p, 'cpu')
    assert torch.equal(legacy, torch.arange(48, dtype=torch.float32).reshape(3, 16))
    assert not legacy.requires_grad and legacy.grad is None
    original = torch.ones(3, 16)
    other = dict(checkpoint_id='B0/seed0/stage0', sha256='synthetic')
    assert torch.equal(b.legacy_input({'prototypes': original}, other, p, 'cpu'), original)
    with pytest.raises(b.ProtocolError, match='missing frozen legacy'):
        b.legacy_input({'prototypes': None}, other, p, 'cpu')
    with pytest.raises(b.ProtocolError, match='original missing payload changed'):
        b.legacy_input({'prototypes': original}, cp, p, 'cpu')


@pytest.mark.parametrize('change', ['unbound', 'wrong_version', 'wrong_checkpoint', 'wrong_checkpoint_hash',
                                  'wrong_bank_hash', 'second_override', 'refit', 'list_of_overrides'])
def test_reconstruction_cannot_be_an_automatic_fallback(tmp_path, change):
    p, cp = synthetic_input(tmp_path)
    spec = p['legacy_prototype_reconstruction']
    if change == 'unbound': p.pop('_legacy_input_contract_verified')
    elif change == 'wrong_version': p['input_contract_version'] = 'v2'
    elif change == 'wrong_checkpoint': spec['checkpoint_id'] = 'C0/seed1/stage1'
    elif change == 'wrong_checkpoint_hash': cp['sha256'] = '0'*64
    elif change == 'wrong_bank_hash': spec['bank_sha256'] = '0'*64
    elif change == 'second_override': spec['other_checkpoint_override_allowed'] = True
    elif change == 'refit': spec['new_refit_allowed'] = True
    else: p['legacy_prototype_reconstruction'] = [spec, copy.deepcopy(spec)]
    with pytest.raises(b.ProtocolError): b.legacy_input({'prototypes': None}, cp, p, 'cpu')


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nan', 'requires_grad'])
def test_invalid_reconstructed_tensor_rejected(tmp_path, kind):
    bank = torch.ones(3, 16)
    if kind == 'shape': bank = bank[:2]
    elif kind == 'dtype': bank = bank.double()
    elif kind == 'nan': bank[1, 2] = float('nan')
    else: bank.requires_grad_(True)
    p, cp = synthetic_input(tmp_path, bank)
    with pytest.raises((b.ProtocolError, b.NonfiniteEvidence)):
        b.legacy_input({'prototypes': None}, cp, p, 'cpu')


def test_all_nine_payloads_checked_without_forward(tmp_path, monkeypatch):
    p, cp = synthetic_input(tmp_path)
    parent = contract()[0]
    p['immutable_baseline'] = copy.deepcopy(parent['immutable_baseline'])
    config_hash = p['immutable_baseline']['configs']['B0']['resolved_config_sha256']
    descriptors = []
    for seed in range(3):
        for stage in range(3):
            identifier = f'B0/seed{seed}/stage{stage}'
            path = tmp_path/f'synthetic_seed{seed}_stage{stage}.pt'
            torch.save(dict(config_hash=config_hash, prototypes=None if identifier == cp['checkpoint_id'] else torch.ones(3, 16)), path)
            descriptors.append(dict(checkpoint_id=identifier, path=str(path), sha256=b.sha256(path)))
    p['immutable_baseline']['checkpoint_inputs'] = descriptors
    p['legacy_prototype_reconstruction']['checkpoint_sha256'] = descriptors[4]['sha256']
    def no_model(*args, **kwargs): raise AssertionError('readiness must not construct a model')
    monkeypatch.setattr(b, 'load_models', no_model)
    audit = b.legacy_input_audit(p)
    assert audit['status'] == 'PASS' and audit['reconstructed_inputs'] == 1 and len(audit['checkpoints']) == 9
    assert audit['model_forwards'] == 0 and audit['labels_read'] is False
    torch.save(dict(config_hash=config_hash, prototypes=None), descriptors[8]['path'])
    descriptors[8]['sha256'] = b.sha256(descriptors[8]['path'])
    with pytest.raises(b.ProtocolError, match='missing frozen legacy'): b.legacy_input_audit(p)
