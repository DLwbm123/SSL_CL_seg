"""Published contracts, role-isolated inputs and zero-update execution guards."""
import csv
import hashlib
from contextlib import ExitStack
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_gate1.binding import (H, S, ProtocolError, require, sha256, check_hash,
    read_json, write_json, write_text, git, verify_ancestor, safe_asset)
from di_dmpa_gate1.feature_extraction import load_models, state_hash
from di_dmpa_gate1_v2.binding import ModelMutation
from di_dmpa_gate1b_v2.binding import validate_freeze

BASE = '4ea4d7723db9cd29295ab000707c7bbb0044d0dc'
FREEZE = 'cda045db7cf9e2fc01903c51c9aca04126494917'
PREREG = '32d32ab5e491f2e14c3edde6b4f319f978217351'
AUTH = 'd6b651fd366dd304ab4d190f7eb5ce9d3afe23ea'
BRANCH = 'codex/gate1c-v2-identity-history-reliability'
REMOTE = 'https://github.com/DLwbm123/SSL_CL_seg.git'
DOMAINS = ('REFUGE', 'RIM_ONE_r3', 'Drishti_GS')
PREREG_HASHES = {'md': 'dee807650b019c3b97c993ec5de1b71a925e38556b14d8bbcbcb4dd7b04c715a',
                 'json': '8b8dc8c56b60e27e3e1521053cd9307bf65d017ec9343476857b9508721c2f57'}
AUTH_HASHES = {'md': '435c302e73820a3f773e8845e33edb3cb308590833fa72cd30f4329fcb37c38e',
               'json': 'aa9fd6616a80ab4e55ade3eefe10780529ac810e773eaa570a4391052c84a6c3'}
FROZEN_HASHES = {
    'GATE1B_V2_FREEZE.json': '85be4cef01435f3908cd0f6bd9b338a67da95b6034d43f75e17bd35243a05ae3',
    'GATE1B_V2_CLOSURE.json': '5ac875b00857f49f5eecb532643543466f5a5b0fe8d25e64e46620b5661f7989',
    'GATE1B_V2_CLOSURE.md': 'e615dc4f966166fc74952063967e0ebf99b77e5a0830f4b1af70430dfb64a6dc',
    'GATE1B_V2_STATUS.json': 'cba67fbf11cd1a5039be20f44d4449d8689b0c6b45476e16650abe30dbc800c6',
    'GATE1B_V2_FINAL_REPORT.md': 'ad43c0d396642f67464332228cd46eed9befb56a00954673f3b690f47fef99fd',
    'GATE1B_V2_PUBLICATION_RECEIPT.json': '91e496a6931c6ca30ad4dc5b7559b962561a57470675968ffe94788896fe53ac',
    'GATE1A_V2_FREEZE.json': '9208473833f68731c0dd0856696c7bb34047aebd106872977fa1ce9f7598de05'}


class NonfiniteEvidence(RuntimeError):
    status = 'BLOCKED_NONFINITE_EVIDENCE'


class IncompleteEvidence(RuntimeError):
    status = 'BLOCKED_INCOMPLETE_EVIDENCE'


class GradientPartitionError(RuntimeError):
    status = 'BLOCKED_GRADIENT_PARTITION_ERROR'


def complete(condition, message):
    if not condition:
        raise IncompleteEvidence(message)


def finite(*arrays):
    for a in arrays:
        ok = bool(torch.isfinite(a).all()) if isinstance(a, torch.Tensor) else bool(np.isfinite(a).all())
        if not ok:
            raise NonfiniteEvidence('nonfinite feature/logit/probability/weight/gradient/state')


def no_updates():
    stack = ExitStack()
    for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'),
                      (torch.autograd, 'backward'), (torch.optim.Optimizer, 'step')):
        stack.enter_context(patch.object(obj, name, side_effect=ProtocolError(f'{name} forbidden in Gate1C')))
    return stack


def validate_contract(p, old, a, b, af):
    require(p['primary'] == dict(baseline='B0', probability_source='ema_teacher', feature_source='ema_teacher',
        gradient_receiver='student', K=2, historical_transform='identity', R4_available=False,
        primary_candidate='R3', primary_normalization='pixel_normalized', feature_source_selection_performed=False), 'primary path changed')
    require(p['gradient_diagnostic'] == old['gradient_diagnostic'], 'fixed pairs/seeds/gradient contract changed')
    pairs = p['gradient_diagnostic']['batch_pairs']
    require(len(pairs) == 72 and len({q['batch_id'] for q in pairs}) == 72 and H(pairs) == p['fixed_batch_pairs_sha256'], '72 exact pairs required')
    require(p['gate1c']['evaluation'] == old['gate1c']['evaluation'] and
            p['gate1c']['primary_gate_conditions'] == old['gate1c']['primary_gate_conditions'] and
            p['gate1c']['formulas'] == old['gate1c']['formulas'], 'inherited reliability semantics changed')
    require(set(p['gate1c']['candidates']) == {'R0', 'R1', 'R2', 'R3'} and not p['primary']['R4_available'], 'R4 forbidden')
    require(p['benchmark'] == old['benchmark'], 'frozen benchmark changed')
    require(all(v is False for v in p['method_flags'].values()), 'method flag enabled')
    require(a['authorization_scope'] == 'GATE1C_V2_ONLY' and a['preregistration_commit'] == PREREG and
            a['preregistration_remote_verified_commit'] == PREREG and a['freeze_remote_verified_commit'] == FREEZE, 'authorization binding changed')
    require(b['transport_status'] == 'FAIL_TRANSPORT_NOT_SUPPORTED' and b['selected_transport'] == 'T0_identity' and
            b['transport_optimizer_steps'] == 6000 and b['model_optimizer_steps'] == 0 and not b['R4_available'] and
            not b['further_transport_attempts_authorized'] and not b['T1_rescue_allowed'] and
            not b['T2_outputs_allowed_for_gate1c'] and not b['drift_calibrated_claim_allowed'], 'transport failure changed')
    require([b['B1_B7'][f'B{i}']['pass_'] for i in range(1, 8)] == [True, True, False, False, False, True, True], 'B gates changed')
    require(len(b['B1_B7']['B4']['units']) == 12 and len(b['B1_B7']['B5']['units']) == 9, 'incomplete frozen transport failure')
    validate_freeze(af)
    for unit, plan in zip(p['validation']['plans'], old['benchmark']['case_plans']):
        require(unit['role'] == 'val' and unit['domain'] == DOMAINS[unit['stage_index']], 'validation role/domain changed')
        require([x['case_id'] for x in unit['cases']] == plan['roles']['val'], 'validation cases changed')
        for row in unit['cases']:
            require(row['teacher_draw0_seed'] == S(['val-teacher-v1', unit['seed'], unit['stage_index'], row['case_id'], 0]) and
                    row['student_seed'] == S(['val-student-v1', unit['seed'], unit['stage_index'], row['case_id'], 0]), 'validation seed changed')
    require(len(p['validation']['plans']) == 9 and sum(len(u['cases']) for u in p['validation']['plans']) == 495, 'incomplete validation plan')


def verify(root, code_commit, *, remote=True, clean=True):
    root = Path(root); repo = root.parents[1]; docs = root/'docs/di_dmpa_jascl'
    require(git(repo, 'rev-parse', 'HEAD') == code_commit, 'execution HEAD mismatch')
    if clean:
        require(not git(repo, 'status', '--porcelain'), 'dirty exact-code execution checkout')
    for parent in (BASE, FREEZE, PREREG, AUTH):
        verify_ancestor(repo, parent, code_commit)
    require(all(x.startswith('A\t') for x in git(repo, 'diff', '--name-status', BASE, code_commit).splitlines()), 'historical file changed')
    hashes = {**FROZEN_HASHES, **{f'DI_DMPA_GATE1C_V2_PREREGISTRATION.{k}': v for k, v in PREREG_HASHES.items()},
              **{f'GATE1C_V2_EXECUTION_AUTHORIZATION.{k}': v for k, v in AUTH_HASHES.items()}}
    for name, digest in hashes.items():
        check_hash(docs/name, digest)
    p = read_json(docs/'DI_DMPA_GATE1C_V2_PREREGISTRATION.json')
    old_path = repo/p['inherited_contract']['path']; check_hash(old_path, p['inherited_contract']['json_sha256'])
    old = read_json(old_path); af = read_json(docs/'GATE1A_V2_FREEZE.json'); b = read_json(docs/'GATE1B_V2_FREEZE.json')
    a = read_json(docs/'GATE1C_V2_EXECUTION_AUTHORIZATION.json')
    validate_contract(p, old, a, b, af); validate_freeze(af, repo)
    require(b['B1_B7'] == read_json(docs/'GATE1B_V2_STATUS.json')['B1_B7'], 'B raw values changed')
    receipt = read_json(docs/'GATE1B_V2_PUBLICATION_RECEIPT.json')
    require(receipt['report_commit'] == p['gate1b_identity']['report_commit'] and
            receipt['formal_artifact_manifest_sha256'] == p['gate1b_identity']['formal_artifact_manifest_sha256'], 'report receipt mismatch')
    for commit, names in ((PREREG, [f'DI_DMPA_GATE1C_V2_PREREGISTRATION.{x}' for x in ('md', 'json')]),
                          (AUTH, [f'GATE1C_V2_EXECUTION_AUTHORIZATION.{x}' for x in ('md', 'json')])):
        for name in names:
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{(docs/name).relative_to(repo)}'])
            require(hashlib.sha256(blob).hexdigest() == hashes[name], 'publication commit blob mismatch')
    remote_sha = None
    if remote:
        reply = subprocess.check_output(['git', 'ls-remote', REMOTE, 'refs/heads/'+BRANCH], text=True, timeout=60)
        remote_sha = reply.split()[0] if reply.split() else None
        require(remote_sha == code_commit, 'remote exact-code barrier failed')
    metadata = dict(registration_id=p['registration_id'], preregistration_commit=PREREG,
        preregistration_file_sha256=PREREG_HASHES, authorization_commit=AUTH, authorization_file_sha256=AUTH_HASHES,
        gate1b_freeze_commit=FREEZE, gate1b_freeze_sha256=FROZEN_HASHES['GATE1B_V2_FREEZE.json'],
        gate1a_freeze_sha256=FROZEN_HASHES['GATE1A_V2_FREEZE.json'], diagnostic_code_commit=code_commit,
        remote_verified_code_commit=remote_sha, publication_file_sha256=hashes, selected_K=2,
        primary_panel='B0-EMA', historical_transform='identity', R4_available=False, fixed_batch_pairs_sha256=H(p['gradient_diagnostic']['batch_pairs']),
        gate1_overall_status='FAIL_TRANSPORT_NOT_SUPPORTED', model_optimizer_steps=0, transport_optimizer_steps_this_gate=0,
        hidden_gt_training_usage='none', test_gt_usage='none', method_registered=False, di_dmpa_training_launched=False,
        use_transport=False, use_proto_inference=False, use_proto_replay=False, use_multi_proto_loss=False,
        next_action='STOP_FOR_INDEPENDENT_REVIEW')
    return p, af, metadata


def checkpoint(p, seed, stage):
    cp = next(c for c in p['immutable_baseline']['checkpoint_inputs'] if c['checkpoint_id'] == f'B0/seed{seed}/stage{stage}')
    require(cp['baseline'] == 'B0' and cp['domain'] == DOMAINS[stage], 'wrong checkpoint source')
    return cp


def load_b0(root, p, seed, stage, device):
    cp = checkpoint(p, seed, stage)
    models, payload = load_models(root, cp, device=device)
    require(payload['config_hash'] == p['immutable_baseline']['configs']['B0']['resolved_config_sha256'], 'B0 config mismatch')
    finite(*(t for m in models.values() for t in m.state_dict().values()))
    legacy = payload.get('prototypes')
    require(isinstance(legacy, torch.Tensor) and legacy.shape == (3, 16), 'missing frozen legacy PAS prototypes')
    finite(legacy); legacy = legacy.detach().to(device)
    require(not legacy.requires_grad and legacy.grad is None, 'legacy prototypes not detached')
    return models, legacy


def records(data_root, p, seed, stage, role):
    require(role in ('train_labeled', 'train_unlabeled', 'val'), 'test/unknown role forbidden')
    require(stage in range(3) and seed in range(3), 'unknown domain/seed')
    unit = next(u for u in p['benchmark']['case_plans'] if u['seed'] == seed and u['stage_index'] == stage)
    pools = [set(unit['roles'][r]) for r in ('train_labeled', 'train_unlabeled', 'val')]
    require(all(not pools[i] & pools[j] for i in range(3) for j in range(i)), 'role overlap')
    asset = next(a for a in p['benchmark']['manifest_assets'] if a['seed'] == seed)
    manifest = Path(data_root)/f'manifests/training/lcrseg_v1_seed{seed}.csv'; check_hash(manifest, asset['sha256'])
    with manifest.open(newline='') as handle:
        selected = [r for r in csv.DictReader(handle) if r['dataset'] == 'fundus' and
                    r['site_or_vendor'] == DOMAINS[stage] and r['primary_20pct_split'] == role]
    selected.sort(key=lambda r: r['case_id'])
    require([r['case_id'] for r in selected] == unit['roles'][role], 'case/domain/role mismatch')
    for r in selected:
        safe_asset(data_root, r['image_h5_relpath'])
        if role == 'train_unlabeled':
            require(not r['label_h5_relpath'] and not r['label_sha256'], 'hidden GT in unlabeled manifest object')
        else:
            require(r['label_h5_relpath'] and r['label_sha256'], 'required visible label missing')
    return selected


def image_only(row):
    return {k: row[k] for k in ('case_id', 'image_h5_relpath', 'image_sha256')}


def array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def tensor_hash(tensor):
    return state_hash({'value': tensor})


def save_arrays(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size,
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_hash(v)) for k, v in arrays.items()})


def read_arrays(desc):
    check_hash(desc['path'], desc['sha256'])
    with np.load(desc['path'], allow_pickle=False) as handle:
        arrays = {k: handle[k] for k in handle.files}
    require(set(arrays) == set(desc['arrays']), 'array cache schema mismatch')
    for k, a in arrays.items():
        ref = desc['arrays'][k]
        require(list(a.shape) == ref['shape'] and str(a.dtype) == ref['dtype'] and array_hash(a) == ref['sha256'], 'array value/shape/hash mismatch')
    return arrays
