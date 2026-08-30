"""Published identities and fail-closed, unchanged historical inputs."""
import math
from pathlib import Path
import subprocess

from di_dmpa_gate1.binding import (H, S, ProtocolError, require, sha256, check_hash,
    read_json, write_json, write_text, git, verify_ancestor, safe_asset)
from di_dmpa_gate1_v2.binding import NonfiniteFeature, ModelMutation

BASE = '9b2ffd04c7a8e9da73f08edb0760be3f269065d8'
FREEZE = '58f19e968700bd7708ec00e44a11759b48ce756f'
PREREG = 'b20f186deff287843f3c9f18bf4ab5633908f441'
AUTH = 'c6f72b86fdfa3683a6e2c7dbf593f73cab74c592'
BRANCH = 'codex/gate1b-v2-null-aware-transport'
REMOTE = 'https://github.com/DLwbm123/SSL_CL_seg.git'
PLAN_SHA = '96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24'
FREEZE_SHA = '9208473833f68731c0dd0856696c7bb34047aebd106872977fa1ce9f7598de05'
PREREG_HASHES = {'md':'4822fc07301b44fd21ae4b6e491771cc7dfa55183930b973e687b6ffa6f0a211',
                 'json':'fbd3d308ff66df0ee5cfd3926b60aeadb138157579329fe1d7388789ee20a3cc'}
AUTH_HASHES = {'md':'4fdf8fda0aeea0c4ead95d47233f127fb1f9b7824843f3af14665e7193b002ce',
               'json':'df74703c2c83813c1ec41507e7e535748f336dbf5d9cd5cd3d77734f879c3465'}
DOMAINS = ('REFUGE','RIM_ONE_r3','Drishti_GS')


class InvalidTransportOutput(RuntimeError):
    status = 'BLOCKED_INVALID_TRANSPORT_OUTPUT'


class IncompleteEvidence(RuntimeError):
    status = 'BLOCKED_INCOMPLETE_EVIDENCE'


class NoDirectionalPairs(RuntimeError):
    status = 'FAIL_DIRECTIONAL_PAIR_SUPPORT_NOT_SUPPORTED'


def validate_freeze(frozen, gitroot=None):
    require(frozen['selected_K']==2 and frozen['passing_K']==[2,3,5], 'selected K changed')
    require(frozen['primary_panel']=='B0-EMA', 'only B0-EMA accepted')
    records=frozen['prototype_records']
    keys=[(r['seed'],r['stage_index'],r['class_id']) for r in records]
    require(len(keys)==27 and set(keys)=={(s,t,c) for s in range(3) for t in range(3) for c in range(3)}, '27 unique prototype inputs required')
    for r in records:
        require(r['panel']=='B0-EMA' and r['feature_source']=='ema_teacher' and r['baseline']=='B0' and r['K']==2, 'invalid prototype source')
        require(r['converged'] is True and r['active_mask']==[True,True], 'K2 inputs must converge with two active centers')
        require(r['training_source']=='train_labeled' and not r['operational_refit_allowed'], 'operational source role/refit changed')
        require(r['sampling_plan_sha256']==PLAN_SHA and r['domain']==DOMAINS[r['stage_index']], 'prototype provenance changed')
        require(len(r['centers'])==2 and all(len(p)==16 for p in r['centers']), 'wrong prototype shape')
        require(all(all(math.isfinite(v) for v in p) and abs(math.sqrt(sum(v*v for v in p))-1)<=1e-12 for p in r['centers']), 'invalid prototype norm')
        if gitroot is not None:
            path=Path(gitroot)/r['source_geometry_unit_path'];check_hash(path,r['source_file_sha256'])
            source=read_json(path)
            require(source['fit']['centers']==r['centers'] and source['fit']['active']==r['active_mask'] and
                    source['fit']['selected_restart']==r['selected_restart'] and source['fit']['converged']==r['converged'], 'prototype differs from original fit')
    require(frozen['model_optimizer_steps']==frozen['transport_optimizer_steps']==0, 'Gate1A optimizer counts changed')
    return records


def verify(root, code_commit, *, remote=True, clean=True):
    root=Path(root);gitroot=root.parents[1];docs=root/'docs/di_dmpa_jascl'
    require(git(gitroot,'rev-parse','HEAD')==code_commit, 'wrong execution HEAD')
    if clean:require(not git(gitroot,'status','--porcelain'), 'dirty execution checkout')
    for parent in (BASE,FREEZE,PREREG,AUTH):verify_ancestor(gitroot,parent,code_commit)
    changes=git(gitroot,'diff','--name-status',BASE,code_commit).splitlines()
    require(all(line.startswith('A\t') for line in changes), 'a historical file was modified')
    hashes={}
    for name,digest in [('GATE1A_V2_FREEZE.json',FREEZE_SHA)]+[(f'DI_DMPA_GATE1B_V2_PREREGISTRATION.{ext}',v) for ext,v in PREREG_HASHES.items()]+[(f'GATE1B_V2_EXECUTION_AUTHORIZATION.{ext}',v) for ext,v in AUTH_HASHES.items()]:
        hashes[name]=check_hash(docs/name,digest)
    p=read_json(docs/'DI_DMPA_GATE1B_V2_PREREGISTRATION.json')
    oldpath=gitroot/p['inherited_contract']['path'];check_hash(oldpath,p['inherited_contract']['json_sha256'])
    old=read_json(oldpath);frozen=read_json(docs/'GATE1A_V2_FREEZE.json');validate_freeze(frozen,gitroot)
    require(p['transport']['split_plans']==old['gate1b']['split_plans'] and p['models']==old['gate1b']['models'], 'inherited transport contract changed')
    require(p['admission']['primary_gate_conditions']==old['gate1b']['primary_gate_conditions'], 'gate thresholds changed')
    require(all(v is False for v in p['method_flags'].values()), 'method switch enabled')
    require(p['primary']==dict(baseline='B0',source='previous-stage ema_teacher',target='current-stage ema_teacher',panel='B0-EMA',K=2,feature_source_selection_performed=False,feature_tap='decoder.dec1 post-ReLU',dimensions=16), 'primary path changed')
    remote_sha=None
    if remote:
        result=subprocess.check_output(['git','ls-remote',REMOTE,'refs/heads/'+BRANCH],text=True,timeout=60)
        remote_sha=result.split()[0] if result.split() else None
        require(remote_sha==code_commit, 'remote exact-code barrier failed')
    metadata=dict(registration_id=p['registration_id'],gate1a_v2_freeze_commit=FREEZE,gate1a_v2_freeze_sha256=FREEZE_SHA,
        preregistration_commit=PREREG,preregistration_file_sha256=PREREG_HASHES,authorization_commit=AUTH,
        diagnostic_code_commit=code_commit,remote_verified_code_commit=remote_sha,publication_file_sha256=hashes,
        selected_K=2,primary_panel='B0-EMA',feature_source='ema_teacher',feature_source_selection_performed=False,
        geometry_sampling_plan_sha256=PLAN_SHA,model_optimizer_steps=0,transport_optimizer_steps_at_start=0,
        expected_transport_optimizer_steps=6000,method_registered=False,di_dmpa_training_launched=False,
        hidden_gt_training_usage='none',test_gt_usage='none',Gate1C=False,next_action='STOP_FOR_INDEPENDENT_REVIEW')
    return p,old,frozen,metadata


def checkpoint(p, seed, stage):
    return next(c for c in p['immutable_baseline']['checkpoint_inputs'] if c['checkpoint_id']==f'B0/seed{seed}/stage{stage}')


def load_b0(root, p, seed, stage, device):
    import torch
    from di_dmpa_gate1.feature_extraction import load_models
    cp=checkpoint(p,seed,stage)
    require(cp['baseline']=='B0' and cp['domain']==DOMAINS[stage], 'non-primary checkpoint')
    models,payload=load_models(root,cp,device=device)
    require(payload['config_hash']==p['immutable_baseline']['configs']['B0']['resolved_config_sha256'], 'B0 config mismatch')
    if not all(torch.isfinite(t).all() for model in models.values() for t in model.state_dict().values()):
        raise NonfiniteFeature('nonfinite frozen model tensor')
    return models
