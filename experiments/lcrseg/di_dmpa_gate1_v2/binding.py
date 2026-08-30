"""V2 publication barriers and immutable inherited v1 contracts."""
import hashlib
from pathlib import Path
import subprocess

from di_dmpa_gate1.binding import (H, S, PANELS, ProtocolError, check_hash, git, read_json,
    require, sha256, verify_ancestor, write_json, write_text, audit_inputs)
from di_dmpa_gate1.recovery import PLAN_SHA, ATTEMPT1, reuse_sampling_plan

BASE='606a5c53a37d0e4c9605415e8b38a1f177d1604f'
CLOSURE='b61f6db0ca9e746d005937e7dfc51c45078e1d80'
PREREG='eaae37bbaa7546679d9e6893023afbeeef0ab5c6'
AUTH='e8f558dcc3fb6054a3f757c1295bd07ede2a002b'
BRANCH='codex/gate1a-v2-null-aware-sphere'
REMOTE='https://github.com/DLwbm123/SSL_CL_seg.git'
FILE_HASHES={'md':'9e051c6f270fa673d7f8078eaceb4f0d916d5b929a1c92765ff111f67bcbc2fd',
             'json':'b97847425cf5ef612aa646e98b1a21fde31fae4dccfa45a9e9b2d1481497a50f'}


class NonfiniteFeature(RuntimeError):
    status='BLOCKED_NONFINITE_FEATURE'


class InvalidCenter(RuntimeError):
    status='BLOCKED_NONFINITE_OR_INVALID_CENTER'


class ModelMutation(RuntimeError):
    status='BLOCKED_MODEL_MUTATION'


class IncompletePanel(RuntimeError):
    status='BLOCKED_INCOMPLETE_PANEL'


def verify_history(root, code_commit):
    gitroot=Path(root).parents[1]
    paths=['experiments/lcrseg/di_dmpa_gate1',
        'experiments/lcrseg/docs/di_dmpa_jascl/gate1a_results',
        'experiments/lcrseg/docs/di_dmpa_jascl/gate1a_recovery_results']
    paths += ['experiments/lcrseg/docs/di_dmpa_jascl/'+name for name in (
        'DI_DMPA_GATE1_PREREGISTRATION.md','DI_DMPA_GATE1_PREREGISTRATION.json',
        'GATE1A_STATUS.json','GATE1A_FINAL_REPORT.md','GATE1A_RECOVERY_STATUS.json',
        'GATE1A_RECOVERY_FINAL_REPORT.md','GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.md',
        'GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.json')]
    require(not git(gitroot,'diff','--name-only',BASE,code_commit,'--',*paths),'historical source/artifact bytes changed')


def verify(root, code_commit, *, remote=True):
    root=Path(root); gitroot=root.parents[1]; docs=root/'docs/di_dmpa_jascl'
    require(git(gitroot,'rev-parse','HEAD')==code_commit,'exact execution code mismatch')
    require(not git(gitroot,'status','--porcelain'),'dirty execution checkout')
    for ancestor in (BASE,CLOSURE,PREREG,AUTH):verify_ancestor(gitroot,ancestor,code_commit)
    verify_history(root,code_commit)
    bound={}
    for commit,stem in ((CLOSURE,'GATE1A_V1_CLOSURE'),(PREREG,'DI_DMPA_GATE1A_V2_PREREGISTRATION'),(AUTH,'GATE1A_V2_EXECUTION_AUTHORIZATION')):
        for suffix in ('md','json'):
            path=docs/f'{stem}.{suffix}'
            blob=subprocess.check_output(['git','-C',str(gitroot),'show',f'{commit}:{path.relative_to(gitroot)}'])
            bound[path.name]=check_hash(path,hashlib.sha256(blob).hexdigest())
            if commit==PREREG:require(bound[path.name]==FILE_HASHES[suffix],'wrong v2 registration hash')
    p=read_json(docs/'DI_DMPA_GATE1A_V2_PREREGISTRATION.json')
    oldpath=gitroot/p['inherited_normative_contract']['path']
    check_hash(oldpath,p['inherited_normative_contract']['raw_sha256'])
    check_hash(oldpath.with_suffix('.md'),p['inherited_normative_contract']['md_raw_sha256'])
    old=read_json(oldpath)
    require(p['immutable_baseline']==old['immutable_baseline'],'changed baseline identity')
    require(all(v is False for v in p['method_flags'].values()),'method flag enabled')
    require(p['panels']==old['panels'],'panel contract changed')
    auth=read_json(docs/'GATE1A_V2_EXECUTION_AUTHORIZATION.json')
    require(auth['preregistration_commit']==PREREG and auth['authorization_scope']=='GATE1A_V2_ONLY','wrong v2 authorization')
    remote_sha=None
    if remote:
        response=subprocess.check_output(['git','ls-remote',REMOTE,'refs/heads/'+BRANCH],text=True,timeout=60)
        remote_sha=response.split()[0] if response.split() else None
        require(remote_sha==code_commit,'remote exact-code barrier failed')
    return p,old,dict(registration_id=p['registration_id'],v2_preregistration_git_commit=PREREG,
        v2_preregistration_file_sha256=FILE_HASHES,execution_authorization_git_commit=AUTH,v1_closure_git_commit=CLOSURE,
        diagnostic_code_git_commit=code_commit,remote_verified_code_commit=remote_sha,publication_file_sha256=bound,
        v1_history_preserved=True,sampling_plan_sha256=PLAN_SHA,primary_panel='B0-EMA',primary_feature_source='ema_teacher',
        feature_source_selection_performed=False,method_flags=p['method_flags'],model_optimizer_steps=0,
        transport_optimizer_steps=0,method_registered=False,di_dmpa_training_launched=False,
        test_gt_usage='none',hidden_gt_training_usage='none',Gate1B=False,Gate1C=False,
        input_checkpoint_sha256={c['checkpoint_id']:c['sha256'] for c in old['immutable_baseline']['checkpoint_inputs']})
