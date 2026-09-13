"""Fail-closed approval guard; never evidence that a user actually approved."""
import hashlib
import json
from pathlib import Path


class ReviewRequired(RuntimeError):pass

BINDINGS=('reviewed_code_commit','reviewed_code_tree_sha256','parent_binding_sha256',
          'expanded_plan_sha256','external_dependency_lock_sha256')
CAPS=('max_formal_optimizer_updates','max_source_optimizer_updates','max_real_L_smoke_updates',
      'max_response_or_probe_VJPs','max_readout_only_forwards')


def require_approval(receipt,actual,phases,budget,parent):
    if receipt.get('is_template',True) or receipt.get('decision')!='APPROVED_FOR_EXPERIMENTS':
        raise ReviewRequired('CODE_ONLY: external review required')
    if receipt.get('user_launch_confirmation') is not True or not receipt.get('review_evidence') or not receipt.get('reviewer'):
        raise ReviewRequired('external evidence and explicit user launch confirmation required')
    for key in BINDINGS:
        x=receipt.get(key);length=40 if key=='reviewed_code_commit' else 64
        if not isinstance(x,str) or len(x)!=length or any(c not in '0123456789abcdef' for c in x) or x!=actual.get(key):
            raise ReviewRequired('review binding mismatch: '+key)
    if not phases or not set(phases)<=set('BCD') or not set(phases)<=set(receipt.get('approved_phases',[])):
        raise ReviewRequired('phase outside B/C/D reviewed scope')
    for key in CAPS:
        cap=receipt.get(key);requested=budget.get(key)
        if type(cap) is not int or type(requested) is not int or not 0<=requested<=cap:
            raise ReviewRequired('unresolved/excessive budget: '+key)
    if receipt['max_real_L_smoke_updates']>24:raise ReviewRequired('real smoke hard cap is 24')
    if parent.get('status')!='BOUND_VERIFIED':raise ReviewRequired('PARENT_BINDING_REQUIRED')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_manifest(root):
    root=Path(root)
    scopes=['experiments/lcrseg/five_frameworks_v1','experiments/lcrseg/tests/five_frameworks_v1',
            'experiments/lcrseg/docs/five_frameworks_v1/delivery/configs/study_plan.json',
            'experiments/lcrseg/docs/five_frameworks_v1/review/DEPENDENCY_LOCK.json']
    files=[]
    for scope in scopes:
        p=root/scope
        candidates=[p] if p.is_file() else sorted(p.rglob('*'))
        for path in candidates:
            if path.is_file() and path.suffix in ('.py','.json') and '__pycache__' not in path.parts:
                files.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'reason':'five-framework code-only review implementation and scientific configuration'})
    files.sort(key=lambda v:v['path'])
    return {'files':files,'reviewed_code_tree_sha256':digest(files),'excludes':'approval, mutable reports, historical run artifacts'}
