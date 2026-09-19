"""Fresh study approval AND user launch, before all production payload IO."""
from dataclasses import dataclass
import os
import subprocess
import sys
from pathlib import Path
from ..five_frameworks_v1.integration import _freeze_metadata
from .protocol import ROOT, DOC, STUDY, ARMS, CAPS, read, digest, canonical_plan, validate_plan, code_manifest, execution_plan
_SEAL = object()


@dataclass(frozen=True)
class Capability:
    bindings: object
    budget: object
    _seal: object
    def __post_init__(self):
        object.__setattr__(self, 'bindings', _freeze_metadata(self.bindings))
        object.__setattr__(self, 'budget', _freeze_metadata(self.budget))
    def validate(self):
        if self._seal is not _SEAL or self.bindings.get('study_id') != STUDY:
            raise PermissionError('new AGMS capability required')


def synthetic_capability(scope='cpu_synthetic', production=None):
    if scope not in ('cpu_synthetic', 'synthetic'):raise PermissionError('unknown generated scope')
    if scope != 'cpu_synthetic':
        if type(production) is not Capability:raise PermissionError('CUDA requires fresh review and launch')
        production.validate()
        if production.bindings['execution_scope'] != 'formal':raise PermissionError('fresh production authority required')
    return Capability({**(dict(production.bindings) if production else {}), 'study_id': STUDY, 'execution_scope': scope}, CAPS, _SEAL)


def authorize(review, launch, actual):
    if (review.get('study_id') != STUDY or review.get('decision') != 'APPROVED_FOR_EXPERIMENTS'
            or review.get('is_template', True) or review.get('reviewer_role') != 'external'
            or not review.get('reviewer') or not review.get('review_evidence')
            or review.get('approved_phases') != ['CUDA', 'SMOKE', 'P1'] or review.get('caps') != CAPS
            or any(review.get(k) != v for k,v in actual.items())):
        raise PermissionError('STOP_AWAITING_EXTERNAL_CODE_REVIEW')
    if (launch.get('study_id') != STUDY or launch.get('user_confirmed') is not True
            or launch.get('review_sha256') != digest(review) or launch.get('prompt') != 'AGMS_OBSERVER_V1_USER_LAUNCH'
            or any(launch.get(k) != v for k,v in actual.items())):
        raise PermissionError('independent current user delegation receipt required')


def preflight(config):
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ:raise PermissionError('optimized interpreter forbidden')
    plan = validate_plan(read(DOC/'PLAN.json'));tree = code_manifest()['code_tree_sha256']
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise PermissionError('dirty checkout')
    if tree != read(DOC/'CODE_MANIFEST.json')['code_tree_sha256']:raise PermissionError('code manifest mismatch')
    actual = dict(reviewed_code_commit=head, code_tree_sha256=tree, plan_sha256=plan['plan_sha256'],
                  prefix_sha256=digest(plan['prefixes']), import_sha256=digest({'A0':plan['imports'],'controls':plan['control_imports']}),
                  environment_sha256=digest(plan['environment']), execution_sha256=digest(execution_plan()))
    for key in ('review','launch_confirmation'):
        path=Path(config[key]).resolve()
        if path==ROOT or ROOT in path.parents:raise PermissionError('authority must be outside reviewed checkout')
    authorize(read(config['review']),read(config['launch_confirmation']),actual)
    if config['execution_commit'] != head or Path(config['code']).resolve() != ROOT:raise PermissionError('runtime checkout mismatch')
    new,old=Path(config['run_root']).resolve(),Path(config['prefix_root']).resolve()
    if new==old or new in old.parents or old in new.parents:raise PermissionError('historical root protected')
    if read(DOC/'EXECUTION_PLAN.json') != execution_plan():raise PermissionError('execution plan mismatch')
    from .tests import validate_cpu
    validate_cpu(plan,tree)
    digests=[digest(dict(domain=n['domain'],seed=163,order=n['order'],stage=2,
                        manifest=plan['manifest_sha256'],split=plan['split_sha256'])) for n in plan['nodes']]
    return plan,Capability({**actual,'study_id':STUDY,'execution_scope':'formal',
                           'authorized_manifest_digests':digests,'nodes':plan['nodes'],'options':plan['options']},CAPS,_SEAL)


def validate_runtime(model, provider, options, arm, permit, node):
    from .model import ObserverModel
    from ..five_frameworks_v1.native_parent import NativeLRParent
    from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
    from ..five_frameworks_v1.native_data import NativeCurrentDomain
    from ..five_frameworks_v1.semantics import resolve_options
    if (type(model) is not ObserverModel or type(model.parent) is not NativeLRParent or model.arm!=arm
            or model.family!='B2_PARENT_PAS_KL' or model.sidecar is not None or arm not in ARMS):
        raise PermissionError('native AGMS model required')
    if type(permit) is not Capability:raise PermissionError('new study capability required')
    permit.validate();scope=permit.bindings['execution_scope']
    if scope in ('cpu_synthetic','synthetic'):
        if not isinstance(provider,SyntheticCurrentDomain):raise PermissionError('generated provider required')
        if scope=='cpu_synthetic' and next(model.parameters()).device.type!='cpu':raise PermissionError('CPU only')
        return
    from .protocol import smoke_nodes
    allowed=permit.bindings['nodes'] if scope=='formal' else smoke_nodes() if scope=='smoke' else []
    if node not in allowed or arm!=node['arm'] or resolve_options(options)!=permit.bindings['options'][node['domain']]:
        raise PermissionError('noncanonical node/options')
    if type(provider) is not NativeCurrentDomain or (provider.seed,provider.order,provider.stage)!=(163,node['order'],2):
        raise PermissionError('wrong real provider')
    if provider.stage_source.get('prefix_binding_sha256')!=node['prefix_binding_sha256']:raise PermissionError('wrong prefix')
    if scope=='smoke' and provider._u is not None:raise PermissionError('smoke U forbidden')


def baseline_environment(actual):
    if actual != canonical_plan()['environment']['fingerprint']:raise RuntimeError('BASELINE_REUSE_BLOCKED: environment mismatch')
    return actual
