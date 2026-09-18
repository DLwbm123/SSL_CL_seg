"""Fresh study-specific authorization, checked before production payload IO."""
from dataclasses import dataclass
from types import MappingProxyType
import subprocess
import sys
import os
from .protocol import ROOT,DOC,STUDY,ARMS,read,digest,canonical_plan,validate_plan,code_manifest
from ..five_frameworks_v1.integration import _freeze_metadata

_SEAL=object()
from .protocol import CAPS,DOSES


def execution_plan():
    cells=[dict(arm=a,dose_id=d,lambda_align=w) for a in ARMS for d,w in DOSES.items()]
    cuda=[dict(**c,kind=k,calls=n) for c in cells for k,n in [('warmup',1),('resume',4),('failure',1)]]
    cuda += [dict(arm=a,kind='low_dose_equivalence',calls=6) for a in ARMS]
    cuda += [dict(kind='adam_oracle',case=i,calls=1) for i in range(4)]
    return dict(study_id=STUDY,science_sha256=canonical_plan()['plan_sha256'],caps=CAPS,
        CUDA=dict(cases=cuda,planned_calls=40,cap=40,data='native generated only',shape=[2,3,384,384]),
        smoke=dict(cases=[dict(**c,calls=8,L_only=True,discarded=True) for c in cells],planned_calls=32,cap=32),
        diagnostics={'Drishti_GS':[525,1050,1575,2100],'RIM_ONE_r3':[800,1600,2400,3200]},
        status='STOP_AWAITING_EXTERNAL_CODE_REVIEW',approval_shipped=False,automatic_retry=False)


@dataclass(frozen=True)
class Capability:
    bindings:object
    budget:object
    _seal:object
    def __post_init__(self):
        object.__setattr__(self,'bindings',_freeze_metadata(self.bindings))
        object.__setattr__(self,'budget',_freeze_metadata(self.budget))
    def validate(self):
        if self._seal is not _SEAL or self.bindings.get('study_id')!=STUDY:
            raise PermissionError('new study capability required')


def synthetic_capability(scope='cpu_synthetic',production=None):
    if scope!='cpu_synthetic':
        if not isinstance(production,Capability):raise PermissionError('CUDA requires real fresh authority')
        production.validate()
        if production.bindings.get('execution_scope')!='formal':raise PermissionError('production authority required')
    values=dict(production.bindings) if production else {'study_id':STUDY}
    return Capability({**values,'execution_scope':scope},CAPS,_SEAL)


def authorize(review,launch,actual):
    if (review.get('study_id')!=STUDY or review.get('decision')!='APPROVED_FOR_EXPERIMENTS'
        or review.get('is_template',True) or review.get('reviewer_role')!='external'
        or not review.get('reviewer') or not review.get('review_evidence')
        or review.get('approved_phases')!=['DOSE'] or review.get('caps')!=CAPS
        or any(review.get(k)!=v for k,v in actual.items())):
        raise PermissionError('STOP_AWAITING_EXTERNAL_CODE_REVIEW')
    if (launch.get('study_id')!=STUDY or launch.get('user_confirmed') is not True
        or launch.get('review_sha256')!=digest(review) or any(launch.get(k)!=v for k,v in actual.items())):
        raise PermissionError('fresh independent user launch receipt required')


def preflight(config):
    if sys.flags.optimize or 'PYTHONOPTIMIZE' in os.environ:raise PermissionError('optimized interpreter forbidden')
    plan=validate_plan(read(DOC/'PLAN.json'));tree=code_manifest()['code_tree_sha256']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise PermissionError('dirty checkout')
    if tree!=read(DOC/'CODE_MANIFEST.json')['code_tree_sha256']:raise PermissionError('code tree mismatch')
    actual=dict(reviewed_code_commit=head,code_tree_sha256=tree,plan_sha256=plan['plan_sha256'],
                prefix_sha256=digest(plan['prefixes']),import_sha256=digest(plan['imports']),environment_sha256=digest(plan['environment']),execution_sha256=digest(execution_plan()))
    from pathlib import Path
    for key in ('review','launch_confirmation'):
        path=Path(config[key]).resolve()
        if path==ROOT or ROOT in path.parents:raise PermissionError('authority artifacts must stay outside checkout')
    authorize(read(config['review']),read(config['launch_confirmation']),actual)
    if config['execution_commit']!=head or Path(config['code']).resolve()!=ROOT:raise PermissionError('runtime checkout mismatch')
    root=Path(config['run_root']).resolve();old=Path(config['prefix_root']).resolve()
    if root==old or root in old.parents or old in root.parents:raise PermissionError('historical roots must be isolated')
    if read(DOC/'EXECUTION_PLAN.json')!=execution_plan():raise PermissionError('execution semantics changed')
    cpu=read(DOC/'CPU/TEST_REPORT.json')
    if cpu['status']!='PASS' or cpu['code_tree_sha256']!=tree or cpu['science_sha256']!=plan['plan_sha256'] or cpu['execution_sha256']!=digest(execution_plan()):raise PermissionError('current CPU integration required')
    digests=[digest(dict(domain=n['domain'],seed=n['seed'],order=n['order'],stage=2,
                        manifest=plan['manifest_sha256'],split=plan['split_sha256'])) for n in plan['nodes']]
    return plan,Capability({**actual,'study_id':STUDY,'execution_scope':'formal',
                           'authorized_manifest_digests':digests,'nodes':plan['nodes'],'options':plan['options']},CAPS,_SEAL)


def validate_runtime(model,provider,options,arm,permit,node,weight,meter=True):
    """Object/options check before StageTrainer can request its initial L batch."""
    from ..five_frameworks_v1.model import Model
    from ..five_frameworks_v1.native_parent import NativeLRParent
    from ..five_frameworks_v1.recipes import SyntheticCurrentDomain
    from ..five_frameworks_v1.native_data import NativeCurrentDomain
    from ..five_frameworks_v1.semantics import resolve_options
    from ..native_key_alignment_v0_1.alignment import binding
    if type(model) is not Model or type(model.parent) is not NativeLRParent or model.family!='B2_PARENT_PAS_KL' or model.sidecar is not None:
        raise PermissionError('native B2 model object required')
    if not isinstance(permit,Capability):raise PermissionError('new study capability required')
    permit.validate();scope=permit.bindings['execution_scope']
    if arm not in ARMS:raise PermissionError('only C2/C3')
    if weight not in (.5,2.):
        if not (scope in ('cpu_synthetic','synthetic') and weight in (0.,.05)):raise PermissionError('dose mismatch')
    if scope not in ('cpu_synthetic','synthetic') and not meter:raise PermissionError('production meter required')
    binding(model.parent)
    if scope in ('cpu_synthetic','synthetic'):
        if not isinstance(provider,SyntheticCurrentDomain):raise PermissionError('generated provider required')
        if scope=='cpu_synthetic' and next(model.parameters()).device.type!='cpu':raise PermissionError('CPU only')
        return
    plan=permit.bindings  # recursively frozen, established by canonical preflight
    if node not in plan['nodes'] or arm!=node['arm'] or weight!=node['lambda_align'] or resolve_options(options)!=plan['options'][node['domain']]:
        raise PermissionError('runtime options/node mismatch')
    if type(provider) is not NativeCurrentDomain or (provider.seed,provider.order,provider.stage)!=(node['seed'],node['order'],2):
        raise PermissionError('actual data object mismatch')
    if provider.stage_source.get('prefix_binding_sha256')!=node['prefix_binding_sha256']:
        raise PermissionError('provider prefix mismatch')
    if scope=='smoke' and provider._u is not None:raise PermissionError('smoke U access forbidden')
    if scope not in ('formal','smoke'):raise PermissionError('unknown scope')


def baseline_environment(actual):
    if actual!=canonical_plan()['environment']['fingerprint']:raise RuntimeError('BASELINE_REUSE_BLOCKED: runtime environment')
    return actual
