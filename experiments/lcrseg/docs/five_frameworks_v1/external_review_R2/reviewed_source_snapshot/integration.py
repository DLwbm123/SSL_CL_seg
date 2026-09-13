"""Native bridge and metadata-only orchestration; no registered real runner yet."""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import torch
from torch import nn
from .gate import ReviewRequired,require_approval,code_manifest,digest,CAPS
from .planner import validate_dag


class NativeParentBridge(nn.Module):
    """Delegate native behavior without choosing/replacing a parent algorithm.

    Hooks belong to a source-reviewed adapter implementation, not a JSON import
    path. Metadata alone cannot supply executable hooks or authorize payload IO.
    The generic bridge is tested on toy native callbacks; original KI is unbound.
    """
    synthetic=False
    def __init__(self,native,metadata,hooks):
        super().__init__();self.native=native;self.metadata=dict(metadata);self.hooks=dict(hooks)
        required={'features','readout','readout_kernel','supervised','constraint_loss','apply_constraints',
                  'stage_entry','stage_exit','configure_modes','output_to_feature','feature_to_output','optimizer_groups'}
        if not required<=hooks.keys() or any(not callable(hooks[k]) for k in required):raise ValueError('incomplete native hooks')
        if not metadata.get('source_identity') or metadata.get('delta_order')!='B @ A':raise ValueError('unbound parent identity/order')
        if metadata.get('constraint_kind') not in ('hard_input_projection','soft_native','none_verified'):raise ValueError('unbound native constraint')
        names=dict(native.named_parameters());groups=metadata['parameter_names']
        if set(groups)!={'input_factors','output_factors','other_allowed','frozen'}:raise ValueError('incomplete parameter groups')
        flat=[n for values in groups.values() for n in values]
        if len(flat)!=len(set(flat)) or set(flat)!=set(names):raise ValueError('native parameter groups must partition actual parameters')
        self.d=metadata['feature_width']
        if type(self.d) is not int or self.d<2:raise ValueError('incompatible feature width')

    def parameter_groups(self):
        names=dict(self.native.named_parameters())
        return {k:[names[n] for n in v] for k,v in self.metadata['parameter_names'].items()}

    def semantic_metadata(self):return self.metadata

    def configure_stage_training(self):
        self.train();self.hooks['configure_modes'](self.native,'student')
        groups=self.parameter_groups()
        for key,values in groups.items():
            for p in values:p.requires_grad_(key!='frozen')

    def features(self,x,mode='student',overrides=None):
        return self.hooks['features'](self.native,x,mode,overrides)

    def native_readout(self,h,output_shape,mode='student'):
        return self.hooks['readout'](self.native,h,output_shape,mode)

    def effective_readout_kernel_at_entry(self):
        kernel=self.hooks['readout_kernel'](self.native)
        if kernel.ndim!=4 or kernel.shape[1]!=self.d:raise ValueError('native readout feature mismatch')
        return kernel

    def supervised(self,logp,labels):return self.hooks['supervised'](self.native,logp,labels)
    def constraint_loss(self):return self.hooks['constraint_loss'](self.native)
    def apply_constraints(self):return self.hooks['apply_constraints'](self.native)
    def stage_entry(self):return self.hooks['stage_entry'](self.native)
    def stage_exit(self):return self.hooks['stage_exit'](self.native)
    def optimizer_groups(self,options):return self.hooks['optimizer_groups'](self.native,options)
    def output_to_feature(self,x,shape,categorical=False):return self.hooks['output_to_feature'](self.native,x,shape,categorical)
    def feature_to_output(self,x,shape):return self.hooks['feature_to_output'](self.native,x,shape)


def validate_current_manifest(manifest):
    required={'domain','seed','order','stage','split_id','L','U'}
    if not required<=manifest.keys():raise ValueError('incomplete current-domain manifest')
    for key in ('seed','order','stage'):
        if type(manifest[key]) is not int:raise ValueError('invalid manifest '+key)
    if manifest['stage'] not in (1,2):raise ValueError('target adapter requires stage 1 or 2')
    for kind,fields in [('L',{'image','label','patient_id','domain'}),('U',{'image','geometry','source_id','domain'})]:
        for row in manifest[kind]:
            if set(row)!=fields or row['domain']!=manifest['domain']:raise ValueError('cross-domain or forbidden '+kind+' fields')
    return digest(manifest)


# Filled only by a future reviewed code change after original parent binding.
# A path, command or class from an approval/manifest is NEVER dynamically imported.
REAL_RUNNERS={}
_PERMIT_SEAL=object()


@dataclass(frozen=True)
class ExecutionPermit:
    bindings:dict
    phases:tuple
    budget:dict
    _seal:object

    def validate(self):
        if self._seal is not _PERMIT_SEAL:raise ReviewRequired('invalid execution capability')


class CurrentDomainDataAdapter:
    """Separate native reader capabilities. U has no label-reader dependency."""
    def __init__(self,manifest,permit,labeled_reader,unlabeled_reader):
        permit.validate()
        self.manifest_digest=validate_current_manifest(manifest)
        if self.manifest_digest not in permit.bindings.get('authorized_manifest_digests',[]):raise ReviewRequired('manifest outside reviewed capability')
        self.manifest=manifest;self.read_L=labeled_reader;self.read_U=unlabeled_reader

    def labeled(self,index):
        row=self.manifest['L'][index]
        return self.read_L(row['image'],row['label'],row['patient_id'])

    def unlabeled(self,index):
        row=self.manifest['U'][index]
        return self.read_U(row['image'],row['geometry'],row['source_id'])


def ready_nodes(nodes,completed):
    """Finite DAG readiness from sealed metadata receipts, not score guesses."""
    validate_dag(nodes)
    ids={n['id'] for n in nodes}
    if not set(completed)<=ids:raise ValueError('unknown completed node')
    for key,receipt in completed.items():
        if receipt.get('node_id')!=key or receipt.get('status') not in ('SEALED','RESOLVED'):
            raise ValueError('unsealed/mismatched receipt')
        node=next(n for n in nodes if n['id']==key)
        if not set(node['dependencies'])<=set(completed):raise ValueError('receipt bypassed predecessor')
    return [n for n in nodes if n['id'] not in completed and set(n['dependencies'])<=set(completed)]


def preflight(root,approval_path,phases):
    """Read only small metadata and actual code, before any data/native factory."""
    root=Path(root);review=root/'experiments/lcrseg/docs/five_frameworks_v1/review'
    parent=json.loads((review/'PARENT_BINDING.json').read_text())
    runner=REAL_RUNNERS.get(parent.get('real_parent_identity'))
    if parent.get('status')!='BOUND_VERIFIED' or runner is None:
        raise ReviewRequired('CODE_ONLY / PARENT_BINDING_REQUIRED; native execution adapter absent')
    plan=json.loads((review/'RESOLVED_PROTOCOL.json').read_text())
    deps=json.loads((review/'DEPENDENCY_LOCK.json').read_text())
    budget=json.loads((review/'BUDGET_PLAN.json').read_text())
    validate_dag(plan['nodes'])
    if budget.get('status')!='BOUND_METADATA':raise ReviewRequired('unresolved execution budget')
    actual={'reviewed_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'reviewed_code_tree_sha256':code_manifest(root)['reviewed_code_tree_sha256'],
            'parent_binding_sha256':digest(parent),'expanded_plan_sha256':digest(plan),
            'external_dependency_lock_sha256':digest(deps)}
    requested={k:budget.get('execution_caps',{}).get(k) for k in CAPS}
    receipt=json.loads(Path(approval_path).read_text())
    require_approval(receipt,actual,phases,requested,parent)
    # Future runner validates bound native modules/manifest digests, dependency
    # files and resource envelopes using metadata before it opens payloads.
    manifests=runner.validate_metadata(parent,plan,deps,requested)
    permit=ExecutionPermit({**actual,'authorized_manifest_digests':manifests},tuple(phases),requested,_PERMIT_SEAL)
    return runner,permit,plan


def run_reviewed(root,approval_path,phases):
    runner,permit,plan=preflight(root,approval_path,phases)
    return runner.run_finite(plan,permit)
