"""R07 ownership regressions. Toy callbacks only; no real capabilities issued."""
import copy
from dataclasses import FrozenInstanceError
import pytest
from experiments.lcrseg.five_frameworks_v1 import integration as integ
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge
from experiments.lcrseg.tests.five_frameworks_v1.test_external_review_R2 import current_manifest,permit_for


@pytest.mark.parametrize('branch',['L','U'])
def test_replacing_input_lists_and_public_records_preserves_admitted_read(branch):
    manifest=current_manifest();permit,_,_=permit_for(manifest);calls=[]
    adapter=integ.CurrentDomainDataAdapter(manifest,permit,lambda *v:calls.append(v),lambda *v:calls.append(v))
    original=copy.deepcopy(manifest)
    manifest[branch][:]=current_manifest('future','9')[branch]
    manifest[branch]=[]
    exposed=adapter.manifest
    exposed[branch][0]['image']='unapproved'
    exposed[branch]=[]
    with pytest.raises(AttributeError):adapter.manifest=manifest
    with pytest.raises(AttributeError):adapter.manifest_digest='changed'
    assert adapter.manifest==original
    assert adapter.manifest_digest==integ.validate_current_manifest(adapter.manifest)
    adapter.labeled(0);adapter.unlabeled(0)
    assert calls==[('approved-L-0','approved-GT-0','l0'),('approved-U-0','geometry-0','u0')]


def test_nested_geometry_and_callback_mutation_do_not_escape_snapshot():
    manifest=current_manifest();manifest['U'][0]['geometry']={'shape':[2,3]}
    permit,_,_=permit_for(manifest);calls=[]
    def reader(image,geometry,source):
        calls.append(copy.deepcopy(geometry));geometry['shape'][0]=999
        return geometry
    adapter=integ.CurrentDomainDataAdapter(manifest,permit,None,reader)
    manifest['U'][0]['geometry']['shape'][0]=888
    adapter.manifest['U'][0]['geometry']['shape'][0]=777
    result=adapter.unlabeled(0);result['shape'].append(666)
    adapter.unlabeled(0)
    assert calls==[{'shape':[2,3]},{'shape':[2,3]}]
    assert adapter.manifest_digest==integ.validate_current_manifest(adapter.manifest)


def test_permit_nested_public_state_and_phases_are_immutable_and_seal_is_preserved():
    manifest=current_manifest();sha=integ.validate_current_manifest(manifest)
    bindings={'authorized_manifest_digests':[sha],'nested':{'values':[1]}}
    phases=['B'];budget={'max_formal_optimizer_updates':0,'nested':{'caps':[0]}}
    permit=integ.ExecutionPermit(bindings,phases,budget,integ._PERMIT_SEAL)
    bindings['nested']['values'][0]=9;budget['nested']['caps'][0]=9;phases.append('C')
    assert permit.bindings['nested']['values']==(1,)
    assert permit.budget['nested']['caps']==(0,) and permit.phases==('B',)
    with pytest.raises(TypeError):permit.bindings['authorized_manifest_digests']=('unapproved',)
    with pytest.raises(AttributeError):permit.bindings['authorized_manifest_digests'].append('unapproved')
    with pytest.raises(TypeError):permit.budget['max_formal_optimizer_updates']=1000000
    with pytest.raises(TypeError):permit.budget['nested']['caps'][0]=9
    with pytest.raises(FrozenInstanceError):permit.phases+=('D',)
    assert permit._seal is integ._PERMIT_SEAL
    permit.validate()
    # Frozen metadata can be reused without copying the private seal object.
    clone=integ.ExecutionPermit(permit.bindings,permit.phases,permit.budget,permit._seal)
    clone.validate()
    calls=[]
    integ.CurrentDomainDataAdapter(manifest,clone,lambda *v:calls.append(v),None).labeled(0)
    assert calls==[('approved-L-0','approved-GT-0','l0')]
    with pytest.raises(integ.ReviewRequired):
        integ.ExecutionPermit(permit.bindings,permit.phases,permit.budget,object()).validate()


def test_native_metadata_whitelist_and_semantic_return_are_owned_snapshots():
    native=SyntheticParentBridge();byid={id(p):n for n,p in native.named_parameters()}
    groups={k:[byid[id(p)] for p in v] for k,v in native.parameter_groups().items()}
    metadata={'source_identity':'SYNTHETIC_CALLBACK_FIXTURE_ONLY','delta_order':'B @ A',
              'constraint_kind':'none_verified','parameter_names':groups,'feature_width':4,
              'coordinates':{'order':['input','output']}}
    # No native payload or algorithm is run by these ownership tests.
    hooks={key:lambda *a:None for key in ('features','readout','readout_kernel','supervised',
        'constraint_loss','apply_constraints','stage_entry','stage_exit','configure_modes',
        'output_to_feature','feature_to_output','optimizer_groups')}
    approved=copy.deepcopy(metadata);bridge=integ.NativeParentBridge(native,metadata,hooks)
    groups['input_factors'].clear();metadata['coordinates']['order'].reverse()
    bridge.metadata['parameter_names']['frozen'].clear()
    bridge.semantic_metadata()['coordinates']['order'].clear()
    with pytest.raises(AttributeError):bridge.metadata={}
    bridge.parameter_groups()['frozen'].clear()
    assert bridge.semantic_metadata()==approved
    assert integ.digest(bridge.semantic_metadata())==integ.digest(approved)
    bridge.configure_stage_training()
    assert bridge.parameter_groups()['input_factors'][0] is native.adapters[0].a
    assert native.adapters[0].a.requires_grad and not native.readout.weight.requires_grad
    assert copy.deepcopy(bridge).semantic_metadata()==approved
