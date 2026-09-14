"""Actual designated U-Net + projected adapters on generated tensors only."""
import copy,os
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from experiments.lcrseg.five_frameworks_v1.native_parent import build,NativeLRParent,IDENTITY,lr
from experiments.lcrseg.five_frameworks_v1.integration import ExecutionPermit,_PERMIT_SEAL
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.train_stage import StageTrainer
from experiments.lcrseg.five_frameworks_v1.recipes import SyntheticCurrentDomain
from experiments.lcrseg.five_frameworks_v1 import checkpoint,native_state
from experiments.lcrseg.tests.five_frameworks_v1.test_integration import assert_state_equal


@pytest.fixture(autouse=True)
def restore_native_runtime_settings():
    # The historical native factory sets two CPU threads. Do not leak that into
    # the existing one-thread independent-process reproducibility tests.
    threads=torch.get_num_threads();deterministic=torch.are_deterministic_algorithms_enabled()
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    yield
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=flags


def native_fixture(family,cwmi,extra_options=None):
    device=torch.device(os.environ.get('NATIVE_TEST_DEVICE','cpu'))
    ref=Path(os.environ['NATIVE_REFERENCE'])
    m=build(ref,device,161)
    parent=NativeLRParent(m,161,{'synthetic_qualification':True,'seed':161,'stage':0})
    model=Model(parent,family).to(device)
    class Provider(SyntheticCurrentDomain):
        def labeled(self,*a,**kw):
            x,y,ids=super().labeled(*a,**kw);return x.to(device),y.to(device),ids
        def unlabeled(self,*a,**kw):
            x,v,ids=super().unlabeled(*a,**kw);return x.to(device),v.to(device),ids
    provider=Provider(seed=161,stage=2,size=int(os.environ.get('NATIVE_TEST_SIZE',128)),stage_source=parent.stage_source)
    permit=ExecutionPermit({},('B',),{},_PERMIT_SEAL)
    options=dict(total_steps=5,lr=.001,weight_decay=4e-5);options.update(extra_options or {})
    return StageTrainer(model,provider,options,cwmi,execution=permit),ref,permit


@pytest.mark.parametrize('family',['F1','F2','F3','F4','F5'])
def test_native_update_gradient_dense_ema_resume_and_merge(family,cwmi,tmp_path):
    t,reference,permit=native_fixture(family,cwmi)
    t.update();grads=t.update()
    ids={id(p) for p in t.model.u_parameters()}
    assert all(v['U'] is None for k,v in grads.items() if k not in ids)
    assert not t.ema.parent.adapters
    assert all(p.grad is None and not p.requires_grad for p in t.ema.parameters())
    assert [g['lr'] for g in t.optimizer.param_groups]==pytest.approx([g['initial_lr']*(1-2/5)**.9 for g in t.optimizer.param_groups])
    identity=dict(family=family,seed=161,order=1,stage=2)
    path=tmp_path/'state.pt';checkpoint.save(t,path,identity)
    t.update();expected=copy.deepcopy(t.model.state_dict());ema=copy.deepcopy(t.ema.state_dict())
    provider=t.provider;options=t.options;device=next(t.model.parameters()).device
    del t
    with patch.object(lr,'basis',side_effect=AssertionError('resume SVD')),patch.object(StageTrainer,'prepare_stage',side_effect=AssertionError('resume probe')):
        restored=native_state.resume(path,reference,device,provider,options,cwmi,identity,permit)
    restored.update();assert_state_equal(expected,restored.model.state_dict());assert_state_equal(ema,restored.ema.state_dict())
    x,_,_=provider.labeled(4)
    with torch.no_grad():before=restored.model(x)
    del restored.ema
    deploy=restored.model.deploy()
    with torch.no_grad():after=deploy(x)
    torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-5)
    assert not deploy.parent.adapters


def test_native_off_geometry_loss_and_frozen_projection(cwmi):
    t,_,_=native_fixture('F1',cwmi);m=t.model.parent.native
    x,y,_=t.provider.labeled(0)
    with torch.no_grad():
        actual=m(x,stochastic_classifier=False)[0];bridged=t.model(x)
    torch.testing.assert_close(actual,bridged,rtol=0,atol=0)
    from experiments.lcrseg.ssl_anchored_mix_v0_1.core import supervised_parts
    assert t.model.parent.supervised(actual.log_softmax(1),y)==sum(supervised_parts(actual.log_softmax(1),y))
    mapped=t.model.parent.output_to_feature(y,x.shape[-2:],True)
    assert (mapped[:,0]==255).all() and (mapped[:,-1]==255).all()
    a=t.model.parent.adapters[-1];p=a.free_projector
    torch.testing.assert_close(p@p,p,atol=1e-10,rtol=1e-10)
    assert all(not x.requires_grad for x in t.model.parent.parameter_groups()['frozen'])
    t.update()
    t.model.parent.apply_constraints()


@pytest.mark.parametrize('family',['F1','F2','F3','F4','F5'])
def test_native_actual_stage_boundary_and_lr_mapping(family,cwmi):
    t,_,permit=native_fixture(family,cwmi)
    t.update();del t.ema
    deployed=t.model.deploy();prior=deployed.transform.detach().clone()
    dense={n:v.detach().clone() for n,v in deployed.parent.native.state_dict().items()}
    source={'domain':'Drishti_GS','seed':161,'immediate_predecessor':True}
    p=NativeLRParent(deployed.parent.native,161,source)
    for name in lr.LAYERS:torch.testing.assert_close(p.native.get_submodule(name[:-7]).weight,dense[name],rtol=0,atol=0)
    assert all(a.b.count_nonzero()==0 for a in p.adapters)
    model=Model(p,family,previous=prior).to(next(p.parameters()).device)
    x,_,_=t.provider.labeled(0)
    with torch.no_grad():torch.testing.assert_close(model(x),deployed(x),rtol=2e-5,atol=2e-5)
    from experiments.lcrseg.five_frameworks_v1.native_runner import options_for
    study={'parent_configs':[{'id':'P3','lr_multiplier':1.,'lr_B_over_A':4.}],
           'priority_order':[],'baseline_search':{}}
    opts=options_for({'family':'PARENT','candidate_id':'P3','domain':'Drishti_GS','phase':'C'},study,{})
    groups=p.optimizer_groups(opts)
    assert [g['lr'] for g in groups]==[.001,.004]


def test_native_F2_all_fourteen_probe_projectors(cwmi):
    t,_,_=native_fixture('F2',cwmi,{'probe_layers':'all_parent_adapters'})
    assert t.probe['complete'] and t.telemetry['probe_vjps']==24
    assert len(t.model.parent.adapters)==14
    for a in t.model.parent.adapters:
        assert a.b.count_nonzero()==0
        v=getattr(t.model.parent,'V64_'+str(a.index))
        assert float((a.a.double()@v).norm()/a.a.double().norm())<1e-5


def test_native_source_worker_generated_data_only(monkeypatch,tmp_path):
    from experiments.lcrseg.five_frameworks_v1 import native_runner as runner
    device=torch.device(os.environ.get('NATIVE_TEST_DEVICE','cpu'))
    class Source(SyntheticCurrentDomain):
        steps_per_epoch=1
        def labeled(self,*a,**kw):
            x,y,ids=super().labeled(*a,**kw);return x.to(device),y.to(device),ids
    provider=Source(size=32)
    monkeypatch.setattr(runner,'NativeCurrentDomain',lambda *a,**kw:provider)
    monkeypatch.setattr(runner,'evaluate',lambda *a,**kw:({'REFUGE':{'macro_Dice':0.}},{}))
    monkeypatch.setitem(runner.STEPS,'REFUGE',2)
    config={'reference':os.environ['NATIVE_REFERENCE'],'data':'NO_REAL_DATA','execution_commit':'SYNTHETIC_TEST_ONLY'}
    permit=ExecutionPermit({},('B',),{},_PERMIT_SEAL)
    from experiments.lcrseg.five_frameworks_v1.native_operations import NativeOperations
    with NativeOperations(tmp_path/'operations') as op:
        r=runner.source_task(config,{'id':'SOURCE_S161','seed':161},permit,tmp_path,device)
    assert op.counts['optimizer_steps']==op.counts['backward']==2
    assert r['step']==r['physical_optimizer_calls']==2 and provider.u_reads==0
    payload=torch.load(tmp_path/'student.pt',map_location='cpu',weights_only=False)
    assert payload['identity']['domain']=='REFUGE' and payload['transform'] is None


def test_native_cost_scopes_are_separate(cwmi):
    from dataclasses import replace
    t,_,permit=native_fixture('F1',cwmi)
    for scope in ('synthetic','smoke','formal'):
        t.execution=replace(permit,bindings={'execution_scope':scope});t.update()
    assert t.telemetry['synthetic_optimizer_updates']==1
    assert t.telemetry['real_smoke_optimizer_updates']==1
    assert t.telemetry['formal_optimizer_updates']==1
