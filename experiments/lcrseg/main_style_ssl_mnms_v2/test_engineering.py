"""Generated inputs and CPU tensor algebra only; actual optimizer steps forbidden."""
import copy
import os
from pathlib import Path
import pytest
import torch
from .protocol import contract, worker, MEMORY_BATCHES
from .memory import History, seal_site
from .protection import OrthogonalSidecar, elastic_displacement, spectrum


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(163)
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer step forbidden in this engineering campaign')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(torch.optim.SGD,'step',forbidden)
    from ..five_frameworks_v1 import native_parent
    original_precision=native_parent.precision
    def cpu_precision():
        original_precision()
        torch.set_num_threads(1)
    monkeypatch.setattr(native_parent,'precision',cpu_precision)
    assert not torch.cuda.is_initialized()


def summary(domain='Siemens', anchor=None):
    anchor={'W':torch.ones(2,2),'__F__':torch.eye(2)} if anchor is None else anchor
    batches=[]
    for _ in range(MEMORY_BATCHES):
        batches.append([(c,1.+c/10,{n:torch.full_like(v,(c+1)/10) for n,v in anchor.items()},4) for c in range(4)])
    return seal_site(domain,anchor,batches)


def test_fixed_budget_and_no_launch(tmp_path):
    p=contract()
    assert (len(p['nodes']),p['inherited_source_updates'],p['new_formal_optimizer_calls'])==(8,87200,209800)
    assert p['future_memory_L_image_exposures']==512 and p['extra_source_bank_L_image_exposures']==4
    assert p['training_runner_implemented']
    assert not any(p[k] for k in ('execution_authorized','test_accessor','ablation'))
    with pytest.raises(PermissionError):worker(output=tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


@pytest.mark.parametrize('magnitude',[0.,1.,100.,10000.])
def test_rotation_has_no_spectral_wall(magnitude):
    previous=None
    for stage in range(3):
        q=torch.linalg.qr(torch.randn(16,4)).Q.contiguous()
        m=OrthogonalSidecar(q,previous)
        with torch.no_grad():m.r.copy_(magnitude*torch.randn(4,4))
        assert m.valid()
        for matrix in m.matrices():
            s=spectrum(matrix);assert abs(s['min']-1)<2e-5 and abs(s['max']-1)<2e-5
        loss=(m.effective()*torch.randn(16,16)).sum()
        assert torch.isfinite(torch.autograd.grad(loss,m.r)[0]).all()
        previous=m.seal()


def test_rotation_backward_teacher_and_seal():
    m=OrthogonalSidecar(torch.eye(16)[:,:4]);teacher=copy.deepcopy(m).requires_grad_(False)
    x=torch.randn(2,16,8,8,requires_grad=True);weights=torch.randn_like(x)
    (m(x)*weights).sum().backward()
    assert m.r.grad is not None and m.r.grad.norm()>0 and x.grad is not None
    with torch.no_grad():m.r[0,1]=8
    m.update_teacher(teacher);assert teacher.valid() and all(p.grad is None for p in teacher.parameters())
    assert torch.allclose(m(x),torch.einsum('ij,bjhw->bihw',m.seal(),x),atol=2e-6)


def test_degraded_target_and_nonfinite_refused():
    with pytest.raises(ValueError):OrthogonalSidecar(torch.eye(16)[:,:4],torch.eye(16)*.068)
    m=OrthogonalSidecar(torch.eye(16)[:,:4])
    with torch.no_grad():m.r[0,1]=float('nan')
    with pytest.raises(ValueError):m.valid()


def guard(proposal,refs):
    p=torch.nn.Parameter(proposal.clone())
    stats=elastic_displacement([p],{id(p):torch.zeros_like(p)},[(name,[g],w) for name,g,w in refs])
    return p.detach(),stats


def test_conflicting_history_cannot_zero_continuous_learning():
    proposal=torch.tensor([1.,2.,3.],dtype=torch.float64)
    actual,s=guard(proposal,[('Siemens/LV',torch.tensor([1.,0.,0.]),1.),
                             ('Philips/LV',torch.tensor([-1.,0.,0.]),1.)])
    assert actual[1:].equal(proposal[1:])
    assert actual[0]>0 and s['continuous_retention']>=.5-1e-12
    assert s['residual_conflicts']>0  # Elastic protection reports, rather than hides, unresolved conflict.


def test_anisotropic_adam_like_proposal_is_checked_after_scaling():
    raw=torch.tensor([1.,-.1]);ref=torch.ones(2)
    assert raw@ref>0
    actual,s=guard(torch.tensor([-.01,1.]),[('historical',ref,1.)])
    assert s['dots_before'][0]>0 and 0<s['dots_after'][0]<s['dots_before'][0]
    assert actual.norm()>0 and s['spectral_backtracks']==0


def test_many_references_stationarity_and_retention():
    g=torch.Generator().manual_seed(164)
    for count in (1,4,14):
        p=torch.randn(37,generator=g,dtype=torch.float64)
        refs=[(str(i),torch.randn(37,generator=g,dtype=torch.float64),1.) for i in range(count)]
        _,s=guard(p,refs)
        assert s['converged'] and .5-1e-10<=s['continuous_retention']<=1+1e-10
        assert s['solver_sweeps']<=64


def test_zero_reference_does_not_fake_protection():
    p=torch.tensor([1.,2.]);actual,s=guard(p,[('zero',torch.zeros(2),1.)])
    assert actual.equal(p) and s['reference_names']==[]
    _,s=guard(torch.zeros(2),[])
    assert s['zero_displacement'] and s['continuous_retention'] is None


def test_fp32_rounding_is_not_claimed_as_nonzero_learning():
    p=torch.nn.Parameter(torch.ones(1));before={id(p):p.detach().clone()}
    with torch.no_grad():p.copy_(torch.nextafter(p,torch.full_like(p,float('inf'))))
    s=elastic_displacement([p],before,[('rounding',[torch.ones(1)],1.)])
    assert s['continuous_norm']>0 and s['continuous_retention']==.5
    assert s['rounding_error']>=0
    assert s['zero_displacement']==bool(torch.equal(p,before[id(p)]))


def test_invalid_reference_does_not_mutate_proposal():
    p=torch.nn.Parameter(torch.ones(2));old=p.detach().clone()
    with pytest.raises(ValueError):elastic_displacement([p],{id(p):torch.zeros(2)},[('bad',[torch.full((2,),float('nan'))],1.)])
    assert torch.equal(p,old)


def test_sidecar_correction_does_not_damp_unrelated_native_update():
    sidecar=OrthogonalSidecar(torch.eye(16)[:,:4])
    native=torch.nn.Parameter(torch.zeros(2));params=[native,sidecar.r]
    before={id(p):p.detach().clone() for p in params}
    with torch.no_grad():native.copy_(torch.tensor([.25,-.125]));sidecar.r[0,1]=1
    reference=[torch.zeros_like(native),torch.ones_like(sidecar.r)]
    stat=elastic_displacement(params,before,[('history',[*reference],1.)])
    assert torch.equal(native,torch.tensor([.25,-.125]))
    assert sidecar.valid() and stat['spectral_backtracks']==0 and not stat['zero_displacement']


def test_per_domain_class_history_and_immutable_handoff():
    first=summary();second=summary('Philips');h=History([first,second],['Siemens','Philips'])
    frozen=h.state();first['anchor']['W'].zero_()
    values={'W':torch.full((2,2),1.1,requires_grad=True),'__F__':torch.eye(2,requires_grad=True)}
    terms=h.terms(values)
    assert len(terms)==8 and 'Siemens/class3' in terms and 'Philips/class3' in terms
    for value in terms.values():
        grads=torch.autograd.grad(value,list(values.values()),retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
    assert torch.equal(h.state()[0]['anchor']['W'],frozen[0]['anchor']['W'])
    assert all(not v.requires_grad for s in h.state() for v in s['anchor'].values())


def test_signed_function_summary_detects_harmful_direction():
    h=History([summary()],['Siemens'])
    higher={'W':torch.ones(2,2)*1.1,'__F__':torch.eye(2)}
    lower={'W':torch.ones(2,2)*.9,'__F__':torch.eye(2)}
    assert all(h.terms(higher)[n]>h.terms(lower)[n] for n in h.terms(higher))


@pytest.mark.parametrize('corruption',['missing_class','negative_importance','shape','coordinate','lineage'])
def test_corrupt_history_rejected(corruption):
    s=summary();expected=['Siemens']
    if corruption=='missing_class':s['classes'][3]['support']=0
    if corruption=='negative_importance':s['classes'][0]['importance']['W'].fill_(-1)
    if corruption=='shape':s['classes'][0]['jacobian']['W']=torch.ones(3)
    if corruption=='coordinate':s['coordinate']='adapter-A-B'
    if corruption=='lineage':expected=['GE']
    with pytest.raises(ValueError):History([s],expected)


def test_no_extra_batches_to_fill_missing_support():
    anchor={'W':torch.ones(2,2)}
    batches=[[(0,1.,{'W':torch.ones(2,2)},4)]]*32
    with pytest.raises(ValueError,match='missing training-L'):seal_site('Siemens',anchor,batches)
    with pytest.raises(ValueError,match='32'):seal_site('Siemens',anchor,batches+[[()]])


def test_empty_anchor_checkpoint_rejected():
    s=summary();s['anchor']={}
    with pytest.raises(ValueError,match='nonempty'):History([s],['Siemens'])


def test_accumulated_memory_overflow_rejected():
    batches=[[(c,1.,{'W':torch.tensor([1e30])},4) for c in range(4)]]*32
    with pytest.raises(ValueError,match='nonfinite accumulated'):seal_site('Siemens',{'W':torch.ones(1)},batches)


def test_core_checkpoint_restores_next_generated_gradient(tmp_path):
    m=OrthogonalSidecar(torch.eye(16)[:,:4]);h=History([summary()],['Siemens'])
    with torch.no_grad():m.r[0,1]=.25
    path=tmp_path/'core_only.pt';torch.save({'sidecar':m.state_dict(),'memory':h.state(),'rng':torch.get_rng_state()},path)
    payload=torch.load(path,weights_only=False)
    restored=OrthogonalSidecar(torch.eye(16)[:,:4]);restored.load_state_dict(payload['sidecar'])
    newh=History(payload['memory'],['Siemens'])
    x=torch.randn(2,16,8,8);w=torch.randn_like(x)
    a=(m(x)*w).sum();b=(restored(x)*w).sum()
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,m.r)[0],torch.autograd.grad(b,restored.r)[0])
    assert list(h.terms({'W':torch.ones(2,2),'__F__':torch.eye(2)}))==list(newh.terms({'W':torch.ones(2,2),'__F__':torch.eye(2)}))


@pytest.mark.parametrize('seed',[163,164])
def test_real_source_readonly_generated_entry_and_native_graph(seed,tmp_path):
    from ..five_frameworks_v1.native_parent import build
    from ..main_style_ssl_v2.runtime import validate_source
    from .model import Parent, MainModel
    root=Path(os.environ['MNMS_SOURCE_ROOT'])
    node=next(n for n in contract()['nodes'] if n['seed']==seed and n['stage']==0)
    source=torch.load(root/'jobs'/node['id']/'sealed.pt',map_location='cpu',weights_only=False)
    validate_source(source,node,torch.device('cpu'))
    native=build(os.environ['SSLCL_V2_REFERENCE'],torch.device('cpu'),seed,input_channels=1,num_classes=4,domain='Siemens')
    native.load_state_dict(source['native'],strict=True);native.eval().requires_grad_(False)
    model=MainModel(Parent(copy.deepcopy(native),seed,{'domain':'Siemens'},False),previous=source['transform'])
    model.configure_training()
    x=torch.randn(2,1,64,64).clamp(-1,1)
    with torch.no_grad():a=native(x,stochastic_classifier=False)[0];b=model(x)
    assert torch.equal(a,b)
    if seed==163:
        full=torch.randn(1,1,384,384).clamp(-1,1)
        with torch.no_grad():assert torch.equal(native(full,stochastic_classifier=False)[0],model(full))
    stem=model.parent.adapters[0];assert not stem.a.requires_grad and not stem.b.requires_grad
    labels=torch.arange(64)[:,None].expand(64,64)//16;labels=labels[None].repeat(2,1,1)
    loss=torch.nn.functional.cross_entropy(model(x),labels)
    loss.backward()
    assert model.sidecar.r.grad is not None and model.sidecar.r.grad.norm()>0
    assert any(a.b.grad is not None and a.b.grad.norm()>0 for a in model.parent.adapters[1:])
    assert stem.a.grad is None and stem.b.grad is None
    path=tmp_path/'candidate_model.pt';torch.save(model.state_dict(),path)
    restored=MainModel(Parent(copy.deepcopy(native),seed,{'domain':'Siemens'},False),previous=source['transform'])
    restored.load_state_dict(torch.load(path,weights_only=False),strict=True)
    with torch.no_grad():assert torch.equal(model(x),restored(x))
