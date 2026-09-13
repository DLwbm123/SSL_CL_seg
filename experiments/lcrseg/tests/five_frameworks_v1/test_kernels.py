"""Adapted reference algebra tests, re-executed against integrated package."""
import pytest
import torch
from torch.nn import functional as F
from experiments.lcrseg.five_frameworks_v1.kernels import (
    masked_kl, masked_jml1, collect_sources, channel_basis,
    gradient_scale_identity, StageSubspaceAdapter, uncertainty_scales,
    gradient_seed_basis, sliced_wasserstein_equal,
    current_prototypes, teacher_pas,
)

DT = torch.float64

def generator(seed=123):
    return torch.Generator().manual_seed(seed)


def test_kl_gradient_equals_soft_ce():
    z=torch.randn(2,3,5,4,generator=generator(),dtype=DT,requires_grad=True)
    tq=torch.randn(2,3,5,4,generator=generator(9),dtype=DT,requires_grad=True)
    q=tq.softmax(1); m=torch.rand(2,5,4,generator=generator(7))>.3
    a=masked_kl(z,q,m); ga,=torch.autograd.grad(a,z,retain_graph=True)
    ce=(-(q.detach()*z.log_softmax(1)).sum(1)*m).sum()/m.sum()
    gb,=torch.autograd.grad(ce,z)
    assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)
    assert tq.grad is None


@pytest.mark.parametrize('fn', [masked_kl, lambda x,q,m:masked_jml1(x.softmax(1),q,m)])
def test_empty_support_is_connected_zero(fn):
    z=torch.randn(1,3,4,4,dtype=DT,requires_grad=True);q=z.detach().softmax(1)
    loss=fn(z,q,torch.zeros(1,4,4,dtype=torch.bool));loss.backward()
    assert loss.item()==0 and z.grad is not None and z.grad.abs().sum()==0


def test_jml_equal_soft_zero():
    p=torch.rand(2,3,4,5,dtype=DT,requires_grad=True)
    loss=masked_jml1(p,p.detach(),torch.ones(2,4,5,dtype=torch.bool));loss.backward()
    assert loss.item()==0 and p.grad.abs().sum()==0


def test_jml_hard_equals_iou():
    p=torch.tensor([[[[.2,.6]],[[.8,.4]]]],dtype=DT,requires_grad=True)
    q=torch.tensor([[[[0.,1.]],[[1.,0.]]]],dtype=DT)
    v=torch.ones(1,1,2,dtype=torch.bool)
    assert torch.allclose(masked_jml1(p,q,v,(1,)),torch.tensor(1-.8/1.4,dtype=DT))


def test_jml_gradient_check_away_from_l1_kinks():
    p=(.6+.3*torch.rand(1,3,2,3,dtype=DT)).requires_grad_()
    q=.1+.2*torch.rand_like(p);m=torch.ones(1,2,3,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x:masked_jml1(x,q,m), (p,))


def test_jml_masks_both_fields_and_missing_images():
    p=torch.tensor([[[[.2,.9]],[[.8,.1]]],[[[.4,.3]],[[.6,.7]]]],dtype=DT)
    q=torch.tensor([[[[.5,.1]],[[.5,.9]]],[[[.4,.3]],[[.6,.7]]]],dtype=DT)
    m=torch.tensor([[[True,False]],[[False,False]]])
    got=masked_jml1(p,q,m,(1,)); expected=2*.3/(1.3+.3)
    assert abs(got.item()-expected)<1e-12


def test_jml_double_empty_class_zero():
    p=torch.zeros(1,3,2,2,dtype=DT,requires_grad=True)
    v=torch.ones(1,2,2,dtype=torch.bool)
    assert masked_jml1(p,p.detach(),v).item()==0


def test_collect_is_selection_not_average():
    a=torch.full((1,3,2,2),1.);b=torch.full_like(a,7.)
    m=torch.tensor([[[True,False],[False,True]]]); l,u=collect_sources(a,b,m)
    assert torch.equal(l[:,0],torch.where(m,1.,7.))
    assert torch.equal(l+u,torch.full_like(l,8.))


def test_basis_orthogonal_and_scale_invariant():
    k=torch.randn(3,7,3,3,dtype=DT,generator=generator())
    q,s=channel_basis(k,3); r,_=channel_basis(10*k,3)
    assert torch.allclose(q.T@q,torch.eye(3,dtype=DT),atol=1e-12)
    assert torch.allclose(q@q.T,r@r.T,atol=1e-10)
    assert (s[:-1]>=s[1:]).all()


def test_basis_uses_previous_transform_in_correct_order():
    k=torch.randn(3,5,3,3,dtype=DT,generator=generator())
    f=torch.randn(5,5,dtype=DT,generator=generator(23))
    q,_=channel_basis(k,2,f)
    effective=torch.einsum('ocij,cd->odij',k,f)
    r,_=channel_basis(effective,2)
    assert torch.allclose(q@q.T,r@r.T,atol=1e-10)


def test_full_readout_gradient_projection_can_be_identity():
    c=torch.randn(3,6,dtype=DT,generator=generator())
    h=torch.randn(6,dtype=DT,requires_grad=True)
    grad,=torch.autograd.grad(-(c@h).log_softmax(0)[1],h)
    _,_,vh=torch.linalg.svd(c,full_matrices=False);q=vh.T
    assert torch.allclose(q@(q.T@grad),grad,atol=1e-12)


def test_adapter_identity_and_exact_merge():
    q,_=torch.linalg.qr(torch.randn(6,3,dtype=DT,generator=generator()))
    f=torch.randn(6,6,dtype=DT,generator=generator(8))
    a=StageSubspaceAdapter(q,f);x=torch.randn(2,6,4,4,dtype=DT)
    assert torch.allclose(a(x),torch.einsum('de,behw->bdhw',f,x),atol=1e-12)
    with torch.no_grad():a.r.normal_(0,.1)
    assert torch.allclose(a(x),torch.einsum('de,behw->bdhw',a.seal(),x),atol=1e-12)


def test_adapter_stage_transition_constant_memory():
    q,_=torch.linalg.qr(torch.randn(6,3,dtype=DT)); x=torch.randn(1,6,3,3,dtype=DT)
    a=StageSubspaceAdapter(q)
    with torch.no_grad():a.r.normal_(0,.03)
    old=a(x);b=StageSubspaceAdapter(q,a.seal())
    assert torch.allclose(old,b(x),atol=1e-12)
    assert b.previous.numel()==36 and b.previous.requires_grad is False


def test_u_gradient_reaches_r_not_parent():
    q,_=torch.linalg.qr(torch.randn(6,3,dtype=DT)); a=StageSubspaceAdapter(q)
    h=torch.randn(1,6,3,3,dtype=DT,requires_grad=True)
    a(h,detach_parent=True).square().mean().backward()
    assert h.grad is None and a.r.grad is not None and a.r.grad.abs().sum()>0
    assert a.previous.grad is None


def test_l_gradient_reaches_parent_and_r():
    q,_=torch.linalg.qr(torch.randn(6,3,dtype=DT)); a=StageSubspaceAdapter(q)
    h=torch.randn(1,6,3,3,dtype=DT,requires_grad=True)
    a(h).square().mean().backward()
    assert h.grad is not None and a.r.grad is not None


def test_custom_backward_is_stated_rule():
    x=torch.randn(2,3,2,2,dtype=DT,requires_grad=True)
    d=torch.rand_like(x,requires_grad=True); weights=torch.randn_like(x)
    y=gradient_scale_identity(x,d)
    assert torch.equal(x,y)
    (y*weights).sum().backward()
    assert torch.allclose(x.grad,weights*d.detach()) and d.grad is None


def test_uncertainty_spectral_limits():
    u=torch.tensor([[[0.,1.]]],dtype=DT)
    s=torch.tensor([1.,.1],dtype=DT)
    d=uncertainty_scales(u,s,2)
    assert torch.equal(d[...,0],torch.ones_like(d[...,0]))
    assert d[0,0,0,1]>d[0,1,0,1]
    assert torch.equal(uncertainty_scales(u,s,0),torch.ones_like(d))


def test_structure_basis_is_in_allowed_input_space():
    gs=[torch.randn(5,7,dtype=DT,generator=generator(i)) for i in range(3)]
    p=torch.diag(torch.tensor([1,1,1,1,0,0,0],dtype=DT))
    v=gradient_seed_basis(gs,3,p)
    assert torch.allclose(p@v,v,atol=1e-12)
    assert torch.allclose(v.T@v,torch.eye(3,dtype=DT),atol=1e-12)


def test_structure_probe_degeneracy_explicit():
    with pytest.raises(ValueError,match='rank'):
        gradient_seed_basis([torch.zeros(3,4,dtype=DT)],2)


def test_swd_same_permuted_set_zero():
    x=torch.randn(8,4,dtype=DT,requires_grad=True);v=torch.randn(4,12,dtype=DT)
    loss=sliced_wasserstein_equal(x,x.detach().flip(0),v)
    loss.backward();assert loss.item()==0 and x.grad.abs().sum()==0


def test_swd_gradient_to_student_only():
    x=torch.randn(7,3,dtype=DT,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
    sliced_wasserstein_equal(x,y,torch.randn(3,8,dtype=DT)).backward()
    assert x.grad is not None and y.grad is None


def test_current_prototypes_ignore_and_missing():
    h=torch.tensor([[[[1.,9.]],[[0.,9.]]]],dtype=DT)
    y=torch.tensor([[[1,255]]]);p,s=current_prototypes(h,y,3)
    assert s.tolist()==[False,True,False]
    assert torch.equal(p[1],torch.tensor([1.,0.],dtype=DT))


def test_teacher_pas_strict_threshold_and_missing_fallback():
    q=torch.tensor([[[[.1,.1]],[[.8,.2]],[[.1,.7]]]],dtype=DT)
    h=torch.tensor([[[[1.,0.]],[[0.,1.]]]],dtype=DT)
    p=torch.tensor([[0.,0.],[1.,0.],[0.,0.]],dtype=DT);s=torch.tensor([False,True,False])
    keep,_=teacher_pas(q,h,p,s,torch.ones(1,1,2,dtype=torch.bool),confidence=.7)
    assert keep.tolist()==[[[True,False]]]


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_invalid_probabilities_fail(bad):
    p=torch.full((1,3,2,2),1/3,dtype=DT);p[0,1,0,0]=bad
    with pytest.raises(ValueError):masked_jml1(p,p,torch.ones(1,2,2,dtype=torch.bool))


def test_projector_geometry_error_fails():
    with pytest.raises(ValueError):channel_basis(torch.randn(3,7,3,3),3,torch.eye(8))


def test_swd_hand_computed_one_dimensional():
    a=torch.tensor([[0.],[2.]],dtype=DT,requires_grad=True)
    b=torch.tensor([[3.],[1.]],dtype=DT,requires_grad=True)
    directions=torch.tensor([[1.,-2.]],dtype=DT,requires_grad=True)
    value=sliced_wasserstein_equal(a,b,directions)
    assert value.item()==1.
    value.backward();assert b.grad is None and directions.grad is None


def test_repeated_eigenvalues_compare_projectors():
    k=torch.zeros(4,4,1,1,dtype=DT)
    k[:,0,0,0]=torch.tensor([1.,-1.,0.,0.],dtype=DT)
    k[:,1,0,0]=torch.tensor([0.,0.,1.,-1.],dtype=DT)
    q,s=channel_basis(k,2)
    assert torch.allclose(q@q.T,torch.diag(torch.tensor([1.,1.,0.,0.],dtype=DT)))
    assert torch.equal(s,torch.tensor([2.,2.],dtype=DT))
