import ast
import hashlib
import importlib.util
import torch
import pytest
from torch.nn import functional as F
from experiments.lcrseg.five_frameworks_v1.losses import CWMI,MissingBackend,second_order_map,convex_shape,stencil_valid,repair_target,class_swd
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge
from experiments.lcrseg.five_frameworks_v1.recipes import generator


def test_cwmi_author_complex_equation_value_and_gradient(dependency_root):
    path=dependency_root/'CWMI/model/CWMI_loss/CWMI_loss.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='b73837240fd9fb79ab82c8cdc8974ea138219a36605413acdf278078b0d198a0'
    # Test-only extraction of the UNCHANGED author method avoids unrelated
    # image/SSIM imports. Production never patches author or parent globals.
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='complex_mi')
    ns={'torch':torch,'_POS_ALPHA':5e-4};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    torch.manual_seed(7)
    a=torch.randn(1,2,4,12,13,dtype=torch.complex128)
    b=torch.randn_like(a,requires_grad=True)
    original=ns['complex_mi'](None,a,b).real.mean()
    adapted=CWMI.complex_structure(a,b,target_ridge=0.)
    go,=torch.autograd.grad(original,b,retain_graph=True);ga,=torch.autograd.grad(adapted,b)
    assert torch.allclose(original,adapted,atol=1e-9,rtol=1e-9)
    assert torch.allclose(go,ga,atol=1e-9,rtol=1e-8)


def test_cwmi_full_pyramid_and_constant_safety(cwmi):
    torch.manual_seed(3)
    logits=torch.randn(1,3,128,128,requires_grad=True)
    labels=torch.randint(3,(1,128,128))
    value=cwmi(logits.softmax(1),labels);g,=torch.autograd.grad(value,logits)
    assert torch.isfinite(value) and torch.isfinite(g).all() and g.norm()>0
    constant=torch.full((1,3,128,128),1/3,requires_grad=True)
    v=cwmi(constant,torch.zeros_like(labels));v.backward()
    assert torch.isfinite(v) and torch.isfinite(constant.grad).all()
    assert cwmi.last_support==1


def test_cwmi_ignore_crops_and_missing_backend(cwmi,tmp_path):
    p=torch.full((1,3,128,256),1/3,requires_grad=True);y=torch.zeros(1,128,256,dtype=torch.long)
    y[:,:,:128]=255
    value=cwmi(p,y);assert cwmi.last_support==1 and torch.isfinite(value)
    y[:]=255;zero=cwmi(p,y);zero.backward()
    assert zero==0 and cwmi.last_support==0 and p.grad.count_nonzero()==0
    with pytest.raises(MissingBackend):CWMI(tmp_path)
    with pytest.raises(ValueError):cwmi(p[:,:,:64,:64],y[:,:64,:64])


def test_dconv_author_parity(dependency_root):
    path=dependency_root/'D-Convexity/loss.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='e3e6b73836327aafab3edb8db31082250b6b1c19ca3b172c07e327ad71a69af8'
    spec=importlib.util.spec_from_file_location('author_dconv',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=torch.rand(2,1,11,13,dtype=torch.float64,requires_grad=True)
    a=module.C2nd_loss(x);b=second_order_map(x).mean()
    ga,=torch.autograd.grad(a,x,retain_graph=True);gb,=torch.autograd.grad(b,x)
    assert torch.allclose(a,b,atol=1e-14,rtol=1e-14) and torch.allclose(ga,gb,atol=1e-14,rtol=1e-12)


def test_shape_disc_union_not_rim_and_conservative_stencil():
    p=torch.rand(1,3,12,13).softmax(1).requires_grad_();v=torch.ones(1,12,13,dtype=torch.bool);v[:,6,6]=False
    mask=stencil_valid(v)
    assert not mask[:,:2].any() and not mask[:,:,:2].any() and not mask[:,6:9,6:9].any()
    expected=second_order_map(torch.stack((p[:,1]+p[:,2],p[:,2]),1))
    expected=(expected*mask[:,None]).sum()/(2*mask.sum())
    assert torch.equal(convex_shape(p,v),expected)
    empty=convex_shape(p,torch.zeros_like(v));empty.backward();assert p.grad.count_nonzero()==0


@pytest.mark.parametrize('shape,steps',[(0.,3),(.1,0),(.1,3)])
def test_real_repair_d_only_low_resolution(shape,steps):
    torch.manual_seed(8);m=Model(SyntheticParentBridge(),'F4').teacher()
    x=torch.randn(2,3,16,18);geo=torch.ones(2,16,18,dtype=torch.bool)
    q=m(x).softmax(1);target,stats=repair_target(m,x,geo,shape,steps=steps,trust=.05)
    again,other=repair_target(m,x,geo,shape,steps=steps,trust=.05)
    assert torch.equal(target,again) and not target.requires_grad and torch.isfinite(target).all()
    assert torch.allclose(target.sum(1),torch.ones_like(target[:,0]),atol=2e-7)
    assert all(p.grad is None for p in m.parameters())
    if shape==0 or steps==0:assert torch.equal(target,q) and stats['coordinate_vjps']==0
    else:
        assert stats['coordinate_grid']==[64,64] and stats['coordinate_vjps']==3
        assert stats['max_correction']<=stats['radius']+1e-7
        assert stats['readout_only_forwards']==4


def test_class_swd_counts_and_stream_isolation():
    s=torch.randn(1,3,8,8,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
    uc=torch.ones(1,8,8,dtype=torch.long);lc=uc.clone();lc[:,:1,:]=2
    valid=torch.ones_like(uc,dtype=torch.bool)
    augmentation=generator(1,1,1,1,'UL');before=augmentation.get_state().clone()
    value,counts=class_swd(s,t,uc,lc,valid,generator(1,1,1,1,'swd_sampling'))
    value.backward();assert s.grad.norm()>0 and t.grad is None
    assert counts=={1:56,2:0} and torch.equal(before,augmentation.get_state())
    zero,counts=class_swd(s,t,uc,lc,torch.zeros_like(valid),generator(1,1,1,1,'swd_sampling'))
    assert zero==0 and counts=={1:0,2:0}


def test_JML_locked_author_parity(dependency_root):
    from experiments.lcrseg.five_frameworks_v1.kernels import masked_jml1
    path=dependency_root/'JDTLosses/losses/jdt_loss.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='2442237e12f24cb99832bb5c25c6710a37657affaa2fc979b75a641f3596920f'
    spec=importlib.util.spec_from_file_location('author_jdt',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    author=module.JDTLoss(mIoUD=0.,mIoUI=1.,mIoUC=0.,smooth=0.,norm=1)
    p=torch.rand(2,3,5,7,dtype=torch.float64).softmax(1).requires_grad_()
    q=torch.rand_like(p).softmax(1)
    # Select foreground before invoking the author's image/class reduction.
    a=author.forward_loss(p[:,1:].flatten(2),q[:,1:].flatten(2),None,'ALL')
    b=masked_jml1(p,q,torch.ones(2,5,7,dtype=torch.bool))
    ga,=torch.autograd.grad(a,p,retain_graph=True);gb,=torch.autograd.grad(b,p)
    assert torch.allclose(a,b,atol=1e-14) and torch.allclose(ga,gb,atol=1e-14)
