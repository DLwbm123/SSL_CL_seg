"""Bounded checks for label-generic hierarchy, stopped targets and live alignment."""
import torch
from .hierarchy import hierarchy_terms,contrast,tangent_residual
from ..f5_hierarchy_proto_v21.hierarchy import hierarchy_terms as original_hierarchy_terms


def check(physical):
    torch.manual_seed(171)
    labels=torch.tensor([0,1,2,0,1,2]).view(1,1,6).expand(2,4,6).clone()
    vectors=torch.tensor([[-1.,0.,0.,0.],[.2,.8,.1,0.],[.2,-.8,.1,0.]],dtype=torch.double)
    features=(vectors[labels].permute(0,3,1,2)+torch.randn(2,4,4,6,dtype=torch.double)*.03).requires_grad_(True)
    terms,stats=hierarchy_terms(features,labels,[0,1,1]);assert stats['fine_groups']==1 and stats['aligned_edges']==2
    assert all(torch.isfinite(v) and v>=0 for v in terms.values()) and terms['alignment']>0
    base,_=original_hierarchy_terms(features,labels,[0,1,1])
    for k in ('coarse','alignment'):assert torch.equal(terms[k],base[k])
    # Renaming leaves and supplying the corresponding ancestry cannot change the method.
    permutation=torch.tensor([2,0,1]);renamed=permutation[labels]
    other,_=hierarchy_terms(features,renamed,[1,1,0])
    for k in terms:assert torch.allclose(terms[k],other[k],atol=1e-12,rtol=1e-12)
    for changed,y in [(features*7,labels),(features.unsqueeze(2),labels.unsqueeze(1)),(features.repeat(2,1,1,1),labels.repeat(2,1,1))]:
        other,_=hierarchy_terms(changed,y,[0,1,1])
        for k in terms:assert torch.allclose(terms[k],other[k],atol=1e-12,rtol=1e-12)
    alternate=torch.tensor([0,1,2,3,0,1]).view(1,1,6).expand(2,4,6).clone()
    four,info=hierarchy_terms(features,alternate,[0,0,1,1]);assert info['fine_groups']==2 and info['aligned_edges']==4
    assert torch.isfinite(sum(four.values()))
    flat,info=hierarchy_terms(features,labels,[0,1,2]);assert flat['parts']==flat['alignment']==0
    missing=labels.clone();missing[missing==2]=255
    part,info=hierarchy_terms(features,missing,[0,1,1]);assert part['parts']==part['alignment']==0 and info['fine_groups']==0
    empty,_=hierarchy_terms(features,torch.full_like(labels,255),[0,1,1])
    zero,=torch.autograd.grad(sum(empty.values()),features,retain_graph=True);assert torch.count_nonzero(zero)==0
    ignored=labels.clone();ignored[:,:,:2]=255
    masked,_=hierarchy_terms(features,ignored,[0,1,1]);gradient,=torch.autograd.grad(sum(masked.values()),features)
    assert torch.count_nonzero(gradient[:,:,:,:2])==0 and torch.isfinite(gradient).all()
    degenerate,info=hierarchy_terms(torch.zeros_like(features),labels,[0,1,1]);assert sum(degenerate.values())==0 and info['zero_leaf_prototypes']==3
    # Finite differences hold the detached prototype targets fixed.
    z=torch.tensor([[[.4,.2,-.1],[.2,.7,.1]]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[0,1,1]]);anchors=torch.eye(2,dtype=torch.double,requires_grad=True);support=torch.ones(2,dtype=torch.bool)
    value=contrast(z,y,anchors,support,.1);g,pg=torch.autograd.grad(value,(z,anchors),allow_unused=True);assert pg is None
    eps=1e-6;hi=z.detach().clone();lo=hi.clone();hi[0,0,0]+=eps;lo[0,0,0]-=eps
    numeric=(contrast(hi,y,anchors,support,.1)-contrast(lo,y,anchors,support,.1))/(2*eps)
    assert torch.allclose(g[0,0,0],numeric,atol=1e-8,rtol=1e-6)
    # Rotation does not change dot-product geometry or ancestry.
    rot=torch.tensor([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.,0.]],dtype=torch.double)
    other,_=hierarchy_terms(torch.einsum('ij,bjhw->bihw',rot,features.detach()),labels,[0,1,1])
    for k in terms:assert torch.allclose(terms[k],other[k],atol=1e-12,rtol=1e-12)
    parent=torch.tensor([1.,0.,0.],dtype=torch.double,requires_grad=True)
    inp=torch.tensor([[[.9,.8,.7],[.1,.3,-.2],[.2,-.1,.4]]],dtype=torch.double,requires_grad=True)
    residual,valid=tangent_residual(inp,parent);assert valid.all()
    assert torch.allclose((residual*parent.detach()[None,:,None]).sum(1),torch.zeros_like(valid,dtype=torch.double),atol=1e-12)
    fixed=torch.tensor([[0.,1.,0.],[0.,0.,1.]],dtype=torch.double,requires_grad=True)
    val=contrast(residual,y,fixed,support,.1);tg,ag,fg=torch.autograd.grad(val,(inp,parent,fixed),allow_unused=True);assert ag is None and fg is None
    hi=inp.detach().clone();lo=hi.clone();hi[0,1,0]+=eps;lo[0,1,0]-=eps
    fd=(contrast(tangent_residual(hi,parent)[0],y,fixed,support,.1)-contrast(tangent_residual(lo,parent)[0],y,fixed,support,.1))/(2*eps)
    assert torch.allclose(tg[0,1,0],fd,atol=1e-7,rtol=1e-6)
    collinear=torch.ones(2,4,4,6,dtype=torch.double,requires_grad=True)
    colterms,colstats=hierarchy_terms(collinear,labels,[0,1,1]);assert colterms['parts']==0 and colstats['zero_residual_pixels']>0
    cg,=torch.autograd.grad(colterms['parts'],collinear);assert torch.count_nonzero(cg)==0 and torch.isfinite(cg).all()
    zeroparent=features.detach().clone();zeroparent[:,0]=0
    opposite=torch.tensor([[[[1.,1.]],[[0.,0.]],[[0.,0.]],[[0.,0.]]]],dtype=torch.double).repeat(1,1,1,3)
    # Opposite siblings cancel the parent mean; residual contrast must be omitted.
    opposite[:,:,:,2:4]*=-1
    ol=torch.tensor([[[1,1,2,2,255,255]]]);opposite.requires_grad_(True);ot,os=hierarchy_terms(opposite,ol,[0,1,1]);assert ot['parts']==0 and os['fine_groups']==0
    og,=torch.autograd.grad(ot['parts'],opposite);assert torch.count_nonzero(og)==0 and torch.isfinite(og).all()
    invalid=0
    for x,y,m,t,s in [(features,labels,[],.1,.25),(features,labels,[0,2,2],.1,.25),
                      (features,labels,[0,True,1],.1,.25),(features,labels.float(),[0,1,1],.1,.25),
                      (features,labels+3,[0,1,1],.1,.25),(features[:,:,:,:3],labels,[0,1,1],.1,.25),
                      (features*float('nan'),labels,[0,1,1],.1,.25),(torch.full_like(features,1e308),labels,[0,1,1],.1,.25),
                      (features,labels,[0,1,1],0.,.25),(features,labels,[0,1,1],.1,-1.)]:
        try:hierarchy_terms(x,y,m,t,s)
        except (ValueError,FloatingPointError):invalid+=1
        else:raise AssertionError('invalid hierarchy input accepted')
    assert invalid==10
    fit=torch.nn.Parameter(features.detach().clone());opt=torch.optim.Adam([fit],lr=.1)
    start=float(sum(hierarchy_terms(fit,labels,[0,1,1])[0].values()).detach())
    for i in range(32):
        opt.zero_grad();parts,_=hierarchy_terms(fit,labels,[0,1,1]);loss=sum(parts.values());loss.backward()
        assert torch.isfinite(fit.grad).all();physical.call(i+1);opt.step()
    end=float(sum(hierarchy_terms(fit,labels,[0,1,1])[0].values()).detach());assert end<start*.8
    return dict(status='PASS',cpu_optimizer_updates=32,extra_cpu_gradient_vjps=6,
                finite_difference_error=float(abs(g[0,0,0]-numeric)),tangent_finite_difference_error=float(abs(tg[0,1,0]-fd)),invalid_rejections=invalid,
                fit_initial=start,fit_final=end,checks=['leaf renaming/4-leaf alternate hierarchy/2D-3D/positive feature scaling/whole-batch replication',
                'absent/singleton/ignore/zero-norm decisions and zero ignored gradients','detached contrast anchors with fixed-target finite difference',
                'live current-child alignment and bounded two-case joint hierarchy fit'])
