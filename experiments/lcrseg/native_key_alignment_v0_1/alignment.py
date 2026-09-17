"""Current-batch alignment in actual convolution input-patch coordinates."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1.kernels import sliced_wasserstein_equal
from ..lctx_weight_memory_v0_1.core import LAYERS,LowRankConv

LAYER='decoder.dec1.merge.block.3'
ARMS=('C0','C1','C2','C3')


@contextmanager
def capture_input(layer):
    values=[]
    handle=layer.register_forward_pre_hook(lambda module,args: values.append(args[0]))
    try:yield values
    finally:handle.remove()


def binding(parent):
    layer=parent.native.get_submodule(LAYER)
    index=LAYERS.index(LAYER+'.weight')
    v=getattr(parent,'V64_'+str(index)).detach()
    if (not isinstance(layer,LowRankConv) or layer.arm!='LR_SRC_A' or layer.left is not None
            or tuple(layer.weight.shape)!=(16,16,3,3) or tuple(v.shape)!=(144,8)
            or layer.stride!=(1,1) or layer.padding!=(1,1) or layer.dilation!=(1,1) or layer.groups!=1
            or not torch.equal(v.to(layer.right),layer.right)):
        raise ValueError('native patch/protection binding mismatch')
    return layer,v


def rng(provider,step,namespace):
    return generator(provider.seed,provider.order,provider.stage,step,'NATIVE_KEY_ALIGNMENT_V0_1/'+namespace)


def coordinate_basis(arm,v,provider):
    if arm not in ARMS:raise ValueError('unknown arm')
    if arm in ('C0','C1'):return None
    if arm=='C3':return v.detach()
    # Fixed once at entry; independent of patch sampling and training RNG.
    q,_=torch.linalg.qr(torch.randn(144,8,dtype=torch.float64,
                       generator=rng(provider,0,'random_key_entry')),mode='reduced')
    return q.to(v).detach()


def interior(valid):
    """All nine input positions must be real, valid image coordinates."""
    return F.avg_pool2d(valid[:,None].float(),3,1,1)[:,0].eq(1)


def patches(features,indices):
    """Gather selected centers only, in Conv2d weight.flatten(1) C/ky/kx order."""
    b,c,h,w=features.shape
    if c!=16 or indices.ndim!=1:raise ValueError('expected 16-channel input and flat centers')
    batch=indices//(h*w);y=(indices%(h*w))//w;x=indices%w
    if not ((y>0)&(y<h-1)&(x>0)&(x<w-1)&(batch>=0)&(batch<b)).all():
        raise ValueError('patch center outside valid interior')
    offsets=torch.arange(-1,2,device=features.device)
    yy=y[:,None]+offsets;xx=x[:,None]+offsets;channels=torch.arange(c,device=features.device)
    return features[batch[:,None,None,None],channels[None,:,None,None],
                    yy[:,None,:,None],xx[:,None,None,:]].reshape(len(indices),144)


def alignment_loss(student,teacher,l_labels,u_classes,u_pas,u_geometry,basis,provider,step):
    """Labels already lie on the input grid: native selected convolution is same-grid.

    PAS is applied at U centers, geometry/ignore masks across all patch pixels.
    No U labels or persistent reference features enter this function.
    """
    for x,f in ((l_labels,teacher),(u_classes,student),(u_pas,student),(u_geometry,student)):
        if x.shape!=(f.shape[0],*f.shape[-2:]):raise ValueError('unregistered patch/label geometry')
    if basis is not None and basis.shape!=(144,8):raise ValueError('expected 144x8 key basis')
    teacher=teacher.detach();basis=None if basis is None else basis.detach().to(student)
    d=144 if basis is None else 8
    directions=torch.randn(d,32,generator=rng(provider,step,'directions/D'+str(d))).to(student)
    lv=interior(l_labels!=255);uv=interior(u_geometry.bool())&u_pas.detach().bool()
    terms=[];stats={'dimension':d,'dimension_scale':d/8,'classes':{},'positions':{},'projection_norms':{}}
    class_means={}
    for cls in (1,2):
        li=torch.where((lv&(l_labels==cls)).flatten())[0]
        ui=torch.where((uv&(u_classes.detach()==cls)).flatten())[0]
        n=min(64,len(li),len(ui))
        stats['classes'][str(cls)]={'n_L':len(li),'n_U':len(ui),'n':n,'valid':n>=8,
            'PAS_coverage':float(((u_classes==cls)&u_pas&u_geometry).sum()/
                                 ((u_classes==cls)&u_geometry).sum().clamp_min(1))}
        if n<8:continue
        li=li[torch.randperm(len(li),generator=rng(provider,step,f'pixels/L/{cls}'))[:n].to(li.device)]
        ui=ui[torch.randperm(len(ui),generator=rng(provider,step,f'pixels/U/{cls}'))[:n].to(ui.device)]
        stats['positions'][str(cls)]={'L':li.tolist(),'U':ui.tolist()}
        a=patches(student,ui);b=patches(teacher,li)
        if basis is not None:a=a@basis;b=b@basis
        stats['projection_norms'][str(cls)]={'L':float(b.norm(dim=1).mean()),'U':float(a.detach().norm(dim=1).mean())}
        a=F.normalize(a,dim=1,eps=1e-6);b=F.normalize(b,dim=1,eps=1e-6)
        class_means[cls]=(a.detach().mean(0),b.mean(0))
        terms.append(sliced_wasserstein_equal(a,b,directions))
    loss=torch.stack(terms).mean()*(d/8) if terms else student.sum()*0
    stats.update(valid_classes=len(terms),invalid_step=not bool(terms),raw_scaled_loss=float(loss.detach()),
        raw_SWD=float(loss.detach())/(d/8),class_separation=None)
    if len(class_means)==2:
        stats['class_separation']={side:float((class_means[1][i]-class_means[2][i]).norm()) for i,side in enumerate(('U','L'))}
    return loss,stats
