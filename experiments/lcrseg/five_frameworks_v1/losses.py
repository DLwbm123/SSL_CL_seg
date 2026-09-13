"""Verified structural equations, external complex pyramid, and current-batch SWD."""
import hashlib
import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
from .kernels import masked_kl, sliced_wasserstein_equal


class MissingBackend(RuntimeError):
    pass


PYRAMID_SHA='e5b19299dc1e7ceb6c57c12ddef00e52d370a03d2dcb026b5565474699a61f1d'


class CWMI:
    """Structural-only conditional complex covariance entropy.

    Author N=2/K=4 complex steerable pyramid; stabilized real representation
    of complex covariance. Target-side ridge is an explicit V1 adaptation.
    The author CE mixing parameter is deliberately absent.
    """
    def __init__(self, root, ridge=5e-4):
        path=Path(root)/'model/CWMI_loss/ComplexSteerablePyramid.py'
        if not path.is_file(): raise MissingBackend('locked CWMI source is required')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=PYRAMID_SHA:
            raise MissingBackend('CWMI source fingerprint mismatch')
        spec=importlib.util.spec_from_file_location('sslcl5_external_pyramid',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class DeviceLocalPyramid(module.ComplexSteerablePyramid):
            def get_mask(self,image_size):
                # Author factories omit device=. Scope ALL grid/mask allocations
                # to the feature device; do not move input tensors through CPU.
                with torch.device(self.device):
                    return super().get_mask(image_size)
        self._pyramid_type=DeviceLocalPyramid
        self.pyramids={}
        self.pyramid=self.pyramid_for(torch.device('cpu'))
        self.ridge=ridge
        self.last_support=0

    def pyramid_for(self,device):
        device=torch.device(device)
        if device.type not in ('cpu','cuda'):raise MissingBackend('unsupported CWMI device '+str(device))
        key=str(device)
        if key not in self.pyramids:
            self.pyramids[key]=self._pyramid_type(complex=True,N=2,K=4,device=device)
        return self.pyramids[key]

    def semantic_identity(self):
        return {'backend':'CWMI_device_local_v2','author_pyramid_sha256':PYRAMID_SHA,
                'N':2,'K':4,'complex':True,'prediction_ridge':5e-4,'target_ridge':self.ridge,
                'crop':'fixed_128_all_valid_v1','covariance_dtype':'float64'}

    @staticmethod
    def complex_structure(target,prediction,target_ridge=5e-4):
        def embedding(x):
            x=x.flatten(-2)
            v=torch.cat((x.real,x.imag),-2)
            j=torch.cat((-x.imag,x.real),-2)
            z=torch.cat((v,j),-1).double()
            return z-z.mean(-1,keepdim=True)
        m,p=embedding(target.detach()),embedding(prediction)
        eye=torch.eye(p.shape[-2],device=p.device,dtype=p.dtype)
        mm=m@m.transpose(-1,-2);pp=p@p.transpose(-1,-2);mp=m@p.transpose(-1,-2)
        cond=mm-mp@torch.linalg.solve(pp+5e-4*eye,mp.transpose(-1,-2))
        cond=(cond+cond.transpose(-1,-2))/2 + target_ridge*eye
        chol=torch.linalg.cholesky(cond)
        return torch.log(chol.diagonal(dim1=-2,dim2=-1)+1e-8).sum(-1).mean()

    def __call__(self, probabilities, labels):
        if labels.device!=probabilities.device:raise ValueError('CWMI labels and probabilities must share a device')
        pyramid=self.pyramid_for(probabilities.device)
        terms=[]
        for p,y in zip(probabilities,labels):
            h,w=y.shape
            if min(h,w)<128:
                raise ValueError('CWMI predefined crops require >=128x128')
            # All-valid full images, otherwise a fixed nonoverlapping geometric grid.
            regions=[(0,0,h,w)] if bool((y!=255).all()) else [
                (i,j,128,128) for i in range(0,h-127,128) for j in range(0,w-127,128)]
            for i,j,hh,ww in regions:
                lab=y[i:i+hh,j:j+ww]
                if not bool((lab!=255).all()):continue
                pred=p[None,1:,i:i+hh,j:j+ww]
                target=F.one_hot(lab,3).permute(2,0,1)[None,1:].to(pred)
                a,b=pyramid(target),pyramid(pred)
                terms.append(sum(self.complex_structure(a[n],b[n],self.ridge) for n in (1,2)))
        self.last_support=len(terms)
        return torch.stack(terms).mean().to(probabilities) if terms else probabilities.sum()*0


def second_order_map(u,margin=1e-3,beta=1.,eps=1e-8):
    """D-Convexity C2nd formula (MIT, Shengzhe Chen/Hao Yan, 2026).

    Tangential Hessian violation, NOT TV or Laplacian smoothness. Algebra follows
    the locked loss.py C2nd_loss including its gradient-magnitude convention.
    """
    ux=torch.diff(u,dim=-1,prepend=u[...,:1])
    uy=torch.diff(u,dim=-2,prepend=u[...,:1,:])
    uxx=torch.diff(ux,dim=-1,prepend=ux[...,:1])
    uyy=torch.diff(uy,dim=-2,prepend=uy[...,:1,:])
    uxy=(torch.diff(ux,dim=-2,prepend=ux[...,:1,:])+
         torch.diff(uy,dim=-1,prepend=uy[...,:1]))/2
    norm=(ux.square()+uy.square()+eps).sqrt()
    tx,ty=-uy/norm,ux/norm
    tangent=tx.square()*uxx+2*tx*ty*uxy+ty.square()*uyy
    return F.softplus((tangent+margin)*(norm>1e-5),beta=beta)*norm


def stencil_valid(geometry):
    # Backward differences consume offsets in the entire conservative 3x3 block.
    x=F.pad(geometry[:,None].float(),(2,0,2,0))
    return (F.avg_pool2d(x,3,stride=1)[:,0]==1).detach()


def convex_shape(probabilities,geometry):
    valid=stencil_valid(geometry)
    organs=torch.stack((probabilities[:,1]+probabilities[:,2],probabilities[:,2]),1)
    value=second_order_map(organs)
    return (value*valid[:,None]).sum()/(2*valid.sum().clamp_min(1))


def repair_target(teacher,x,geometry,shape_weight,scale=1.,steps=3,trust=.1,cached=None):
    with torch.no_grad():
        h = teacher.parts(x,mode="teacher")[1] if cached is None else cached[0]
        qbasis=teacher.sidecar.q.detach()
        previous=teacher.sidecar.previous.detach()
        def readout(value):
            return teacher.parent.native_readout(torch.einsum('ij,bjhw->bihw',previous,value),x.shape[-2:],'teacher')
        base=readout(h).softmax(1) if cached is None else cached[1]
    stats={'full_forwards':int(cached is None),'readout_only_forwards':0,'coordinate_vjps':0,'objectives':[],
           'coordinate_grid':[64,64],'max_correction':0.,'radius':0.}
    if shape_weight==0 or steps==0:return base.detach(),stats
    # A per-image scalar feature RMS bound guarantees the same bound after
    # bilinear interpolation (convex combinations) in feature coordinates.
    radius=trust*h.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
    stats['radius']=float(radius.max())
    d=h.new_zeros(h.shape[0],qbasis.shape[1],64,64)
    def corrected(d):
        up=F.interpolate(d,size=h.shape[-2:],mode='bilinear',align_corners=False)
        return h+torch.einsum('dk,bkhw->bdhw',qbasis,up)
    for _ in range(steps):
        with torch.enable_grad():
            d=d.detach().requires_grad_(True)
            logits=readout(corrected(d))
            objective=(masked_kl(logits,base,geometry)+shape_weight*scale*
                       convex_shape(logits.softmax(1),geometry)+.1*d.square().mean())
            if not torch.isfinite(objective):raise FloatingPointError('nonfinite repair objective')
            grad,=torch.autograd.grad(objective,d)
            if not torch.isfinite(grad).all():raise FloatingPointError('nonfinite repair gradient')
        stats['objectives'].append(float(objective.detach()))
        with torch.no_grad():
            d=d-.05*radius*grad/grad.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-12)
            d=d*torch.minimum(torch.ones_like(d[:,:1]),radius/d.norm(dim=1,keepdim=True).clamp_min(1e-12))
    with torch.no_grad():result=readout(corrected(d)).softmax(1)
    stats.update(readout_only_forwards=steps+1,coordinate_vjps=steps,max_correction=float(d.norm(dim=1).max()))
    return result.detach(),stats


def class_swd(student,teacher,u_classes,l_classes,u_valid,rng):
    """Both arguments are Q^T G h before F_prev, in registered feature grids."""
    s=F.normalize(student,dim=1,eps=1e-6).permute(0,2,3,1)
    t=F.normalize(teacher.detach(),dim=1,eps=1e-6).permute(0,2,3,1)
    terms=[];counts={}
    directions=torch.randn(s.shape[-1],32,generator=rng).to(s)
    for c in (1,2):
        a=s[(u_classes==c)&u_valid];b=t[l_classes==c]
        n=min(64,len(a),len(b));counts[c]=n
        if n<8:continue
        ia=torch.randperm(len(a),generator=rng)[:n].to(a.device)
        ib=torch.randperm(len(b),generator=rng)[:n].to(b.device)
        terms.append(sliced_wasserstein_equal(a[ia],b[ib],directions))
    return (torch.stack(terms).mean() if terms else student.sum()*0),counts
