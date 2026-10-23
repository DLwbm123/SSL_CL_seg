"""Fixed Gaussian graph mean around existing L-neighbor votes, no model calls."""
import torch
from ..posterior_teacher_v25.teacher import read_image


def edges(x,adaptive):
    dx=(x[:,1:]-x[:,:-1]).square();dy=(x[1:,:]-x[:-1,:]).square()
    scale=(dx.sum()+dy.sum())/(dx.numel()+dy.numel())
    if not adaptive or float(scale)==0:return torch.ones_like(dx),torch.ones_like(dy)
    return torch.exp(-dx/(2*scale)),torch.exp(-dy/(2*scale))


def apply(v,wx,wy):
    # Each undirected edge counted once; natural boundaries, no wrapping.
    out=v.clone();dx=(v[:,:-1]-v[:,1:])*wx/4;dy=(v[:-1,:]-v[1:,:])*wy/4
    out[:,:-1]+=dx;out[:,1:]-=dx;out[:-1,:]+=dy;out[1:,:]-=dy
    return out


@torch.no_grad()
def solve(r,wx,wy):
    z=r.clone();res=r-apply(z,wx,wy);direction=res.clone();rr=(res*res).sum()
    threshold=1e-10*max(1.,float(torch.linalg.vector_norm(r)));iterations=0
    for _ in range(128):
        if float(rr.sqrt())<=threshold:break
        ad=apply(direction,wx,wy);alpha=rr/(direction*ad).sum()
        z+=alpha*direction;res-=alpha*ad;newrr=(res*res).sum()
        direction=res+(newrr/rr)*direction;rr=newrr;iterations+=1
    residual=float(torch.linalg.vector_norm(apply(z,wx,wy)-r))
    assert residual<=threshold*1.01 and torch.isfinite(z).all()
    assert z.min()>=-1e-9 and z.max()<=1+1e-9
    # Only remove floating point excursions; record maximum correction.
    clipped=z.clamp(0,1);correction=float((clipped-z).abs().max());z=clipped
    assert abs(float(z.mean()-r.mean()))<1e-9
    return z,dict(iterations=iterations,residual=residual,clamp_max=correction)


def fixture():
    x=torch.tensor([[0.,0.,2.],[0.,1.,2.],[0.,1.,2.]],dtype=torch.float64)
    r=torch.tensor([[0.,0.,1.],[0.,.5,1.],[0.,0.,1.]],dtype=torch.float64)
    for adaptive in (False,True):
        wx,wy=edges(x,adaptive);z,info=solve(r,wx,wy)
        eye=torch.eye(9,dtype=torch.float64)
        matrix=torch.stack([apply(v.reshape(3,3),wx,wy).reshape(-1) for v in eye],1)
        assert torch.allclose(matrix,matrix.T) and torch.linalg.eigvalsh(matrix).min()>=1-1e-12
        exact=torch.linalg.solve(matrix,r.reshape(-1)).reshape(3,3)
        assert torch.allclose(z,exact,atol=1e-10,rtol=1e-10)
        constant=torch.full_like(r,.3);assert torch.allclose(solve(constant,wx,wy)[0],constant)
        assert torch.allclose(solve(1-r,wx,wy)[0],1-z,atol=1e-10,rtol=1e-10)
    zero=torch.zeros_like(x);a,b=edges(zero,True);assert torch.equal(a,torch.ones_like(a)) and torch.equal(b,torch.ones_like(b))
    wx,wy=edges(x,True);assert wx[0,0]>wx[0,1]
    return dict(dense_solution_match=True,constant_and_mass_preservation=True,label_complement=True,flat_image_finite=True,image_edge_attenuation=True,fixture_reverse_calls=0)


@torch.no_grad()
def build_cache(data,base,root):
    assert len(base['mean'])==data.counts['U']==271
    fields={a:[] for a in 'IE'};stats={a:[] for a in 'IE'};changes={a:[] for a in 'IE'}
    for index,r in enumerate(base['mean'][:,1]):
        x=read_image(data,index)[0,0].double().cpu();r=r.double().cpu()
        assert x.shape==r.shape==(128,128)
        for arm in 'IE':
            z,info=solve(r,*edges(x,arm=='E'))
            fields[arm].append(torch.stack((1-z,z)));stats[arm].append(info);changes[arm].append(float((z-r).abs().mean()))
    caches={};summary={}
    for arm in 'IE':
        m=torch.stack(fields[arm]);cache=dict(mean=m,weight=torch.ones_like(m[:,0]),evidence_fingerprint=base['evidence_fingerprint'],teacher_payload=base['teacher_payload'],target_source='image_graph_'+arm)
        with (root/f'PRIVATE_GRAPH_{arm}.pt').open('xb') as f:torch.save(cache,f)
        caches[arm]=cache
        summary[arm]=dict(mean_foreground_mass=float(m[:,1].mean()),argmax_foreground_fraction=float((m.argmax(1)==1).double().mean()),target_entropy=float(-(m*m.clamp_min(1e-12).log()).sum(1).mean()),mean_absolute_change=sum(changes[arm])/271,max_solver_iterations=max(s['iterations'] for s in stats[arm]),max_residual=max(s['residual'] for s in stats[arm]),max_clamp_correction=max(s['clamp_max'] for s in stats[arm]))
    return caches,dict(status='COMPLETE',arms=summary,U_slices=271,graph_solves=542,model_forwards=0,source_updates=0,U_labels_read=False,L_labels_read=False,existing_neighbor_cache_reused=True,solver_dtype='float64',source_unchanged=True)
