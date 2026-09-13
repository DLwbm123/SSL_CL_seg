"""Stateless paired randomness and source-preserving common inputs."""
import hashlib
import torch
from .kernels import collect_sources


def generator(seed, order, stage, step, stream):
    key = f'SSLCL_FIVE_FRAMEWORKS_V1/{seed}/{order}/{stage}/{step}/{stream}'
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1))


def complementary(anchor, donor, rng, ratio=2/3, noise=.02):
    if anchor.shape != donor.shape:
        raise ValueError('anchor and context must have identical registered geometry')
    b, _, h, w = anchor.shape
    rh, rw = max(1, round(h*ratio)), max(1, round(w*ratio))
    m = torch.zeros(b,h,w,dtype=torch.bool,device=anchor.device)
    for i in range(b):
        y = int(torch.randint(h-rh+1, (), generator=rng))
        x = int(torch.randint(w-rw+1, (), generator=rng))
        m[i,y:y+rh,x:x+rw] = True
    donor = donor + noise*torch.randn(donor.shape,generator=rng).to(donor)
    return torch.where(m[:,None],anchor,donor), torch.where(m[:,None],donor,anchor), m


def collected_forward(forward, anchor, donor, rng, scales=None):
    a,b,m = complementary(anchor,donor,rng)
    scale_a, scale_b = (None,None) if scales is None else scales(m)
    pa = forward(a,scale_a).log_softmax(1)
    pb = forward(b,scale_b).log_softmax(1)
    return collect_sources(pa,pb,m)[0], m


def unique_indices(ids):
    seen=set(); indices=[]
    for i,key in enumerate(ids):
        if key not in seen:
            indices.append(i);seen.add(key)
    return indices


class SyntheticCurrentDomain:
    """Separate L and U capabilities. No U-label storage or accessor exists."""
    def __init__(self, seed=161, order=1, stage=1, size=16, manifest_id="synthetic_v1", split_id="current_L_U", stage_source=None):
        self.seed,self.order,self.stage,self.size=seed,order,stage,size
        self.l_reads=self.u_reads=0
        self.manifest_id,self.split_id=manifest_id,split_id
        self.stage_source=stage_source or {'kind':'synthetic_entry','seed':seed,'order':order,'preceding_stage':stage-1}

    def semantic_metadata(self):
        return {'provider':type(self).__module__+'.'+type(self).__qualname__,
                'seed':self.seed,'order':self.order,'stage':self.stage,'size':self.size,
                'manifest':self.manifest_id,'split':self.split_id,'stage_source':self.stage_source,
                'stream':'sha256(study/seed/order/stage/cursor/stream)_v1_no_arm',
                'cursor':'successful_step; stateless two-source batches',
                'labels':{'background':0,'rim':1,'cup':2,'ignore':255}}


    def labeled(self, step, stream='labeled'):
        self.l_reads+=1
        g=generator(self.seed,self.order,self.stage,step,stream)
        x=torch.randn(2,3,self.size,self.size,generator=g)
        labels=torch.randint(3,(2,self.size,self.size),generator=g)
        return x,labels,('synthetic_patient_0','synthetic_patient_1')

    def unlabeled(self, step):
        self.u_reads+=1
        g=generator(self.seed,self.order,self.stage,step,'unlabeled')
        x=torch.randn(2,3,self.size,self.size,generator=g)
        return x,torch.ones(2,self.size,self.size,dtype=torch.bool),('u0','u1')


def mixed_feature_scales(parent,mask,uncertainty_u,uncertainty_context,feature_shape,spectrum,kappa):
    from .kernels import uncertainty_scales
    ua=torch.where(mask,uncertainty_u,uncertainty_context)
    ub=torch.where(mask,uncertainty_context,uncertainty_u)
    a=parent.output_to_feature(ua,feature_shape)
    b=parent.output_to_feature(ub,feature_shape)
    return uncertainty_scales(a,spectrum,kappa),uncertainty_scales(b,spectrum,kappa)
