"""One fixed private teacher cache; no U labels, teacher update or calibration claim."""
import torch
from torch.nn import functional as F
from ..semantic_evidence_v22.method import sample_semantics
from ..stream_state import digest


def read_image(data,index):
    x=data.read(data.records['U'][index],'image').to(getattr(torch,data.config.dtype))[None,None]
    return F.interpolate(x,(data.config.height,data.config.width),mode='bilinear',align_corners=False)


@torch.no_grad()
def build_cache(state,data,path):
    before=digest(state.payload());means=[];disagreements=[]
    for index in range(data.counts['U']):
        x=read_image(data,index).to(state.q.mean.device);outer_probs=[]
        global_rng=torch.Generator().manual_seed(251001)
        for outer in range(8):
            theta=state.q.rsample(global_rng);a=state.current.rsample(theta,global_rng)
            local_rng=torch.Generator().manual_seed(252000+100*index+outer)
            probs=[sample_semantics(state,x,theta,a,local_rng)[0] for _ in range(4)]
            outer_probs.append(torch.stack(probs).mean(0))
        p=torch.stack(outer_probs);means.append(p.mean(0)[0].cpu())
        disagreements.append(p.var(0,unbiased=False).sum(1)[0].cpu())
    mean=torch.stack(means);d=torch.stack(disagreements);tau=d.mean()
    w=torch.ones_like(d) if tau==0 else tau/(tau+d)
    assert torch.isfinite(mean).all() and torch.isfinite(w).all() and (w>0).all()
    assert torch.allclose(mean.sum(1),torch.ones_like(d),atol=1e-10)
    assert digest(state.payload())==before
    cache=dict(mean=mean,disagreement=d,weight=w,tau=tau,evidence_fingerprint=data.evidence['fingerprint'],teacher_payload=before,outer=8,local=4)
    with path.open('xb') as f:torch.save(cache,f)
    groups=mean.argmax(1);summary={}
    for label,name in [(0,'background'),(1,'foreground')]:
        ww=w[groups==label]
        summary[name]=dict(pixels=len(ww),fraction=float((groups==label).double().mean()),weight_mean=float(ww.mean()) if len(ww) else 0.,effective_pixels=float(ww.sum().square()/ww.square().sum()) if len(ww) else 0.)
    return cache,dict(status='COMPLETE',slices=len(mean),outer=8,local=4,global_draws=8*len(mean),semantic_prediction_forwards=32*len(mean),tau=float(tau),disagreement_mean=float(d.mean()),disagreement_max=float(d.max()),groups=summary,teacher_unchanged=True,labels_accessed=False,optimizer_updates=0)


class CachedData:
    def __init__(self,base,cache):
        self.base=base;self.cache=cache;self.evidence=base.evidence;self.counts=base.counts
        assert cache['evidence_fingerprint']==base.evidence['fingerprint']

    def batch(self,role,size,generator):
        if role=='L':return self.base.batch(role,size,generator)
        assert role=='U'
        index=torch.randint(self.counts['U'],(size,),generator=generator)
        return dict(x=torch.cat([read_image(self.base,i) for i in index.tolist()]),role='trainU',stage=self.base.stage,
                    teacher_mean=self.cache['mean'][index],teacher_weight=self.cache['weight'][index])
