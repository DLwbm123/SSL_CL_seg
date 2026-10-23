"""Fixed image-feature neighbors anchored only by observed training-L labels."""
import torch
from torch.nn import functional as F
from ..data_adapter import leaf_observations
from ..stream_state import digest
from ..posterior_teacher_v25.teacher import read_image


def feature(phi,x):
    image=x
    if phi.config.coordinate_channels:
        yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[-2],device=x.device,dtype=x.dtype),torch.linspace(-1,1,x.shape[-1],device=x.device,dtype=x.dtype),indexing='ij')
        image=torch.cat((x,torch.stack((xx,yy))[None].expand(len(x),-1,-1,-1)),1)
    h=torch.tanh(phi.enc(image));low=torch.tanh(phi.down(F.avg_pool2d(h,2)))
    return torch.tanh(phi.up(torch.cat((h,F.interpolate(low,x.shape[-2:],mode='bilinear',align_corners=False)),1)))


@torch.no_grad()
def votes(query,bank,labels):
    assert query.ndim==bank.ndim==2 and query.shape[1]==bank.shape[1] and len(bank)>=8
    assert labels.shape==(len(bank),) and set(labels.unique().tolist())<={0,1}
    query=F.normalize(query.float(),dim=1,eps=1e-8)
    bank=F.normalize(bank.float(),dim=1,eps=1e-8)
    result=[]
    for part in query.split(512):
        nearest=(part@bank.T).topk(8,dim=1,sorted=False).indices
        fg=labels[nearest].double().mean(1)
        result.append(torch.stack((1-fg,fg),1))
    return torch.cat(result)


def fixture():
    # Two orthogonal clusters with eight labeled anchors each, no tie across classes.
    bank=torch.cat((torch.tensor([[1.,0.]]).repeat(8,1),torch.tensor([[0.,1.]]).repeat(8,1)))
    y=torch.cat((torch.zeros(8,dtype=torch.long),torch.ones(8,dtype=torch.long)))
    q=torch.eye(2);p=votes(q,bank,y)
    assert torch.equal(p,torch.eye(2,dtype=torch.float64))
    assert torch.equal(votes(q,bank,1-y),p.flip(1))
    assert torch.equal(votes(q,bank,y*0),torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64))
    return dict(known_neighbors=True,label_permutation_changes_targets=True,all_background_finite=True,fixture_reverse_calls=0)


@torch.no_grad()
def build_cache(s,data,path):
    before=digest(s.payload());bank=[];labels=[]
    assert s.config.height==s.config.width==128 and data.counts['L']==80 and data.counts['U']==271
    for record in data.records['L']:
        x=data.read(record,'image').to(s.q.mean)[None,None]
        x=F.interpolate(x,(128,128),mode='bilinear',align_corners=False)
        raw=data.read(record,'label')[None,None].double().to(x.device)
        y=F.interpolate(raw,(128,128),mode='nearest')[:,0].long()
        obs=leaf_observations(y,s.config.classes,s.config.label_groups)
        assert obs['obs_mask'].all() and obs['y_obs'].shape[1]==2
        z=feature(s.phi,x)[:,:,4::8,4::8].flatten(2).transpose(1,2)[0]
        bank.append(z.float());labels.append(obs['y_obs'].long().argmax(1)[:,4::8,4::8].reshape(-1))
    bank=torch.cat(bank);labels=torch.cat(labels);assert len(bank)==80*256
    means=[]
    for index in range(data.counts['U']):
        x=read_image(data,index).to(s.q.mean.device)
        z=feature(s.phi,x).flatten(2).transpose(1,2)[0]
        means.append(votes(z,bank,labels).T.reshape(2,128,128).cpu())
    mean=torch.stack(means);w=torch.ones_like(mean[:,0])
    assert torch.isfinite(mean).all() and torch.allclose(mean.sum(1),w,atol=1e-12)
    assert digest(s.payload())==before
    cache=dict(mean=mean,weight=w,evidence_fingerprint=data.evidence['fingerprint'],teacher_payload=before,target_source='training_L_feature_neighbors')
    with path.open('xb') as f:torch.save(cache,f)
    fg=mean[:,1];entropy=-(mean*mean.clamp_min(1e-12).log()).sum(1)
    return cache,dict(status='COMPLETE',L_slices=80,U_slices=271,anchors=len(bank),anchors_foreground=int(labels.sum()),neighbors=8,feature_dimensions=bank.shape[1],backbone_forwards=351,semantic_prediction_forwards=0,global_posterior_draws=0,mean_foreground_mass=float(fg.mean()),argmax_foreground_fraction=float((mean.argmax(1)==1).double().mean()),target_entropy=float(entropy.mean()),source_unchanged=True,U_labels_read=False,L_labels_read=True,optimizer_updates=0,feature_affinity_dtype='float32',target_dtype='float64')
