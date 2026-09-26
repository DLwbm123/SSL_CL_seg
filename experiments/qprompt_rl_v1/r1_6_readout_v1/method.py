"""Additional FP32 readout supervision; frozen matching and deployment unchanged."""
import ast
import inspect
import math
import torch
from torch.nn import functional as F
from qprompt import losses as old
from qprompt.models import DINOv2Adapter


def readout_loss(p, labels):
    if p.ndim!=4 or p.shape[1]!=3 or labels.shape!=p.shape[:1]+p.shape[2:]:raise ValueError('readout geometry')
    if labels.dtype not in (torch.int64,torch.int32):raise ValueError('integer labels required')
    if not torch.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('invalid probabilities')
    if not torch.allclose(p.float().sum(1),torch.ones_like(p[:,0]).float(),rtol=0,atol=.002):raise ValueError('probabilities must be normalized')
    if not ((labels==255)|((labels>=0)&(labels<=2))).all():raise ValueError('invalid labels')
    with torch.autocast(p.device.type,enabled=False):
        p=p.float();valid=labels!=255;active=valid.flatten(1).any(1)
        if not active.any():
            zero=p.sum()*0
            return dict(total=zero,nll=zero,dice=zero,log_floor_fraction=0.)
        truth=labels.masked_fill(~valid,0).long();chosen=p.gather(1,truth[:,None]).squeeze(1)
        nll=(-chosen.clamp_min(1e-8).log()*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
        dice=[]
        for c in (1,2):
            pred=p[:,c]*valid;gt=(labels==c)&valid
            dice.append(1-(2*(pred*gt).flatten(1).sum(1)+1)/(pred.flatten(1).sum(1)+gt.flatten(1).sum(1)+1))
        d=(dice[0]+dice[1])/2
        return dict(total=(nll[active]+d[active]).mean(),nll=nll[active].mean(),dice=d[active].mean(),log_floor_fraction=float((chosen[valid]<1e-8).float().mean().detach()))


def objective(model,x,y,arm,bank=None):
    out=model(x)
    with torch.autocast(x.device.type,enabled=False):
        terms={'Lset':old.supervised_query_loss(out['class_logits'].float(),out['mask_logits'].float(),y)['total']}
        current=support=None
        if arm in ('B1','B3'):
            if bank is None:raise ValueError('bank required')
            current,support=old.image_prototypes(out['pixels'].float(),y,vit_pad=isinstance(model,DINOv2Adapter))
            terms['Limg']=old.image_alignment_loss(current,support,bank)
        elif bank is not None:raise ValueError('bank forbidden')
        read={}
        if arm in ('B2','B3'):
            rd=readout_loss(out['semantic'],y);terms.update(Lreadout=rd['total'],Lnll=rd['nll'],Ldice=rd['dice']);read['log_floor_fraction']=rd['log_floor_fraction']
        total=terms['Lset']+10*terms.get('Limg',0)+terms.get('Lreadout',0)
        with torch.no_grad():
            mass=torch.einsum('bkc,bkhw->bhw',out['class_logits'].float().softmax(-1)[...,:3],out['mask_logits'].float().sigmoid())
            read['uniform_fallback_fraction']=float((mass<1e-12).float().mean())
        if not torch.isfinite(total):raise FloatingPointError('nonfinite objective')
    return total,terms,read,current,support,out


# Export the original matcher's assignments without changing any cost or selection line.
_tree=ast.parse(inspect.getsource(old.supervised_query_loss))
_return=next(n for n in ast.walk(_tree) if isinstance(n,ast.Return))
_return.value.keywords.append(ast.keyword(arg='assignments',value=ast.Name(id='targets',ctx=ast.Load())))
ast.fix_missing_locations(_tree)
_ns=dict(old.__dict__);exec(compile(_tree,'<original-matcher-with-assignment-export>','exec'),_ns)
matched_assignments=_ns['supervised_query_loss']


def gradient_blocks(model, terms):
    named=list(model.named_parameters());params=tuple(p for _,p in named)
    losses={'set':terms['Lset'],'readout':terms.get('Lreadout',terms['Lset']*0),'img':10*terms.get('Limg',terms['Lset']*0)}
    gradients={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in losses.items()}
    groups={'all':list(range(len(named))),'body':[i for i,(n,_) in enumerate(named) if n.startswith(('body.','backbone.'))],'readout':[i for i,(n,_) in enumerate(named) if not n.startswith(('body.','backbone.'))]}
    result={}
    for block,ids in groups.items():
        norms={k:math.sqrt(sum(float(g[i].detach().float().square().sum()) for i in ids if g[i] is not None)) for k,g in gradients.items()}
        row={'norms':norms,'cosines':{},'null_reasons':{}}
        for a,b in [('set','readout'),('set','img'),('readout','img')]:
            key=a+'_'+b
            if not norms[a] or not norms[b]:row['cosines'][key]=None;row['null_reasons'][key]='ZERO_GRADIENT_NORM'
            else:
                dot=sum(float((gradients[a][i].detach().float()*gradients[b][i].detach().float()).sum()) for i in ids if gradients[a][i] is not None and gradients[b][i] is not None)
                row['cosines'][key]=dot/(norms[a]*norms[b])
        result[block]=row
    return result
