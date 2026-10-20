"""Native spatial adapter; shared paired GYRO objective."""
import copy
import torch
from torch import nn
from .module import SpatialProjection,JointGenerator,unit,frame,rows,sample_positions,centroid,anchored_loss,rate_gap
from ..space_lora_v1.core import source_model
from ..siemens_ssl_reproduction_v1.methods import pixel_ce,strong_view

class Model(nn.Module):
    def __init__(self,parent):
        super().__init__();self.parent=parent;self.projection=SpatialProjection(128);self.generator=JointGenerator(1)
        self.code=nn.Parameter(torch.zeros(32,32,8,8))
        self.register_buffer('frames',torch.stack([frame(90163,10,32),frame(90164,10,32)]))
    def forward(self,x):return self.parent.native_readout(self.parent.features(x),x.shape[-2:])
    def encode(self,x):
        n=self.parent.native;h=n.enc1(x);h=n.enc2(n.pool(h));h=n.enc3(n.pool(h));h=n.bottleneck(n.pool(h))
        return self.projection(h)
    def teacher(self):return copy.deepcopy(self).eval().requires_grad_(False)

def build(cfg):
    torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    source=source_model(cfg)
    saved=torch.load(cfg['calibrated'],map_location='cpu',weights_only=False)
    assert saved['step']==saved['physical_updates']==69
    source.parent.load_state_dict({k[7:]:v for k,v in saved['model'].items() if k.startswith('parent.')})
    model=Model(copy.deepcopy(source.parent)).cuda()
    assert all(torch.equal(v.cpu(),saved['model']['parent.'+k]) for k,v in model.parent.state_dict().items())
    del source,saved
    return model

def soft_ce(logits,mask):return -(mask*logits.log_softmax(1)).sum(1).mean()

def code_indices(step):
    g=torch.Generator().manual_seed(163+step//4)
    return torch.randperm(32,generator=g)[(step%4)*8:(step%4+1)*8].cuda()

def augmented(x,step):
    value=strong_view(x,step)
    # Exact legacy forward; explicitly declared straight-through input Jacobian.
    return value+(x-x.detach()) if x.requires_grad else value

def geometry(model,x,step,reference=None,memory_count=0,arm='C'):
    pos=sample_positions(len(x),step,x.device)
    fields=model.encode(torch.cat([x,augmented(x,step),augmented(x,step+2000)]))
    a,b,c=[rows(f,pos) for f in fields.chunk(3)]
    center=centroid(a,b,c)
    if reference is not None:
        center=torch.cat([center[:-memory_count*4],centroid(a[-memory_count*4:],b[-memory_count*4:],c[-memory_count*4:],rows(reference,pos[-memory_count:]))])
    ids=torch.tensor([1]*((len(x)-memory_count)*4)+[0]*(memory_count*4),device=x.device)
    loss,_=anchored_loss(center,model.frames,ids)
    if arm=='E':loss=loss+rate_gap(rows(reference,pos[-memory_count:]),a[-memory_count*4:])
    return loss

def target_setup(cfg):
    model=build(cfg)
    memory=torch.load(cfg['memory'],map_location='cpu',weights_only=False)
    model.projection.load_state_dict(memory['projection']);model.generator.load_state_dict(memory['generator']);model.code.data.copy_(memory['code']);model.frames.copy_(memory['frames'])
    model.parent.configure_stage_training();model.projection.requires_grad_(False).eval();model.generator.requires_grad_(False).eval();model.code.requires_grad_(False)
    model.cfg=cfg;model.arm=cfg['arm']
    groups=[dict(params=[p for p in model.parent.parameters() if p.requires_grad and (p.ndim>1)==matrix],lr=.001,fixed_lr=.001,name='native_'+str(matrix),weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
    opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True)
    from .runtime import attach
    attach(opt,cfg,cfg['arm'],cfg['phase'])
    return model.teacher(),model,model.teacher(),opt

def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);u,_,_=data.batch('U',step,14)
    indices=code_indices(step)
    with torch.no_grad():
        z=unit(model.code[indices]);images,mask=model.generator(z,x.shape[-2:])
    ce=pixel_ce(model(x),y).mean()+soft_ce(model(images),mask)
    reference=None
    if arm in ('B','C','D'):reference=unit(model.code[(indices+1)%32]) if arm=='D' else z
    # E uses ordinary image centroids and only adds the ensemble closure.
    allx=torch.cat([x,u,images])
    geo=geometry(model,allx,step,reference,8,arm='C')
    if arm=='E':
        pos=sample_positions(len(allx),step,allx.device)[-8:]
        geo=geo+rate_gap(rows(z,pos),rows(model.encode(images),pos))
    return ce,.1*geo,dict(supervised=float(ce.detach()),unsupervised=float(.1*geo.detach()),accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=96,geometry_images=72,historical_raw_reads=0)

def update_ema(ema,model):
    with torch.no_grad():
        for k,v in ema.state_dict().items():
            src=model.state_dict()[k]
            if k.startswith('parent.') and v.is_floating_point() and not k.endswith(('sigma.weight','grad_update')):v.mul_(.999).add_(src,alpha=.001)
            else:v.copy_(src)
