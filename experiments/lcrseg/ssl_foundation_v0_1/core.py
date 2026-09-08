"""Six recipes share explicit RNG, single-domain data, objective, and EMA primitives."""
import contextlib, copy, hashlib, math, random, subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.lcrseg.di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, _official_probabilistic_classifier
from experiments.lcrseg.di_dmpa_jascl.data import stable_seed, batch_indices
from experiments.lcrseg.single_teacher_scd_v0_1.data import CurrentData, geometry, MANIFEST_SHA, SPLIT_SHA
from experiments.lcrseg.single_teacher_scd_v0_1.engine import precision, state_hash, tensor_bytes, write_json, append, optimizer_for, UPSTREAM, UPSTREAM_PATH
from experiments.lcrseg.single_teacher_scd_v0_1.objectives import supervised

DOMAINS=('RIM_ONE_r3','Drishti_GS')
ARMS=tuple(f'{kind}_G{gas}' for gas in (0,1) for kind in ('SUP','MT_CONF','MT_PAS'))
COUNTS={'RIM_ONE_r3':(16,63,40),'Drishti_GS':(10,41,25)}

def config(arm):
    if arm not in ARMS: raise ValueError('unregistered arm')
    return dict(gas=arm.endswith('G1'),ssl=not arm.startswith('SUP'),pas=arm.startswith('MT_PAS'))

@contextlib.contextmanager
def rng(device,*key):
    py,npstate=random.getstate(),np.random.get_state()
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        s=stable_seed(*key);torch.manual_seed(s);random.seed(s);np.random.seed(s%2**32)
        try:yield
        finally:random.setstate(py);np.random.set_state(npstate)

def build(reference,device,seed,domain):
    if domain not in DOMAINS:raise PermissionError('single authorized domain required')
    if subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM:raise RuntimeError('upstream mismatch')
    subprocess.run(['git','-C',str(reference),'diff','--quiet','HEAD'],check=True)
    # Import also resets global RNG/backend. Scope it before initializing the backbone.
    with rng(device,'upstream_import'):_official_probabilistic_classifier(reference,upstream_path=UPSTREAM_PATH)
    with rng(device,seed,domain,0,0,'initialization',0):
        model=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
    model.decoder.conv_logit.grad_update.requires_grad_(False)
    precision()
    return model.to(device)

def ema_from(student):
    teacher=copy.deepcopy(student).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return teacher

@torch.no_grad()
def update_ema(student,teacher):
    for name,p in student.named_parameters():
        q=dict(teacher.named_parameters())[name]
        if name.endswith('grad_update'):q.copy_(p)
        elif p.is_floating_point():q.mul_(.99).add_(p,alpha=.01)
        else:q.copy_(p)
    for name,b in student.named_buffers():
        q=dict(teacher.named_buffers())[name]
        if b.is_floating_point():q.mul_(.99).add_(b,alpha=.01)
        else:q.copy_(b)

class DomainData:
    def __init__(self,data,domain,role,*,evaluator=False,expected=(MANIFEST_SHA,SPLIT_SHA),shape=(384,384)):
        if domain not in DOMAINS or role not in (('val',) if evaluator else ('train_labeled','train_unlabeled')):raise PermissionError('single-domain role isolation')
        self.domain,self.role=domain,role
        self._dataset=CurrentData(data,DOMAINS.index(domain)+1,role,purpose='evaluate' if evaluator else 'train',expected=expected,shape=shape)
        self.opens=0
    def __len__(self):return len(self._dataset)
    def __getitem__(self,i):
        self.opens+=1
        return self._dataset[i]

def orders(n,seed,domain,epoch,stream,steps):
    result=[];cycle=0
    while len(result)<steps:
        result.extend(v for _,v in batch_indices(n,2,shuffle=True,seed_parts=(seed,domain,epoch,cycle,stream,0)));cycle+=1
    return result[:steps]

def batch(ds,indices,device,seed,epoch,step,stream,augment=True):
    out=[]
    for pos,i in enumerate(indices):
        item=ds[i]
        if augment:
            g=torch.Generator().manual_seed(stable_seed(seed,ds.domain,epoch,step,stream+'_geometry',pos))
            x,y,v=geometry(item['image'],item.get('label'),item['geometry'],g)
            item=dict(image=x,geometry=v)
            if y is not None:item['label']=y
        out.append(item)
    return {k:torch.stack([r[k] for r in out]).to(device) for k in out[0]}

def fwd(model,x,gas,key,counters,stream):
    with rng(x.device,*key,stream,0):result=model(x,stochastic_classifier=gas)
    counters[stream+'_forward']+=1;counters[stream+'_images']+=len(x)
    return result

def noisy(x,key):
    return torch.stack([(image+.02*torch.randn(image.shape,device=x.device,generator=torch.Generator(device=x.device).manual_seed(stable_seed(*key,'input_noise',pos)))).clamp(0,1) for pos,image in enumerate(x)])

@torch.no_grad()
def centers(feature,label):
    out=feature.new_zeros(3,feature.shape[0]);support=torch.zeros(3,dtype=torch.bool,device=feature.device)
    for c in range(3):
        m=label==c
        if m.any():
            v=feature[:,m].mean(1);norm=v.norm()
            if norm>0:out[c]=v/norm;support[c]=True
    return out,support

@torch.no_grad()
def prototypes(student,ds,device,counters,seed,epoch,stream):
    was=student.training;student.eval();total=torch.zeros(3,16,device=device);n=torch.zeros(3,device=device)
    try:
        for i in range(len(ds)):
            item=ds[i];_,feat=fwd(student,item['image'][None].to(device),False,(seed,ds.domain,epoch,i),counters,stream)
            p,s=centers(feat[0],item['label'].to(device));total+=p;n+=s
    finally:student.train(was)
    p=total/n.clamp_min(1)[:,None];supported=(n>0)&(p.norm(dim=1)>0)
    return p,supported

@torch.no_grad()
def masks(ps,pt,fs,ft,proto,support,geometry):
    ps,pt,fs,ft=ps.detach(),pt.detach(),fs.detach(),ft.detach()
    cs,ys=ps.max(1);ct,yt=pt.max(1)
    conf=geometry.bool()&(cs>.7)&(ct>.7)
    if proto is None:return conf,None
    sim_s=F.cosine_similarity(fs.movedim(1,-1),proto[ys],dim=-1)
    sim_t=F.cosine_similarity(ft.movedim(1,-1),proto[yt],dim=-1)
    pas=conf&support[ys]&support[yt]&(sim_s>.7)&(sim_t>.7)
    return conf,pas

def consistency(ps,pt,mask):
    m=mask.detach();return ((ps-pt.detach()).square().sum(1)*m).sum()/m.sum().clamp_min(1)

def weight(epoch):return .5*min(1.,max(0.,(epoch-20)/20))

def grad_norm(loss,model):
    gs=torch.autograd.grad(loss,[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
    return sum(float(g.detach().double().square().sum()) for g in gs if g is not None)**.5

def train_step(student,teacher,opt,l,u,proto,support,arm,seed,domain,epoch,step,counters,diagnostic=False):
    spec=config(arm);opt.zero_grad(set_to_none=True);key=(seed,domain,epoch,step)
    z,_=fwd(student,l['image'],spec['gas'],key,counters,'student_l')
    sup=supervised(z,l['label']);gas=torch.zeros_like(student.decoder.conv_logit.grad_update)
    if spec['gas']:
        gas=torch.autograd.grad(sup,student.decoder.conv_logit.mu.weight,retain_graph=True)[0].detach().square();counters['gas_autograd']+=1
    con=z.sum()*0;accepted=0;u_pixels=0;disagreement=0
    if spec['ssl'] and epoch>20:
        if u is None:raise ValueError('missing U')
        zs,fs=fwd(student,noisy(u['image'],key),spec['gas'],key,counters,'student_u')
        with torch.no_grad():zt,ft=fwd(teacher,u['image'],spec['gas'],key,counters,'ema_u')
        ps=zs.softmax(1);pt=zt.softmax(1).detach()
        conf,pas=masks(ps,pt,fs,ft,proto,support,u['geometry']);m=pas if spec['pas'] else conf
        if m is None:raise ValueError('PAS without training prototypes')
        con=consistency(ps,pt,m);accepted=int(m.sum());u_pixels=int(u['geometry'].sum());disagreement=int(((ps.argmax(1)!=pt.argmax(1))&u['geometry']).sum())
    elif u is not None:raise PermissionError('SUP/warm-up received U')
    loss=sup+weight(epoch)*con
    grads={}
    if diagnostic:
        grads={k:grad_norm(v,student) for k,v in [('supervised',sup),('consistency',con),('weighted_consistency',weight(epoch)*con)]};counters['diagnostic_autograd']+=3
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();counters['backward']+=1
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):raise FloatingPointError('nonfinite gradient')
    if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('EMA gradient')
    opt.step();counters['optimizer_steps']+=1
    with torch.no_grad():student.decoder.conv_logit.grad_update.copy_(gas)
    update_ema(student,teacher);counters['ema_updates']+=1
    return dict(supervised=float(sup.detach()),consistency=float(con.detach()),loss=float(loss.detach()),lambda_cons=weight(epoch),accepted=accepted,u_pixels=u_pixels,disagreement=disagreement,gradients=grads,sigma_has_gradient=student.decoder.conv_logit.sigma.weight.grad is not None)
