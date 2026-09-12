"""Arm-specific core plasticity and the unchanged CIST transport; current L only."""
from experiments.lcrseg.cist_v0_1 import core as cist_core
from experiments.lcrseg.cist_v0_1.core import *
from experiments.lcrseg.lctx_weight_memory_v0_1.core import LAYERS,training_access

ARMS=('GN_L','GN_CIST_L','CONV_CIST_L')
MAIN='CONV_CIST_L'
BLOCKS=('enc1','enc2','enc3','bottleneck','decoder.dec3.merge','decoder.dec2.merge','decoder.dec1.merge')
GN=tuple(f'{b}.block.{i}.{p}' for b in BLOCKS for i in (1,4) for p in ('weight','bias'))
SIZES={'GN_L':1408,'GN_CIST_L':4328,'CONV_CIST_L':441112}

class Student(nn.Module):
    readout=cist_core.Student.readout
    def __init__(self,core,tensors,task):
        super().__init__();self.core=core.eval();self.arm=task['arm'];self.bypass=False
        if self.arm not in ARMS:raise PermissionError('unregistered plasticity arm')
        names=LAYERS if self.arm==MAIN else GN
        for name,p in core.named_parameters():p.requires_grad_(name in names)
        assert sum(p.numel() for p in core.parameters() if p.requires_grad)==(438192 if self.arm==MAIN else 1408)
        for b in BLOCKS:
            for i in (1,4):assert isinstance(core.get_submodule(f'{b}.block.{i}'),nn.GroupNorm)
        self.transport=None if self.arm=='GN_L' else Transport(tensors,'ISO_COND_L',task['seed'],task['source_domain']).to(next(core.parameters()).device)
        self.trainable_names=tuple(n for n,p in self.named_parameters() if p.requires_grad)
        assert tuple(n for n,p in core.named_parameters() if p.requires_grad)==tuple(names)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==SIZES[self.arm]
        # CPU source values for descriptive parameter drift; no second model or source-image forward.
        self.initial_values={n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
        self.observations=[]

    def features(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or any(n%8 for n in x.shape[-2:]):raise ValueError('input geometry')
        m=self.core;e1=m.enc1(x);e2=m.enc2(m.pool(e1));e3=m.enc3(m.pool(e2));b=m.bottleneck(m.pool(e3))
        h=m.decoder.dec1(m.decoder.dec2(m.decoder.dec3(b,e3),e2),e1)
        count('model_forward');count('image_forward',len(x));count('plastic_core_forward');count('plastic_core_images',len(x))
        return h

    def forward(self,x,*,stochastic_classifier=False):
        if stochastic_classifier:raise PermissionError('fixed raw linear head')
        h=self.features(x)
        if self.transport is None or self.bypass:hp=h;detail=None
        else:hp,detail=self.transport(h)
        if torch.is_grad_enabled():
            self.observations.append(dict(feature_requires_grad=h.requires_grad,**mechanism(h,hp,detail)))
        return self.readout(hp,x.shape[-2:]),(h,hp,detail)

    def fixed_state(self):return {n:t for n,t in self.state_dict().items() if n not in self.trainable_names}
    def dynamic_state(self):return {n:t for n,t in self.state_dict().items() if n in self.trainable_names}
    def restore_dynamic(self,state):
        if tuple(state)!=self.trainable_names:raise PermissionError('incomplete dynamic checkpoint')
        with torch.no_grad():
            for n,p in self.named_parameters():
                if n in state:p.copy_(state[n])

@torch.no_grad()
def mechanism(h,hp,detail):
    if detail is None:return dict(transport_present=False,distance_relative_max=0.,distance_relative_mean=0.,Q_norm=0.,b_norm=0.,Q_batch_variance=0.,b_batch_variance=0.,orthogonality64_max=0.)
    return dict(transport_present=True,**cist_core.mechanism(h,hp,detail))

def optimizer(model):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    expected=tuple(n for n in model.trainable_names if not (model.bypass and n.startswith('transport.')))
    assert tuple(n for n,p in named)==expected
    opt=optimizer_for(model)
    assert [id(p) for g in opt.param_groups for p in g['params']]==[id(p) for n,p in named]
    assert opt.defaults['lr']==.001 and opt.defaults['betas']==(.9,.999) and opt.defaults['eps']==1e-8 and opt.defaults['weight_decay']==4e-5
    return opt

def step(student,opt,l,task,epoch,index,counts,patients):
    opt.zero_grad(set_to_none=True);student.observations=[]
    ce,dice,coords=cist_core.labeled_loss(student,l,task['seed'],task['domain'],epoch,index,counts,patients);loss=ce+dice
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
    loss.backward();counts['backward']+=1
    for name,p in student.named_parameters():
        if not p.requires_grad and p.grad is not None:raise RuntimeError('gradient outside whitelist')
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient')
    grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in student.named_parameters() if p.requires_grad}
    observations=student.observations;student.observations=[]
    assert observations and all(r['feature_requires_grad'] for r in observations)
    opt.step();counts['optimizer_steps']+=1
    return dict(loss=float(loss.detach()),CE=float(ce.detach()),GT_Dice=float(dice.detach()),anchor_label_records=len(l['image']),
                labeled_source_scoring_multiplicity=1,donor_context_records=len(l['image']) if coords else 0,donor_extra_image_reads=0,
                U_image_records=0,pseudo_label_records=0,EMA_updates=0,mix_rectangles=coords,gradient_norms=grads,current_H_geometry=observations)

@torch.no_grad()
def drift(model):
    rows=[]
    for name,p in model.named_parameters():
        if name not in model.initial_values:continue
        source=model.initial_values[name].double();delta=p.detach().cpu().double()-source
        rows.append(dict(parameter=name,change_norm=float(delta.norm()),source_norm=float(source.norm()),relative_change=float(delta.norm()/source.norm()) if source.norm()>0 else None))
    return rows
