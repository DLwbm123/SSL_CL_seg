"""Explicit head modes; original numerical primitives remain immutable."""
import hashlib
from experiments.lcrseg.ssl_foundation_v0_1.core import *
from experiments.lcrseg.ssl_foundation_v0_1 import core as base
from experiments.lcrseg.di_dmpa_jascl.modeling import _official_probabilistic_classifier

ARMS=('LIN_SUP','LIN_MT_CONF','LIN_MT_PAS','SUP_G1')
LINEAR='raw_linear3_same_geometry';NORMAL='official_normalized_gas'
def head_mode(arm):
    if arm not in ARMS:raise ValueError('unregistered recipe')
    return NORMAL if arm=='SUP_G1' else LINEAR

def config(arm):
    head_mode(arm)
    return dict(gas=arm=='SUP_G1',ssl=arm.startswith('LIN_MT'),pas=arm=='LIN_MT_PAS')

class LinearHead(torch.nn.Module):
    def __init__(self,original):
        super().__init__();self.mu=original.mu;self.sigma=original.sigma;self.grad_update=original.grad_update
        self.sigma.requires_grad_(False);self.grad_update.requires_grad_(False)
    def forward(self,x,stochastic=False):
        if stochastic:raise PermissionError('linear head cannot sample')
        return F.conv2d(x,self.mu.weight,bias=None,stride=1,padding=0)

def set_head(model,mode):
    if mode==LINEAR:
        if not isinstance(model.decoder.conv_logit,LinearHead):model.decoder.conv_logit=LinearHead(model.decoder.conv_logit)
    elif mode!=NORMAL:raise ValueError('explicit head_mode required')
    model.head_mode=mode;return model

def build(reference,device,seed,domain,arm):return set_head(base.build(reference,device,seed,domain),head_mode(arm))

def from_state(reference,device,state,mode):
    if subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM:raise RuntimeError('upstream mismatch')
    subprocess.run(['git','-C',str(reference),'diff','--quiet','HEAD'],check=True)
    with rng(device,'head_load_import'):_official_probabilistic_classifier(reference,upstream_path=UPSTREAM_PATH)
    with rng(device,'head_load_meta'),torch.device('meta'):
        m=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
    m.load_state_dict(state,assign=True);state.clear();m.decoder.conv_logit.grad_update.requires_grad_(False)
    precision();return set_head(m,mode)

@torch.no_grad()
def local_gradient(ps,pt,mask,lam):
    delta=ps.detach()-pt.detach();g=2*ps.detach()*(delta-(ps.detach()*delta).sum(1,keepdim=True))
    return g*mask[:,None]*lam/mask.sum().clamp_min(1)

@torch.no_grad()
def class_loss_stats(ps,pt,mask,lam):
    ps,pt=ps.detach(),pt.detach();yt=pt.argmax(1);err=(ps-pt).square().sum(1);den=mask.sum().clamp_min(1);gz=local_gradient(ps,pt,mask,lam);out=[]
    for cls in range(3):
        m=mask&(yt==cls);n=int(m.sum());ss=float((err*m).sum());contribution=ss/int(den)
        out.append(dict(class_id=cls,accepted=n,squared_probability_sum=ss,probability_MSE_accepted_mean=ss/n if n else None,MSE_loss_contribution=contribution,weighted_MSE_loss_contribution=lam*contribution,weighted_local_logit_gradient_norm=float((gz*m[:,None]).double().square().sum().sqrt()),loss_denominator=int(den)))
    return out

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
    by_class=class_loss_stats(ps,pt,m,weight(epoch)) if spec['ssl'] and epoch>20 else []
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
    update_ema(student,teacher)
    if head_mode(arm)==LINEAR:
        with torch.no_grad():teacher.decoder.conv_logit.sigma.weight.copy_(student.decoder.conv_logit.sigma.weight)
    counters['ema_updates']+=1
    return dict(by_teacher_class=by_class,supervised=float(sup.detach()),consistency=float(con.detach()),loss=float(loss.detach()),lambda_cons=weight(epoch),accepted=accepted,u_pixels=u_pixels,disagreement=disagreement,gradients=grads,sigma_has_gradient=student.decoder.conv_logit.sigma.weight.grad is not None)

def optimizer_hash(opt):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(x.detach().cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(opt.state_dict());return h.hexdigest()
