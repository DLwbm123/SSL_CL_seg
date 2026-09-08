"""Fixed joint/target mask × MSE/soft-CE; the qualified linear head is unchanged."""
from experiments.lcrseg.ssl_head_control_v0_1.core import *
from experiments.lcrseg.ssl_head_control_v0_1 import core as old
ARMS=('SUP','J_MSE','T_MSE','J_SCE','T_SCE');SEEDS=(31,32,33)
def config(arm):
    if arm not in ARMS:raise ValueError('unregistered arm')
    return dict(ssl=arm!='SUP',gas=False,pas=False,mask=arm[:1],loss=arm.split('_')[-1])
def head_mode(arm):config(arm);return LINEAR
def build(reference,device,seed,domain,arm):config(arm);return old.build(reference,device,seed,domain,'LIN_SUP')
def validate_pair(z,q,geometry):
    if not all(torch.is_tensor(x) for x in (z,q,geometry)):raise TypeError('tensor inputs required')
    if z.ndim!=4 or z.shape!=q.shape or z.shape[1]!=3 or min(z.shape)==0 or geometry.shape!=z.shape[:1]+z.shape[2:]:raise ValueError('B3HW and BHW required')
    if z.dtype not in (torch.float32,torch.float64) or q.dtype!=z.dtype or geometry.dtype!=torch.bool:raise TypeError('matching float32/64 and bool geometry required')
    if z.device!=q.device or z.device!=geometry.device:raise ValueError('mixed device')
    if not torch.isfinite(z).all() or not torch.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('nonfinite or invalid probabilities')
    if not torch.allclose(q.sum(1),torch.ones_like(q[:,0]),atol=1e-6,rtol=1e-6):raise ValueError('unnormalized q')
def validate_batch(item,labeled):
    x=item['image'];g=item['geometry']
    if x.ndim!=4 or x.shape[1]!=3 or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('finite float32 B3HW images required')
    if g.dtype!=torch.bool or g.shape!=x.shape[:1]+x.shape[2:] or g.device!=x.device:raise ValueError('invalid geometry')
    if labeled:
        y=item['label']
        if y.dtype!=torch.long or y.shape!=g.shape or y.device!=x.device or not ((y>=0)&(y<3)|(y==255)).all():raise ValueError('integer class0/1/2/255 labels required')
    elif 'label' in item:raise PermissionError('U labels forbidden')
@torch.no_grad()
def target_masks(p,q,geometry):
    t=geometry&(q.detach().max(1)[0]>.7);return t&(p.detach().max(1)[0]>.7),t

def objectives(z,q,geometry):
    validate_pair(z,q,geometry);q=q.detach();p=z.softmax(1);j,t=target_masks(p,q,geometry)
    return p,q,j,t,dict(MSE=(p-q).square().sum(1),SCE=-(q*z.log_softmax(1)).sum(1),entropy=-torch.xlogy(q,q).sum(1))
def reduce_loss(values,mask):return (values*mask.detach()).sum()/mask.sum().clamp_min(1)
@torch.no_grad()
def gradients(p,q,mask,lam,loss):
    if loss not in ('MSE','SCE'):raise ValueError('unregistered objective')
    delta=p.detach()-q.detach();raw=delta if loss=='SCE' else 2*p.detach()*(delta-(p.detach()*delta).sum(1,keepdim=True))
    return raw,raw*mask[:,None]*lam/mask.sum().clamp_min(1)
@torch.no_grad()
def class_stats(p,q,mask,values,lam,loss):
    raw,g=gradients(p,q,mask,lam,loss);yt=q.argmax(1);den=int(mask.sum().clamp_min(1));out=[]
    for cls in range(3):
        m=mask&(yt==cls);n=int(m.sum());v=float((values[loss]*m).sum())/den;ent=float((values['entropy']*m).sum())/den;sce=float((values['SCE']*m).sum())/den
        out.append(dict(class_id=cls,accepted=n,denominator=den,raw_loss_contribution=v,weighted_loss_contribution=lam*v,SCE=sce,target_entropy=ent,KL=sce-ent,local_logit_gradient_norm=float((g*m[:,None]).double().square().sum().sqrt())))
    return out

def train_step(student,teacher,opt,l,u,proto,support,arm,seed,domain,epoch,step,counters,diagnostic=False):
    spec=config(arm)
    if proto is not None or support is not None:raise PermissionError('no prototypes')
    validate_batch(l,True);opt.zero_grad(set_to_none=True);key=(seed,domain,epoch,step)
    z,_=fwd(student,l['image'],False,key,counters,'student_l');sup=supervised(z,l['label']);con=z.sum()*0;stats=[];accepted=0;upix=0;entropy=0.;sce=0.;kl=0.
    if spec['ssl'] and epoch>20:
        if u is None:raise ValueError('missing U')
        validate_batch(u,False)
        zs,_=fwd(student,noisy(u['image'],key),False,key,counters,'student_u')
        with torch.no_grad():zt,_=fwd(teacher,u['image'],False,key,counters,'ema_u')
        p,q,j,t,values=objectives(zs,zt.softmax(1).detach(),u['geometry']);mask=j if spec['mask']=='J' else t
        con=reduce_loss(values[spec['loss']],mask);accepted=int(mask.sum());upix=int(u['geometry'].sum())
        stats=class_stats(p,q,mask,values,weight(epoch),spec['loss']);entropy=float(reduce_loss(values['entropy'],mask));sce=float(reduce_loss(values['SCE'],mask).detach());kl=sce-entropy
    elif u is not None:raise PermissionError('SUP/warmup cannot access U')
    loss=sup+weight(epoch)*con
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();counters['backward']+=1
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):raise FloatingPointError('nonfinite gradient')
    if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('EMA gradient')
    opt.step();counters['optimizer_steps']+=1
    with torch.no_grad():student.decoder.conv_logit.grad_update.zero_()
    update_ema(student,teacher)
    with torch.no_grad():teacher.decoder.conv_logit.sigma.weight.copy_(student.decoder.conv_logit.sigma.weight)
    counters['ema_updates']+=1
    return dict(supervised=float(sup.detach()),unsupervised=float(con.detach()),loss=float(loss.detach()),lambda_cons=weight(epoch),accepted=accepted,u_pixels=upix,SCE=sce,target_entropy=entropy,KL=kl,by_teacher_class=stats,sigma_has_gradient=student.decoder.conv_logit.sigma.weight.grad is not None)
