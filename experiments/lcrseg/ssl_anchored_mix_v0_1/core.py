"""Real-label CE+Dice and complementary source collection on the unchanged model."""
from experiments.lcrseg.ssl_target_path_v0_1.core import *
from experiments.lcrseg.ssl_target_path_v0_1 import core as predecessor
ARMS=('SUP_CED','MT_CED','MIX_CTX','MIX_CED')
SEEDS=(31,32,33,41,42)
def config(arm):
    if arm not in (*ARMS,'SUP_CE'):raise ValueError('unregistered arm')
    return dict(ssl=arm in ('MT_CED','MIX_CTX','MIX_CED'),mix=arm.startswith('MIX'),pseudo=arm in ('MT_CED','MIX_CED'),dice=arm!='SUP_CE')
def head_mode(arm):config(arm);return LINEAR
def build(reference,device,seed,domain,arm):config(arm);return predecessor.build(reference,device,seed,domain,'SUP')

def supervised_parts(logp,y):
    if logp.ndim!=4 or logp.shape[1]!=3 or y.shape!=logp.shape[:1]+logp.shape[2:] or y.dtype!=torch.long:raise ValueError('B3HW log probabilities and BHW int64 labels')
    if logp.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logp).all() or not (((y>=0)&(y<3))|(y==255)).all():raise ValueError('invalid log probabilities or labels')
    v=y!=255;pixels=-logp.gather(1,y.masked_fill(~v,0)[:,None]).squeeze(1);ce=(pixels*v).sum()/v.sum().clamp_min(1)
    # Spatial sums in FP64; equal image and foreground-class weight, including empty classes.
    p=logp.exp().double();valid=v[:,None];target=torch.stack([y==1,y==2],1)
    pp=p[:,1:]*valid;tt=target.double()*valid
    dice=(2*(pp*tt).sum((2,3))+1e-5)/(pp.sum((2,3))+tt.sum((2,3))+1e-5)
    loss=((1-dice.mean(1))*v.flatten(1).any(1)).mean()
    return ce,loss

def rectangles(batch_size,height,width,key,device):
    if height%3 or width%3:raise ValueError('mix geometry must admit exact 2/3 sides')
    ph,pw=2*height//3,2*width//3;m=torch.zeros(batch_size,height,width,dtype=torch.bool,device=device);coords=[]
    for pos in range(batch_size):
        g=torch.Generator().manual_seed(stable_seed(*key,'mix_rectangle',pos))
        top=int(torch.randint(height-ph+1,(),generator=g));left=int(torch.randint(width-pw+1,(),generator=g))
        m[pos,top:top+ph,left:left+pw]=True;coords.append([top,left,ph,pw])
    return m,coords

def pairing(nl,nu,device):
    if not (1<=nu<=nl<=2):raise ValueError('fixed batches need 1<=U<=L<=2')
    ix=torch.arange(nl,device=device)%nu;counts=torch.bincount(ix,minlength=nu)
    return ix,counts[ix].double().reciprocal()

def collect_sources(loga,logb,m):
    if loga.shape!=logb.shape or loga.ndim!=4 or m.shape!=loga.shape[:1]+loga.shape[2:] or m.dtype!=torch.bool:raise ValueError('source collection shape')
    m=m.detach()[:,None]
    return torch.where(m,loga,logb),torch.where(m,logb,loga)

def soft_target(logp,q,geometry,index=None,repeat_weight=None):
    # q is unique-U; contexts are loss-weighted, never averaged in logit space.
    if index is None:index=torch.arange(len(q),device=q.device);repeat_weight=q.new_ones(len(q))
    validate_pair(logp,q[index],geometry[index]);q=q.detach();m=geometry&(q.max(1)[0]>.7)
    values=-(q[index]*logp).sum(1);weighted=m[index]*repeat_weight[:,None,None]
    den=m.sum().clamp_min(1);loss=(values.double()*weighted).sum()/den
    ent=(-torch.xlogy(q,q).sum(1)*m).double().sum()/den
    return loss,ent,m

def train_step(student,teacher,opt,l,u,proto,support,arm,seed,domain,epoch,step,counters,diagnostic=False):
    spec=config(arm);validate_batch(l,True)
    if proto is not None or support is not None:raise PermissionError('no prototypes')
    active=spec['ssl'] and epoch>20;key=(seed,domain,epoch,step)
    if active:
        if u is None:raise ValueError('missing U')
        validate_batch(u,False)
    elif u is not None:raise PermissionError('SUP/warmup U access')
    opt.zero_grad(set_to_none=True);coords=[];ix=None;rw=None;u_log=None
    if active and spec['mix']:
        ix,rw=pairing(len(l['image']),len(u['image']),l['image'].device)
        m,coords=rectangles(len(l['image']),*l['image'].shape[-2:],key,l['image'].device)
        ux=noisy(u['image'],key)[ix]
        xa=torch.where(m[:,None],l['image'],ux);xb=torch.where(m[:,None],ux,l['image'])
        za,_=fwd(student,xa,False,key,counters,'student_mix_A');zb,_=fwd(student,xb,False,key,counters,'student_mix_B')
        logl,u_log=collect_sources(za.log_softmax(1),zb.log_softmax(1),m)
        del xa,xb,ux
    else:
        z,_=fwd(student,l['image'],False,key,counters,'student_l');logl=z.log_softmax(1)
        if active:
            zs,_=fwd(student,noisy(u['image'],key),False,key,counters,'student_u');u_log=zs.log_softmax(1)
    ce,dice=supervised_parts(logl,l['label']);sup=ce+(dice if spec['dice'] else 0)
    # Preserve exact original CE objective/precision for P2 SUP_CE compatibility.
    if arm=='SUP_CE':sup=predecessor.supervised(z,l['label']);ce=sup
    con=logl.sum()*0;entropy=0.;accepted=None;by_u=[]
    if active and spec['pseudo']:
        with torch.no_grad():zt,_=fwd(teacher,u['image'],False,key,counters,'ema_u')
        q=zt.softmax(1).detach();con,ent,mask=soft_target(u_log,q,u['geometry'],ix,rw);entropy=float(ent);accepted=int(mask.sum())
        with torch.no_grad():
            for c in range(3):
                predicted=(q.argmax(1)==c)&u['geometry'];by_u.append(dict(class_id=c,predicted=int(predicted.sum()),accepted=int((predicted&mask).sum())))
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
    return dict(supervised=float(sup.detach()),CE=float(ce.detach()),GT_Dice=float(dice.detach()),dice_weight=1 if spec['dice'] else 0,
        unsupervised=float(con.detach()),SCE=float(con.detach()),target_entropy=entropy,KL=float(con.detach())-entropy,
        loss=float(loss.detach()),lambda_cons=weight(epoch),all_ignore_L=int((l['label']==255).flatten(1).all(1).sum()),
        labeled_class_exposure=[int((l['label']==c).sum()) for c in range(3)],labeled_source_pixels=l['label'].numel(),labeled_source_scoring_multiplicity=1,
        labeled_image_records=len(l['image']),unlabeled_unique_batch_sources=len(u['image']) if active else 0,
        unlabeled_context_records=len(l['image']) if coords else (len(u['image']) if active else 0),
        U_geometry_pixels=int(u['geometry'].sum()) if active else 0,U_accepted=accepted,U_class=by_u,
        mix_rectangles=coords,U_source_index=[] if ix is None else ix.tolist(),U_loss_repeat_weights=[] if rw is None else rw.tolist(),
        sigma_has_gradient=student.decoder.conv_logit.sigma.weight.grad is not None)
