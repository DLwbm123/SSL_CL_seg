"""Explicit effective-weight constraints on the frozen fourteen convolutions."""
import contextlib,hashlib,math
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from experiments.lcrseg.ams_seq_transfer_v0_1 import core as parent
from experiments.lcrseg.ams_seq_transfer_v0_1.core import *
from experiments.lcrseg.ssl_anchored_mix_v0_1 import telemetry

ARMS=('F_FULL','F_CONV','LR_FREE','LR_RAND','LR_SRC_A','LR_SRC_AB')
LAYERS=tuple(x+'.weight' for x in ('enc1.block.0','enc1.block.3','enc2.block.0','enc2.block.3','enc3.block.0','enc3.block.3','bottleneck.block.0','bottleneck.block.3','decoder.dec3.merge.block.0','decoder.dec3.merge.block.3','decoder.dec2.merge.block.0','decoder.dec2.merge.block.3','decoder.dec1.merge.block.0','decoder.dec1.merge.block.3'))

def hash_state(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def basis(weight,seed,source_domain,name):
    w=weight.detach().cpu().double().flatten(1);m,d=w.shape;r=8;k=min(m//2,d-r)
    if m<16 or d<27 or not torch.isfinite(w).all() or w.norm()==0:raise ValueError('source geometry/numerical anomaly')
    u,s,vh=torch.linalg.svd(w,full_matrices=False)
    rank=int((s>s[0]*max(m,d)*torch.finfo(torch.float64).eps).sum())
    if rank<m:raise ValueError('numerical source rank anomaly')
    g=torch.Generator(device='cpu').manual_seed(stable_seed('LCTX_WEIGHT_MEMORY_V0_1',seed,source_domain,name,'adapter_G'))
    G=torch.randn(r,d,dtype=torch.float64,generator=g)/math.sqrt(d)
    qg=torch.Generator(device='cpu').manual_seed(stable_seed('LCTX_WEIGHT_MEMORY_V0_1',seed,source_domain,name,'random_protection'))
    q,_=torch.linalg.qr(torch.randn(d,k,dtype=torch.float64,generator=qg),mode='reduced')
    tensors=dict(W0=weight.detach().cpu().clone(),V=vh[:k].T.contiguous(),U=u[:,:k].contiguous(),Q=q,G=G)
    meta=dict(layer=name,shape=list(weight.shape),rank=r,k=k,numerical_rank=rank,selected_energy=float(s[:k].square().sum()/s.square().sum()),tensor_hashes={key:hash_state({key:v}) for key,v in tensors.items()},svd_visible_workspace_bytes=tensor_bytes([w,u,s,vh]),svd_backend_workspace='not directly observable',basis_storage_bytes=tensor_bytes([tensors['V'],tensors['U'],q]),raw_AB_parameters=r*(m+d),effective_rank_manifold_FREE=r*(m+d-r),effective_rank_manifold_A=r*(m+d-k-r),effective_rank_manifold_AB=r*(m+d-2*k-r))
    return tensors,meta

def initial_A(tensors,arm):
    G=tensors['G'];P=tensors['Q'] if arm=='LR_RAND' else tensors['V']
    if arm!='LR_FREE':
        gp=G-(G@P)@P.T
        if gp.norm()<=torch.finfo(torch.float64).eps:raise ValueError('zero projected initialization')
        G=gp*(G.norm()/gp.norm())
    return G.to(tensors['W0'])

class LowRankConv(nn.Module):
    def __init__(self,conv,tensors,arm):
        super().__init__();self.weight=conv.weight;self.weight.requires_grad_(False);self.bias=conv.bias
        self.stride,self.padding,self.dilation,self.groups=conv.stride,conv.padding,conv.dilation,conv.groups
        self.arm=arm;w=conv.weight;m,d=w.flatten(1).shape
        P=tensors['Q'] if arm=='LR_RAND' else tensors['V']
        self.A=nn.Parameter(initial_A(tensors,arm).to(w).clone());self.B=nn.Parameter(w.new_zeros(m,8))
        self.register_buffer('right',None if arm=='LR_FREE' else P.to(w).clone())
        self.register_buffer('left',tensors['U'].to(w).clone() if arm=='LR_SRC_AB' else None)
    def delta(self):
        a=self.A if self.right is None else self.A-(self.A@self.right)@self.right.T
        b=self.B if self.left is None else self.B-self.left@(self.left.T@self.B)
        return b@a
    def effective(self):return self.weight+self.delta().reshape_as(self.weight)
    def forward(self,x):return F.conv2d(x,self.effective(),self.bias,self.stride,self.padding,self.dilation,self.groups)

def configure(model,arm,load_basis):
    if arm not in ARMS:raise PermissionError('unknown arm')
    actual=tuple(n+'.weight' for n,m in model.named_modules() if type(m) is nn.Conv2d and n+'.weight' in LAYERS)
    if actual!=LAYERS:raise ValueError('fourteen-layer architecture mismatch')
    if arm=='F_FULL':return model
    model.requires_grad_(False)
    for name in LAYERS:
        module=name[:-7];conv=model.get_submodule(module)
        if conv.bias is not None or conv.groups!=1:raise ValueError('conv geometry mismatch')
        if arm=='F_CONV':conv.weight.requires_grad_(True)
        else:
            par,_,child=module.rpartition('.');model.get_submodule(par)._modules[child]=LowRankConv(conv,load_basis(name),arm)
    return model

def dense_items(model):
    for name,value in model.state_dict().items():
        parts=name.rsplit('.',1)
        if len(parts)==2 and isinstance(model.get_submodule(parts[0]),LowRankConv):
            if parts[1]!='weight':continue
            value=model.get_submodule(parts[0]).effective().detach()
        yield name,value

@torch.no_grad()
def dense_ema(student,teacher):
    for name,q in teacher.state_dict().items():
        module,_,key=name.rpartition('.');sm=student.get_submodule(module)
        p=sm.effective() if key=='weight' and isinstance(sm,LowRankConv) else getattr(sm,key)
        if key=='grad_update' or not q.is_floating_point():q.copy_(p)
        else:q.mul_(.99).add_(p,alpha=.01)
    if telemetry.ACTIVE is not None:
        telemetry.ACTIVE.counts['ema_updates_attempts']+=1;telemetry.ACTIVE.counts['ema_updates']+=1

def step(student,teacher,opt,l,seed,domain,epoch,index,counts,patients,arm):
    context=patch.object(parent.mathcore,'update_ema',dense_ema) if arm.startswith('LR_') else contextlib.nullcontext()
    with context:return parent.step(student,teacher,opt,l,None,'T_LCTX',seed,domain,epoch,index,counts,patients)

@contextlib.contextmanager
def training_access(domain):
    original=CurrentData.__init__;stage=DOMAINS.index(domain)+1
    def init(self,data,actual_stage,role,**kw):
        if actual_stage!=stage or kw.get('domain',stage)!=stage or role!='train_labeled' or kw.get('purpose','train')!='train':raise PermissionError('only current L accessor allowed; no U/source/evaluation')
        return original(self,data,actual_stage,role,**kw)
    with patch.object(CurrentData,'__init__',init):yield

@torch.no_grad()
def merge(model):
    for name in LAYERS:
        path=name[:-7];old=model.get_submodule(path)
        if not isinstance(old,LowRankConv):continue
        w=old.effective();m,cin,kh,kw=w.shape
        with torch.device('meta'):new=nn.Conv2d(cin,m,(kh,kw),stride=old.stride,padding=old.padding,dilation=old.dilation,groups=old.groups,bias=False)
        new.weight=nn.Parameter(w,requires_grad=False);par,_,child=path.rpartition('.');model.get_submodule(par)._modules[child]=new
    return model

@torch.no_grad()
def diagnostics(model,load_basis,epoch,arm):
    rows=[]
    for name in LAYERS:
        t=load_basis(name);layer=model.get_submodule(name[:-7]);w=layer.effective() if isinstance(layer,LowRankConv) else layer.weight
        delta=(w.detach().cpu().double()-t['W0'].double()).flatten(1);den=float(delta.norm())+1e-12
        right=t['Q'] if arm=='LR_RAND' else t['V']
        row=dict(epoch=epoch,layer=name,arm=arm,relative_update_norm=float(delta.norm()/t['W0'].double().norm()),source_right_leak=float((delta@t['V']).norm())/den,right_leak=float((delta@right).norm())/den,left_leak=float((t['U'].T@delta).norm())/den,nonzero_update=bool(delta.norm()>0))
        # Measure parameterization delta separately from rounded addition to W0.
        if isinstance(layer,LowRankConv):
            dw=layer.delta().double();den2=float(dw.norm())+1e-12
            row['parameterized_right_leak']=float((dw@right.to(dw)).norm())/den2
            row['parameterized_left_leak']=float((t['U'].T.to(dw)@dw).norm())/den2
            if arm!='LR_FREE' and row['nonzero_update']:
                if row['parameterized_right_leak']>1e-5 or row['right_leak']>1e-5:raise FloatingPointError('right constraint residual exceeds 1e-5')
                if arm=='LR_SRC_AB' and (row['parameterized_left_leak']>1e-5 or row['left_leak']>1e-5):raise FloatingPointError('left constraint residual exceeds 1e-5')
        rows.append(row)
    return rows
