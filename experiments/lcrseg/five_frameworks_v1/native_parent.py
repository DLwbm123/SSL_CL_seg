"""User-designated LR_SRC_A reference; explicitly not original KI recovery."""
import copy
import subprocess
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from experiments.lcrseg.di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, _official_probabilistic_classifier
from experiments.lcrseg.ssl_head_control_v0_1.core import set_head, LINEAR
from experiments.lcrseg.ssl_anchored_mix_v0_1.core import supervised_parts
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as lr
from experiments.lcrseg.ssl_foundation_v0_1.core import rng
from experiments.lcrseg.single_teacher_scd_v0_1.engine import precision, UPSTREAM, UPSTREAM_PATH
from .numerics import finite

IDENTITY='NATIVE_LR_SRC_A_3DOMAIN_V1'


def build(reference,device,seed):
    if subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM:
        raise ValueError('wrong native upstream commit')
    subprocess.run(['git','-C',str(reference),'diff','--quiet','HEAD'],check=True)
    with rng(device,'native_upstream_import'):_official_probabilistic_classifier(reference,upstream_path=UPSTREAM_PATH)
    with rng(device,IDENTITY,seed,'REFUGE','initialization'):
        model=build_lcrseg_unet_jascl_model(reference,upstream_path=UPSTREAM_PATH,input_channels=3,num_classes=3)
    precision();return set_head(model,LINEAR).to(device)


class Adapter:
    """Descriptor over actual projected Conv2d parameters, not a toy factorization."""
    def __init__(self,parent,index):self.parent,self.index=parent,index
    @property
    def layer(self):return self.parent.native.get_submodule(lr.LAYERS[self.index][:-7])
    @property
    def a(self):return self.layer.A
    @property
    def b(self):return self.layer.B
    @property
    def base(self):return self.layer.weight.flatten(1)
    @property
    def new(self):return True
    @new.setter
    def new(self,value):
        if not value:raise ValueError('sealed native adapters must be merged, not reused')
    @property
    def free_projector(self):
        v=getattr(self.parent,'V64_'+str(self.index))
        return torch.eye(v.shape[0],dtype=v.dtype,device=v.device)-v@v.T
    def effective_weight(self):return self.layer.effective().flatten(1)


class NativeLRParent(nn.Module):
    synthetic=False
    identity=IDENTITY
    def __init__(self,native,seed,stage_source,adapt=True):
        super().__init__();self.native=native;self.d=16
        self.seed=seed;self.stage_source=copy.deepcopy(stage_source)
        if adapt:
            tensors={}
            for i,name in enumerate(lr.LAYERS):
                t,_=lr.basis(native.get_parameter(name),seed,stage_source.get('domain',stage_source),name)
                tensors[name]=t;self.register_buffer('V64_'+str(i),t['V'].to(native.get_parameter(name).device))
            lr.configure(native,'LR_SRC_A',lambda name:tensors[name])
        self.configure_stage_training() if adapt else self.eval().requires_grad_(False)

    @property
    def adapters(self):
        return [Adapter(self,i) for i,n in enumerate(lr.LAYERS) if isinstance(self.native.get_submodule(n[:-7]),lr.LowRankConv)]

    def parameter_groups(self):
        aa=[a.a for a in self.adapters];bb=[a.b for a in self.adapters];ids={id(p) for p in aa+bb}
        return dict(input_factors=aa,output_factors=bb,other_allowed=[],frozen=[p for p in self.parameters() if id(p) not in ids])

    def configure_stage_training(self):
        self.train();self.requires_grad_(False)
        for a in self.adapters:a.a.requires_grad_(True);a.b.requires_grad_(True)
        self.native.decoder.conv_logit.eval()

    def features(self,x,mode='student',overrides=None):
        # Preserve native encoder/decoder operations while bypassing the readout.
        m=self.native
        @contextmanager
        def override():
            handles=[]
            try:
                for i,w in (overrides or {}).items():
                    layer=self.adapters[i].layer
                    def hook(mod,args,result,weight=w):
                        return F.conv2d(args[0],weight.reshape_as(mod.weight),mod.bias,mod.stride,mod.padding,mod.dilation,mod.groups)
                    handles.append(layer.register_forward_hook(hook))
                yield
            finally:
                for handle in handles:handle.remove()
        with override():
            e1=m.enc1(x);e2=m.enc2(m.pool(e1));e3=m.enc3(m.pool(e2));b=m.bottleneck(m.pool(e3))
            return m.decoder.dec1(m.decoder.dec2(m.decoder.dec3(b,e3),e2),e1)

    def native_readout(self,h,output_shape,mode='student'):
        return F.interpolate(self.native.decoder.conv_logit(h,stochastic=False),size=output_shape,mode='bilinear',align_corners=True)
    def effective_readout_kernel_at_entry(self):return self.native.decoder.conv_logit.mu.weight.detach()
    def supervised(self,logp,y):return sum(supervised_parts(logp,y))
    def constraint_loss(self):return self.adapters[0].a.sum()*0
    def stage_entry(self):pass  # Constructor installs exactly one new stage's adapters.
    def stage_exit(self):
        lr.merge(self.native)
        for key in list(self._buffers):
            if key.startswith('V64_'):delattr(self,key)
    def apply_constraints(self):
        # The constraint is inside effective(), never an extra orthogonal loss.
        for a in self.adapters:
            finite(a.layer.delta(),'native projected increment')
            v=getattr(self,'V64_'+str(a.index));d=a.layer.delta().double()
            if float((d@v).norm())>1e-5*(float(d.norm())+1e-12):
                raise FloatingPointError('native parameterized right projection residual')

    def output_to_feature(self,x,shape,categorical=False):
        bchw=x.ndim==4;z=x if bchw else x[:,None]
        inner=(shape[0]-2,shape[1]-2)
        z=F.interpolate(z.float(),size=inner,mode='nearest' if categorical else 'bilinear',**({} if categorical else {'align_corners':True}))
        if categorical:z=F.pad(z,(1,1,1,1),value=0 if x.dtype==torch.bool else 255)
        else:z=F.pad(z,(1,1,1,1),mode='replicate')
        z=z if bchw else z[:,0]
        return z.to(x)
    def feature_to_output(self,x,shape):
        bchw=x.ndim==4;z=x if bchw else x[:,None]
        z=F.interpolate(z[...,1:-1,1:-1],size=shape,mode='bilinear',align_corners=True)
        return z if bchw else z[:,0]
    def optimizer_groups(self,options):
        g=self.parameter_groups();base=options['lr']*options['parent_lr_multiplier']
        return [{'params':g['input_factors'],'lr':base,'name':'A'},
                {'params':g['output_factors'],'lr':base*options['lr_B_over_A'],'name':'B'}]
    def semantic_metadata(self):
        return {'identity':IDENTITY,'synthetic':False,'source':copy.deepcopy(self.stage_source),'seed':self.seed,
                'rank':8,'layers':list(lr.LAYERS),'effective':'W0+B@(A-(A@V)@V.T)',
                'geometry':'3x3_valid_centers_then_bilinear_align_corners_true','head_stochastic':False,
                'teacher':'dense_effective_weight_EMA','precision':'FP32_no_autocast'}

    @torch.no_grad()
    def update_dense_ema(self,teacher,validate_only=False):
        for name,q in teacher.native.state_dict().items():
            module,_,key=name.rpartition('.');sm=self.native.get_submodule(module)
            p=sm.effective() if key=='weight' and isinstance(sm,lr.LowRankConv) else getattr(sm,key)
            candidate=p if key=='grad_update' or not q.is_floating_point() else q.detach().clone().mul_(.99).add_(p,alpha=.01)
            finite(candidate,'native dense EMA '+name)
            if not validate_only:
                if key=='grad_update' or not q.is_floating_point():q.copy_(p)
                else:q.mul_(.99).add_(p,alpha=.01)
