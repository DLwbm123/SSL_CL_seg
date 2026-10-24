"""Public pretrained encoder, shared production Trainer, no historical images."""
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34
from lcrseg.models.unet import UpBlock, ConvNormAct
from lcrseg.contracts import MethodStepOutput, LabeledBatch, UnlabeledBatch
from lcrseg.engine.trainer import Trainer
import campaign as c

LegacyNetwork = c.Network


class Network(nn.Module):
    def __init__(self, dataset):
        super().__init__(); self.dataset=dataset
        self.encoder=resnet34(weights=None)
        weights=torch.load(c.CFG['pretrained_weights'],map_location='cpu',weights_only=True)
        self.encoder.load_state_dict(weights,strict=True)
        self.encoder.fc=nn.Identity();self.encoder.avgpool=nn.Identity()
        self.up3=UpBlock(512,256,256);self.up2=UpBlock(256,128,128)
        self.up1=UpBlock(128,64,64);self.up0=UpBlock(64,64,32)
        self.full=nn.Sequential(nn.ConvTranspose2d(32,16,2,2),ConvNormAct(16,16))
        self.head=nn.Conv2d(16,4 if dataset=='mnms' else 3,1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('sd',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self

    def forward(self,x):
        c.COUNTS['native']+=len(x)
        z=(x+1)*.5 if self.dataset=='mnms' else x
        if z.shape[1]==1:z=z.expand(-1,3,-1,-1)
        z=(z-self.mean)/self.sd;e=self.encoder
        x0=e.relu(e.bn1(e.conv1(z)));x1=e.layer1(e.maxpool(x0));x2=e.layer2(x1);x3=e.layer3(x2);x4=e.layer4(x3)
        h=self.up0(self.up1(self.up2(self.up3(x4,x3),x2),x1),x0)
        out=self.head(self.full(h));assert out.shape[-2:]==x.shape[-2:]
        return out


def legacy(dataset):
    model=LegacyNetwork(dataset).cuda()
    path=Path(c.CFG['source_reference'])/f'{dataset}_0_SOURCE'/'final.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['step']==0
    model.load_state_dict(payload['teacher'],strict=True)
    return model.eval().requires_grad_(False)


BaseMethod=c.Method
class Method(BaseMethod):
    def __init__(self,model,dataset,arm,old=None,importance=None,bank=None):
        super().__init__(model,dataset,arm,old,{},[])
        if arm=='C':self.model.encoder.requires_grad_(False)
        self.reset_restore_tracking()

    def reset_restore_tracking(self):
        # Strict state restoration itself increments Tensor versions without a weight update.
        self.encoder_versions=[p._version for p in self.model.encoder.parameters()]
        self.encoder_buffer_versions=[b._version for b in self.model.encoder.buffers()]

    def assert_old_state_unchanged(self):
        super().assert_old_state_unchanged()
        assert self.encoder_buffer_versions==[b._version for b in self.model.encoder.buffers()]
        if self.arm=='C':
            assert self.encoder_versions==[p._version for p in self.model.encoder.parameters()]
            assert all(p.grad is None for p in self.model.encoder.parameters())

    def training_step(self,labeled,unlabeled,global_step,site_step):
        x,y,u=labeled.image,labeled.label,unlabeled.weak_image
        self.teacher.eval()
        z=self.model(x);zs=self.model(c.strong(x,self.dataset,[]))
        sup=.5*(c.supervised(z,y,self.dataset)+c.supervised(zs,y,self.dataset))
        with torch.no_grad():q=self.teacher(u)
        zu=self.model(c.strong(u,self.dataset,[]));mask=q.softmax(1).max(1).values>=.9
        con=c.kl(zu,q,mask);keep=z.sum()*0
        if self.old_model is not None and self.arm in ('SOURCE','C'):
            with torch.no_grad():
                ql=self.old_model(x);qu=self.old_model(u)
                pl=ql.softmax(1);pu=qu.softmax(1)
                ml=(pl.max(1).values>=.7)&(pl.argmax(1)==y);mu=pu.max(1).values>=.7
            weak_u=self.model(u);keep=.5*(c.kl(z,ql,ml)+c.kl(weak_u,qu,mu))
        strength=.1 if self.arm=='SOURCE' else .5 if self.arm=='C' else 0.
        total=sup+min((site_step+1)/1000,1.)*con+strength*keep
        return MethodStepOutput(total,dict(supervised=sup,consistency=con,history=keep),dict(accepted=float(mask.float().mean())))

    @torch.no_grad()
    def after_optimizer_step(self):
        current=dict(self.model.named_parameters())
        for name,q in self.teacher.named_parameters():
            p=current[name]
            if not p.requires_grad:q.copy_(p)
            else:q.mul_(.999).add_(p,alpha=.001)
        for q,p in zip(self.teacher.buffers(),self.model.buffers(),strict=True):q.copy_(p)


def trainer(method,lr,steps,bucket):
    enc=[p for n,p in method.model.named_parameters() if n.startswith('encoder.') and p.requires_grad]
    dec=[p for n,p in method.model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    groups=[dict(params=dec,lr=lr,name='decoder')]
    if enc:groups.append(dict(params=enc,lr=lr*.02,name='encoder'))
    opt=torch.optim.Adam(groups,weight_decay=1e-5)
    def pre(*args):
        if c.COUNTS['optimizer']>=c.CFG['optimizer_cap']:raise RuntimeError('Frozen optimizer cap')
    def post(*args):
        c.COUNTS['optimizer']+=1
        with (Path(c.CFG['root'])/'physical_optimizer.jsonl').open('a') as f:f.write(json.dumps(dict(call=c.COUNTS['optimizer'],bucket=bucket))+'\n')
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:.1+.9*.5*(1+math.cos(math.pi*min(k,steps)/steps)))
    return Trainer(method,optimizer=opt,scheduler=sch,device='cuda',amp=False,grad_clip_norm=5.)


@torch.no_grad()
def diagnostic(method,l,u,out,step):
    before=c.capture_rng_state();was=method.model.training
    try:
        torch.manual_seed(20272499);torch.cuda.manual_seed_all(20272499)
        lb,ub=c.batches(l,u,770001,method.dataset);method.model.eval()
        result=method.training_step(lb,ub,0,step-1)
        row=dict(step=step,total=float(result.total_loss),components={k:float(v) for k,v in result.losses.items()})
        assert math.isfinite(row['total'])
        path=out/'TRAINING_SUFFICIENCY_DESCRIPTIVE.json'
        rows=json.loads(path.read_text()) if path.exists() else [];rows.append(row);c.write(path,rows)
    finally:c.restore_rng_state(before);method.model.train(was)


def sufficient(out):
    rows=json.loads((out/'TRAINING_SUFFICIENCY_DESCRIPTIVE.json').read_text())
    a,b=rows[-2:];delta=abs(b['total']-a['total'])/max(1.,abs(a['total']))
    c.write(out/'TRAINING_SUFFICIENCY_GATE.json',dict(status='PASS' if delta<=.01 else 'SOURCE_STOP',normalized_change=delta,threshold=.01,not_global_convergence=True))
    return delta<=.01


def selfcheck():
    initial=c.COUNTS['native'];specs={}
    c.COUNTS['synthetic_reverse']=0
    for dataset in ('mnms','fundus'):
        net=Network(dataset).cuda();specs[dataset]=sum(p.numel() for p in net.parameters())
        ch=1 if dataset=='mnms' else 3;nc=4 if dataset=='mnms' else 3
        x=torch.rand(2,ch,64,64,device='cuda');y=torch.randint(nc,(2,64,64),device='cuda')
        old=copy.deepcopy(net).eval().requires_grad_(False)
        with torch.no_grad():old.head.bias[0].add_(3.)
        for arm in ('SOURCE','C'):
            model=copy.deepcopy(net);method=Method(model,dataset,arm,old)
            lb=LabeledBatch(x,y,torch.ones_like(y[:,None],dtype=torch.bool),[],[],[],[])
            ub=UnlabeledBatch(x,x,torch.ones_like(x[:,:1],dtype=torch.bool),[],[],[],[],[])
            result=method.training_step(lb,ub,0,1000)
            c.COUNTS['synthetic_reverse']+=1
            result.total_loss.backward()
            assert torch.isfinite(result.total_loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in method.model.parameters() if p.requires_grad)
            assert float(result.losses['history'].detach())>0
            method.after_optimizer_step()
            if arm=='C':
                assert all(torch.equal(v,method.teacher.encoder.state_dict()[k]) for k,v in method.model.encoder.state_dict().items())
            method.assert_old_state_unchanged()
    c.write(Path(c.CFG['root'])/'SYNTHETIC.json',dict(status='PASS',native=c.COUNTS['native']-initial,reverse=4,optimizer=0,parameter_counts=specs,encoder_frozen_and_BN_unchanged=True,complete_source_target_backward=True,public_pretrain_not_medical_data=True))


def main(cfg):
    c.Network=Network;c.warm=lambda dataset:Network(dataset).cuda();c.Method=Method;c.trainer=trainer
    c.source_legacy=legacy;c.summarize_current=lambda *args:({}, {})
    c.record_training_diagnostic=diagnostic;c.training_sufficient=sufficient;c.selfcheck=selfcheck
    c.main(cfg)
