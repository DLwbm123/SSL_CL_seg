"""Dense RCF transfer with full-resolution targets and canonical pixel identity."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.contracts import MethodStepOutput
from lcrseg.methods.base import ContinualSegMethod
from .reliability import fit_gmm, posterior, threshold


def canonical(dataset):
    result=copy.copy(dataset);result.transform=None
    return result


def key(sample):
    return sample['case_id'],sample['slice_index']


def pixel_indices(n, limit=2048):
    return np.linspace(0,n-1,min(n,limit),dtype=np.int64)


@torch.no_grad()
def class_prototypes(model, labeled):
    device=next(model.parameters()).device
    sums=torch.zeros(4,768,device=device);counts=torch.zeros(4,device=device)
    for sample in canonical(labeled):
        image=sample['image'][None].to(device)
        feature=[]
        for dim in (None,-1,-2):
            f=model.frozen_features(image if dim is None else image.flip(dim))
            feature.append(f if dim is None else f.flip(dim))
        f=F.normalize(torch.stack(feature).mean(0),dim=1)[0]
        # Pixel labels supervise token centers; this is explicit dense adaptation.
        y=F.interpolate(sample['label'][None,None].float().to(device),(14,14),mode='nearest')[0,0].long()
        for c in range(4):
            mask=y==c
            if mask.any():sums[c]+=f[:,mask].sum(1);counts[c]+=mask.sum()
    if (counts==0).any():raise ValueError('visible labels do not support every semantic prototype')
    return F.normalize(sums/counts[:,None],dim=1).cpu()


@torch.no_grad()
def evidence(model,image,prototypes):
    p=(model(image).logits/2).softmax(1)
    p2=(model(image.flip(-1)).logits.flip(-1)/2).softmax(1)
    mean=(p+p2)/2
    confidence,label=mean.max(1)
    js=.5*((p*(p.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1)+
            (p2*(p2.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1))/math.log(2)
    stable=(confidence*(1-js)).clamp(1e-6,1-1e-6)
    pred=torch.logit(stable)
    f=model.frozen_features(image)
    f2=model.frozen_features(image.flip(-1)).flip(-1)
    proto=prototypes.to(image.device)
    similarities=(torch.einsum('cd,bdhw->bchw',proto,f)+torch.einsum('cd,bdhw->bchw',proto,f2))/2
    similarities=F.interpolate(similarities,image.shape[-2:],mode='bilinear',align_corners=False)
    own=similarities.gather(1,label[:,None])[:,0]
    other=similarities.scatter(1,label[:,None],-torch.inf).max(1).values
    coords=torch.stack([pred,own-other],-1)
    return coords[0].cpu().numpy(),label[0].to(torch.uint8).cpu(),confidence[0].cpu()


class RCFMethod(ContinualSegMethod):
    method_name='rcf_transunet_transfer'
    def __init__(self,model,labeled,unlabeled,arm):
        super().__init__(model,config=dict(lambda_dice=0.))
        self.labeled,self.unlabeled=canonical(labeled),canonical(unlabeled)
        self.arm=arm
        self.binding=tuple((s.row['case_id'],s.slice_index) for s in self.unlabeled.samples)
        self.prototypes=None
        self.gmm=None;self.tau=None;self.history={};self.targets={};self.selected={}
        self.last_epoch=-1;self.accounting={}

    @torch.no_grad()
    def prepare(self,step):
        epoch=step//150
        if epoch==self.last_epoch:return
        self.last_epoch=epoch
        if epoch<2:return
        self.model.eval()
        if self.arm=='confidence':
            device=next(self.model.parameters()).device
            for sample in self.unlabeled:
                if 'label' in sample:raise RuntimeError('hidden label in U')
                image=sample['weak_image'][None].to(device)
                p=(self.model(image).logits/2).softmax(1)
                p2=(self.model(image.flip(-1)).logits.flip(-1)/2).softmax(1)
                confidence,y=((p+p2)/2).max(1)
                self.targets[key(sample)]=y[0].to(torch.uint8).cpu()
                self.selected[key(sample)]=(confidence[0]>=.95).cpu()
            total=sum(m.numel() for m in self.selected.values())
            self.accounting=dict(threshold=.95,selected_fraction=sum(int(m.sum()) for m in self.selected.values())/total)
            return
        if self.prototypes is None:self.prototypes=class_prototypes(self.model,self.labeled)
        rows=[];fit=[]
        device=next(self.model.parameters()).device
        for sample in self.unlabeled:
            if 'label' in sample:raise RuntimeError('hidden label in U')
            xy,y,_=evidence(self.model,sample['weak_image'][None].to(device),self.prototypes)
            flat=xy.reshape(-1,2)
            fit.append(flat[pixel_indices(len(flat))])
            rows.append((key(sample),xy,y))
        fitted=fit_gmm(np.concatenate(fit))
        reused=fitted is None and self.gmm is not None
        if fitted is not None:self.gmm=fitted
        if self.gmm is None:
            self.targets={};self.selected={};self.accounting=dict(gmm_valid=False);return
        scores=[]
        for k,xy,y in rows:
            p=posterior(xy.reshape(-1,2),self.gmm)[0][:,self.gmm['clean']].reshape(y.shape).astype('float32')
            if k in self.history:p=.7*self.history[k]+.3*p
            self.history[k]=p;self.targets[k]=y
            scores.append(p.reshape(-1))
        # Threshold uses all unique pixels, not repeated minibatch occurrences.
        self.tau=threshold(np.concatenate(scores),.90,self.tau)
        self.selected={k:torch.from_numpy(p>=self.tau) for k,p in self.history.items()}
        accepted=sum(int(m.sum()) for m in self.selected.values())
        total=sum(m.numel() for m in self.selected.values())
        self.accounting=dict(gmm_valid=True,gmm_reused=reused,threshold=self.tau,
                             selected_fraction=accepted/total,calibration_pixels=sum(len(f) for f in fit),
                             unique_pixels=total)

    def training_step(self,labeled_batch,unlabeled_batch,global_step,site_step):
        if self.last_epoch!=site_step//150:raise RuntimeError('prepare epoch before optimizer update')
        out=self.model(labeled_batch.image)
        losses=self._supervised_losses(out,labeled_batch)
        ssl=out.logits.sum()*0;coverage=0.
        if site_step>=300 and unlabeled_batch is not None:
            batch=unlabeled_batch
            if self.targets:
                ys=[];masks=[]
                for case,z,record in zip(batch.case_id,batch.slice_index,batch.geometry_record):
                    y=self.targets[(case,z)];mask=self.selected[(case,z)]
                    for field,dim in (('hflip',-1),('vflip',-2)):
                        if record.get(field):y=y.flip(dim);mask=mask.flip(dim)
                    ys.append(y);masks.append(mask)
                y=torch.stack(ys).to(out.logits.device).long();mask=torch.stack(masks).to(out.logits.device)
            else:y=None
            if y is not None:
                logits=self.model(batch.strong_image).logits
                mask=mask&batch.strong_valid_mask[:,0]
                per_pixel=F.cross_entropy(logits,y,reduction='none')
                ssl=(per_pixel*mask).sum()/mask.sum().clamp_min(1)
                coverage=float(mask.float().mean())
        losses['loss_assim']=ssl
        return MethodStepOutput(losses['loss_sup']+ssl,self._all_loss_keys(out.logits,losses),
                                dict(self.accounting,batch_selection=coverage))

    def method_state_dict(self):
        return copy.deepcopy(dict(binding=self.binding,arm=self.arm,prototypes=self.prototypes,gmm=self.gmm,
             tau=self.tau,history=self.history,targets=self.targets,selected=self.selected,
             last_epoch=self.last_epoch,accounting=self.accounting))

    def load_method_state_dict(self,state):
        if state['binding']!=self.binding or state['arm']!=self.arm:
            raise ValueError('resume population/arm changed')
        for k in ('prototypes','gmm','tau','history','targets','selected','last_epoch','accounting'):
            setattr(self,k,copy.deepcopy(state[k]))
