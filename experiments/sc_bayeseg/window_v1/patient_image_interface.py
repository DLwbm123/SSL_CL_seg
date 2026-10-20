"""Image-only frozen U-Net features -> H(c,R) -> frozen categorical G.

This conditional-generator variant omits F and is not the official BayeSeg model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from sc_bayeseg.window_v1.r_start import draw_g,local_sample


class PatientImageInterface(nn.Module):
    def __init__(self,backbone,generator):
        super().__init__()
        self.backbone=backbone.requires_grad_(False)
        self.generator=generator.requires_grad_(False)
        self.hc=nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,32))
        self.hr=nn.Conv2d(16,32,1)

    def head_parameters(self):
        return tuple(self.hc.parameters())+tuple(self.hr.parameters())

    def posterior(self,image):
        if image.ndim!=4 or image.shape[1]!=1:
            raise ValueError('image must be [B,1,H,W]')
        resized=F.interpolate(image,(128,128),mode='bilinear',align_corners=False)
        with torch.no_grad():feature=self.backbone(resized).decoder_features['dec1']
        b,c,h,w=feature.shape
        assert (c,h,w)==(16,128,128)
        cm,cs=self.hc(feature.mean((2,3))).chunk(2,1)
        rm,rs=self.hr(feature.reshape(b,16,16,8,16,8).mean((3,5))).chunk(2,1)
        return cm,-4+3*torch.tanh(cs/3),rm,-4+3*torch.tanh(rs/3)

    @torch.no_grad()
    def predict_probabilities(self,image,*,seed=620163,theta_draws=4,local_draws=4):
        if theta_draws<1 or local_draws<1:
            raise ValueError('sampling counts must be positive')
        native=image.shape[-2:];q=self.posterior(image)
        tg=torch.Generator(device=image.device).manual_seed(seed)
        lg=torch.Generator(device=image.device).manual_seed(seed+1000000)
        result=torch.zeros((len(image),4,128,128),device=image.device,dtype=image.dtype)
        try:
            for _ in range(theta_draws):
                draw_g(self.generator,tg)
                for _ in range(local_draws):
                    c,r=local_sample(q,lg)
                    result+=self.generator.spatial_logits(c,r).softmax(1)
            result/=theta_draws*local_draws
            result=F.interpolate(result,native,mode='bilinear',align_corners=False)
            return result/result.sum(1,keepdim=True)
        finally:self.generator.clear_draw()
