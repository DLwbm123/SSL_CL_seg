"""Existing U-Net trunk inside the unchanged conditional c/R posterior heads."""
from pathlib import Path
import torch
from lcrseg.models.unet import UNet2D
from ..recognition import Recognition
from ..stream_state import StreamState


class MultiscaleRecognition(Recognition):
    def __init__(self,config,projection):
        super().__init__(config,projection)
        assert config.hidden==16
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.backbone=UNet2D(config.channels+(2 if config.coordinate_channels else 0),config.classes,base_channels=16)
        del self.backbone.segmentation_head,self.backbone.projection_head
        del self.enc,self.down,self.up
        self.to(dtype=getattr(torch,config.dtype))

    def context(self,x,theta,a):
        image=x
        if self.config.coordinate_channels:
            yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[-2],device=x.device,dtype=x.dtype),torch.linspace(-1,1,x.shape[-1],device=x.device,dtype=x.dtype),indexing='ij')
            image=torch.cat((x,torch.stack((xx,yy))[None].expand(len(x),-1,-1,-1)),1)
        b=self.backbone;h1=b.enc1(image);h2=b.enc2(b.pool(h1));h3=b.enc3(b.pool(h2));h4=b.bottleneck(b.pool(h3))
        feature=torch.tanh(b.dec1(b.dec2(b.dec3(h4,h3),h2),h1))
        extra=torch.cat([self.projection(theta),a])[None,:,None,None].expand(len(x),-1,*x.shape[-2:])
        return torch.cat((feature,extra),1)


class MultiscaleState(StreamState):
    architecture='multiscale_recognition_v46'

    def __init__(self,config):
        super().__init__(config)
        # Recreate the original heads with exactly the same initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed);self.phi=MultiscaleRecognition(config,self.projection)

    def payload(self):
        return dict(super().payload(),architecture=self.architecture)

    @classmethod
    def load(cls,path):
        p=torch.load(Path(path),map_location='cpu',weights_only=True)
        assert p.get('architecture')==cls.architecture
        return super().load(path)
