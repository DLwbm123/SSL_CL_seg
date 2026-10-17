"""Dense native source adapter; preserves the sealed linear readout geometry."""
import torch
from torch import nn
from torch.nn import functional as F
from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model
from lcrseg.contracts import SegModelOutput

UPSTREAM_PATH = 'Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT'


class NativeSegmenter(nn.Module):
    num_classes = 4
    def __init__(self, reference, source):
        super().__init__()
        self.native = build_lcrseg_unet_jascl_model(reference, upstream_path=UPSTREAM_PATH,
                                                   input_channels=1, num_classes=4)
        expected = dict(dataset='mnms', domain='Siemens', seed=163, stage=0, sealed=True, updates=43600)
        if any(source.get(k) != v for k, v in expected.items()):
            raise ValueError('source must be the sealed Siemens V1 seed163 endpoint')
        self.native.load_state_dict(source['native'], strict=True)
        self.register_buffer('transform', source['transform'].clone())
        self.native.decoder.conv_logit.sigma.requires_grad_(False)
        self.native.decoder.conv_logit.grad_update.requires_grad_(False)

    def forward(self, image):
        m = self.native
        e1 = m.enc1(image); e2 = m.enc2(m.pool(e1)); e3 = m.enc3(m.pool(e2))
        feature = m.decoder.dec1(m.decoder.dec2(m.decoder.dec3(m.bottleneck(m.pool(e3)), e3), e2), e1)
        feature = torch.einsum('ij,bjhw->bihw', self.transform, feature)
        logits = F.conv2d(feature, m.decoder.conv_logit.mu.weight, bias=None, padding=0)
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=True)
        return SegModelOutput(logits, F.normalize(feature, dim=1), {'dec1': feature})
