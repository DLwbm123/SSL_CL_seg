"""Independent local posterior with MAP G/F and an image reconstruction path."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from sc_bayeseg.model import normal_log_prob, student_log_prob, sar, sar_logdet
from sc_bayeseg.window_v1.unet_qualification import build_model
from sc_bayeseg.window_v1.patient_loss_factor import foreground_dice_loss
from lcrseg.models.unet import UNet2D


class Model(nn.Module):
    def __init__(self, family, num_classes=4):
        super().__init__()
        self.family = family
        self.backbone = build_model() if num_classes == 4 else UNet2D(1, num_classes, base_channels=16)
        self.backbone.projection_head = nn.Identity()
        if family == 'B':
            self.backbone.segmentation_head.requires_grad_(False)
            self.hc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32))
            self.hr = nn.Conv2d(16, 32, 1)
            self.hs = nn.Conv2d(16, 2, 1)
            self.hm = nn.Conv2d(16, 2, 1)
            self.g = nn.ModuleList([nn.Linear(16, 256), nn.Conv2d(16, 16, 3, padding=1), nn.Conv2d(16, num_classes - 1, 3, padding=1)])
            self.f = nn.Sequential(nn.Conv2d(num_classes, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1))
            self.register_buffer('logdet', torch.tensor(sar_logdet(128)))

    @staticmethod
    def gaussian(raw):
        m, r = raw.chunk(2, 1)
        return m, -4 + 3 * torch.tanh(r / 3)

    def forward(self, x, draws=4, image_loss=True):
        v = self.backbone(x)
        if self.family == 'U':
            return [v.logits], v.logits.sum() * 0
        feature = v.decoder_features['dec1']
        q = dict(c=self.gaussian(self.hc(feature.mean((2, 3)))),
                 r=self.gaussian(self.hr(feature.reshape(len(x), 16, 32, 4, 32, 4).mean((3, 5)))))
        if image_loss:
            q.update(s=self.gaussian(self.hs(feature)), m=self.gaussian(self.hm(feature)))
        logits = []; losses = []
        for _ in range(draws):
            z = {k: mean + sd.exp() * torch.randn_like(mean) for k, (mean, sd) in q.items()}
            base = F.interpolate(F.relu(self.g[0](z['c'])).reshape(-1, 16, 4, 4), (128, 128), mode='bilinear', align_corners=False)
            r = F.interpolate(z['r'], (128, 128), mode='bilinear', align_corners=False)
            a = self.g[2](F.relu(self.g[1](F.relu(base + r))))
            l = torch.cat([torch.zeros_like(a[:, :1]), a], 1); logits.append(l)
            if image_loss:
                structure = self.f(l.softmax(1)); structure = structure - structure.mean((2, 3), keepdim=True)
                logq = sum(normal_log_prob(z[k], *q[k]) for k in ('s', 'm'))
                joint = student_log_prob(x - z['s'] - z['m'], .1)
                joint = joint + student_log_prob(sar(z['s'] - structure), .2) + self.logdet
                joint = joint + normal_log_prob(z['m'], torch.zeros_like(z['m']), torch.full_like(z['m'], math.log(.5)))
                kl = sum(.5 * (m.square() + (2 * sd).exp() - 1 - 2 * sd).flatten(1).sum(1) for m, sd in (q['c'], q['r']))
                losses.append(((logq - joint + kl) / (128 * 128)).mean())
        return logits, torch.stack(losses).mean() if losses else a.sum() * 0

    def probability(self, x):
        logits, _ = self(x, image_loss=False)
        return torch.stack([l.softmax(1) for l in logits]).mean(0)


def supervised(logits, target):
    ce = torch.stack([F.cross_entropy(l, target, reduction='none').mean() for l in logits]).mean()
    p = torch.stack([l.softmax(1) for l in logits]).mean(0)
    if p.shape[1] == 4:
        return ce + foreground_dice_loss(p, target)
    truth = F.one_hot(target, p.shape[1]).permute(0, 3, 1, 2)[:, 1:].to(p.dtype)
    foreground = p[:, 1:]; count = truth.sum((2, 3)); present = count > 0
    dice = (2 * (foreground * truth).sum((2, 3)) + 1e-6) / (foreground.sum((2, 3)) + count + 1e-6)
    return ce + (((1 - dice) * present).sum(1) / present.sum(1).clamp_min(1)).mean()


def initialize(s, family):
    torch.manual_seed(163)
    model = Model(family, s.get('num_classes', 4))
    if s.get('initialization') == 'scratch':
        return model.cuda()
    ck = torch.load(s['pretrain'], map_location='cpu', weights_only=False)
    assert ck['step'] == 5000
    model.backbone.load_state_dict(ck['model'])
    if family == 'B':
        g = torch.load(s['generator'], map_location='cpu', weights_only=False)['model']
        for i, layer in enumerate(model.g):
            layer.weight.data.copy_(g[f'anatomy.{i}.mean_w']); layer.bias.data.copy_(g[f'anatomy.{i}.mean_b'])
        h = torch.load(s['recognition'], map_location='cpu', weights_only=False)['model']
        model.hc.load_state_dict({k[3:]: v for k, v in h.items() if k.startswith('hc.')})
        model.hr.load_state_dict({k[3:]: v for k, v in h.items() if k.startswith('hr.')})
    return model.cuda()
