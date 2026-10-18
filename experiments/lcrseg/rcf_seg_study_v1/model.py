"""Official TransUNet, task-isolated Q/V LoRA, and label-free prototype routing."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lcrseg.contracts import SegModelOutput
from .upstream.vit_seg_modeling import VisionTransformer, CONFIGS


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.dropout = nn.Dropout(.05)
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return self.up(self.down(self.dropout(x))) * 2.


class TaskLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active = 'source'
    def forward(self, x):
        y = self.base(x)
        return y if self.active == 'source' else y + self.adapters[self.active](x)


class Segmenter(nn.Module):
    num_classes = 4
    def __init__(self):
        super().__init__()
        cfg = copy.deepcopy(CONFIGS['R50-ViT-B_16'])
        cfg.n_classes, cfg.n_skip, cfg.patches.grid = 4, 3, (14, 14)
        self.net = VisionTransformer(cfg, img_size=224, num_classes=4)
        self.isolated = False
        self.active = 'source'
        self.routers = {}

    def pretrained(self, path):
        with np.load(path) as weights:
            self.net.load_from(weights)

    def isolate(self):
        self.requires_grad_(False)
        for block in self.net.transformer.encoder.layer[-6:]:
            for name in ('query', 'value'):
                if not isinstance(getattr(block.attn, name), TaskLinear):
                    setattr(block.attn, name, TaskLinear(getattr(block.attn, name)))
        self.isolated = True
        self.train(False)

    def add_task(self, name):
        if not self.isolated or name == 'source':
            raise ValueError('isolate source before adding a target expert')
        self.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        for layer in self.task_layers():
            if name in layer.adapters:
                raise ValueError('expert already exists')
            layer.adapters[name] = Adapter(layer.base.in_features).to(layer.base.weight.device)
        self.activate(name)

    def task_layers(self):
        return [m for m in self.net.modules() if isinstance(m, TaskLinear)]

    def activate(self, name):
        self.active = name
        for layer in self.task_layers():
            if name != 'source' and name not in layer.adapters:
                raise ValueError('unknown expert')
            layer.active = name

    def train(self, mode=True):
        super().train(mode)
        if self.isolated:
            self.net.eval()  # Includes decoder BN buffers and frozen backbone dropout.
            if self.active != 'source':
                for layer in self.task_layers():
                    layer.adapters[self.active].train(mode)
        return self

    def encode(self, x):
        x = F.interpolate(x, (224,224), mode='bilinear', align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        return self.net.transformer(x)

    def forward(self, x):
        tokens, _, skips = self.encode(x)
        decoded = self.net.decoder(tokens, skips)
        logits = self.net.segmentation_head(decoded)
        logits = F.interpolate(logits, x.shape[-2:], mode='bilinear', align_corners=False)
        features = tokens.transpose(1,2).reshape(x.shape[0],768,14,14)
        return SegModelOutput(logits, F.normalize(features, dim=1))

    @torch.no_grad()
    def frozen_features(self, x):
        previous = self.active
        self.activate('source')
        try:
            tokens, _, _ = self.encode(x)
        finally:
            self.activate(previous)
        return F.normalize(tokens.transpose(1,2).reshape(x.shape[0],768,14,14), dim=1)

    @torch.no_grad()
    def routed(self, x):
        if not self.routers:
            raise RuntimeError('no admitted router')
        features = self.frozen_features(x)
        vectors = F.normalize(features.mean((-2,-1)), dim=1)
        names = list(self.routers)
        scores = torch.stack([(vectors @ self.routers[n].to(x.device).T).topk(
            min(3,len(self.routers[n])), dim=1).values.mean(1) for n in names],1)
        choices = scores.argmax(1)
        previous = self.active
        logits = x.new_empty((len(x),4,*x.shape[-2:]))
        try:
            for j,n in enumerate(names):
                mask = choices == j
                if mask.any():
                    self.activate(n)
                    logits[mask] = self(x[mask]).logits
        finally:
            self.activate(previous)
        return SegModelOutput(logits, features), choices


class RoutedView(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.counts = [0] * len(model.routers)
    def forward(self, x):
        out, choice = self.model.routed(x)
        for k in range(len(self.counts)):
            self.counts[k] += int((choice == k).sum())
        return out


def cosine_kmeans(x, k):
    x = F.normalize(x.float(), dim=1)
    k = min(k,len(x))
    if k < 1: raise ValueError('empty routing population')
    # Deterministic farthest-point initialization, without labels or test data.
    indices = [0]
    for _ in range(1,k):
        indices.append(int((x @ x[indices].T).max(1).values.argmin()))
    centers = x[indices].clone()
    for _ in range(25):
        labels = (x @ centers.T).argmax(1)
        new = torch.stack([x[labels==j].mean(0) if (labels==j).any() else centers[j] for j in range(k)])
        new = F.normalize(new,dim=1)
        delta = (new-centers).abs().max()
        centers = new
        if delta < 1e-4: break
    return centers, (x @ centers.T).argmax(1)


def routing_prototypes(vectors):
    x = F.normalize(vectors.float(),dim=1)
    centers, labels = cosine_kmeans(x,10)
    result = [centers]
    for j,c in enumerate(centers):
        subset = x[labels==j]
        if len(subset) < 2: continue
        distance = 1-subset@c
        lo,hi = torch.quantile(distance,torch.tensor([.70,.95]))
        boundary = subset[(distance>=lo)&(distance<=hi)]
        if len(boundary): result.append(cosine_kmeans(boundary,3)[0])
    return torch.cat(result).cpu()


def load_deployment(payload,device='cpu'):
    model=Segmenter().to(device)
    tasks=payload.get('tasks',[])
    if tasks:
        model.isolate()
        for task in tasks:model.add_task(task)
    model.load_state_dict(payload['model'],strict=True)
    model.routers=payload['routers']
    model.requires_grad_(False);model.eval()
    return RoutedView(model)
