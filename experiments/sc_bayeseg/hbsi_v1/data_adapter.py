import csv
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F


def leaf_observations(labels, classes, groups=()):
    """Leaf codes 0..K-1, parent codes K+j, -1 means unobserved."""
    if labels.dtype != torch.long or torch.any((labels < -1) | (labels >= classes+len(groups))):
        raise ValueError('invalid labels')
    mapping=torch.eye(classes,dtype=torch.bool,device=labels.device)
    for group in groups:
        if not group or any(not 0<=k<classes for k in group):raise ValueError('invalid leaf group')
        row=torch.zeros(classes,dtype=torch.bool,device=labels.device);row[list(group)]=True
        mapping=torch.cat([mapping,row[None]],0)
    return {'y_obs': mapping[labels.clamp_min(0)].permute(0,3,1,2), 'obs_mask': labels>=0}


class AccessGate:
    def __init__(self, stage, mode='train'):
        if mode not in ('train', 'evaluation'):
            raise ValueError('access mode')
        self.stage, self.mode = stage, mode
        self.events = []

    def check(self, record, kind, prediction_sealed=False):
        role = record['role']
        if role == 'test' or kind not in ('image', 'label'):
            raise PermissionError('sealed test or unknown capability')
        if self.mode == 'train':
            if record['stage'] != self.stage or role not in ('trainL', 'trainU'):
                raise PermissionError('old domain or non-training data')
            if role == 'trainU' and kind == 'label':
                raise PermissionError('U label access')
        elif role != 'val' or (kind == 'label' and not prediction_sealed):
            raise PermissionError('evaluation labels require sealed predictions')
        self.events.append({'role': role, 'kind': kind})

    def open(self, record, kind, callback, prediction_sealed=False):
        self.check(record, kind, prediction_sealed)
        return callback()


def confined(root, relative):
    p = Path(relative)
    if not relative or p.is_absolute() or '..' in p.parts:
        raise PermissionError('unsafe data path')
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PermissionError('data path escapes root')
    return target


class CurrentTrainingData:
    """Uniform slice sampling; no augmentation, previous-domain records, or U label capability."""
    def __init__(self, root, dataset, domain, stage, config, seed=0):
        import h5py
        self.root = Path(root)
        self.config, self.dataset, self.stage = config, dataset, stage
        self.gate = AccessGate(stage)
        manifest = self.root / 'manifests' / 'training' / f'lcrseg_v1_seed{seed}.csv'
        raw = manifest.read_bytes()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        self.records = {'L': [], 'U': []}
        for row in csv.DictReader(raw.decode().splitlines()):
            if row['dataset'] != dataset or row.get('site_or_vendor', row.get('site')) != domain:
                continue
            role = {'train_labeled': 'L', 'train_unlabeled': 'U'}.get(row['primary_20pct_split'])
            if role is None:
                continue
            if role == 'U' and any(row.get(k) for k in ('label_h5_relpath', 'hidden_label', 'diagnostic_path')):
                raise PermissionError('unlabeled manifest contains hidden label capability')
            record = {'role': 'train' + role, 'stage': stage,
                      'image': confined(self.root / 'h5/v1', row['image_h5_relpath'])}
            if role == 'L':
                record['label'] = confined(self.root / 'h5/v1', row['label_h5_relpath'])
            self.gate.check(record, 'image')
            with h5py.File(record['image'], 'r') as f:
                shape = f['image'].shape
            if len(shape) != 3:
                raise ValueError('expected frozen MRI/RGB shape')
            indices = [None] if dataset == 'fundus' else range(shape[0])
            self.records[role].extend(dict(record, index=i) for i in indices)
        if not self.records['L']:
            raise ValueError('no current labelled training observations')

    @property
    def counts(self):
        return {k: len(v) for k, v in self.records.items()}

    def read(self, record, kind):
        import h5py
        import numpy as np
        self.gate.check(record, kind)
        with h5py.File(record[kind], 'r') as f:
            a = f[kind][...] if record['index'] is None else f[kind][record['index']]
        return torch.from_numpy(np.array(a, copy=True))

    def batch(self, role, size, generator):
        records = self.records[role]
        if not records:
            return None
        if size < 1:
            raise ValueError('batch size')
        indices = torch.randint(len(records), (size,), generator=generator).tolist()
        xs = []; ys = []; native_shapes = []
        for i in indices:
            record = records[i]
            x = self.read(record, 'image').to(getattr(torch, self.config.dtype))
            if self.dataset == 'fundus':
                x = x / 255.
            else:
                x = x[None]
            native_shapes.append(list(x.shape[-2:]))
            xs.append(F.interpolate(x[None], (self.config.height, self.config.width), mode='bilinear', align_corners=False)[0])
            if role == 'L':
                y = self.read(record, 'label')
                ys.append(F.interpolate(y[None, None].double(), (self.config.height, self.config.width), mode='nearest')[0, 0].long())
        result = {'x': torch.stack(xs), 'role': 'train' + role, 'stage': self.stage, 'native_shapes': native_shapes}
        if role == 'L':
            result['observations'] = leaf_observations(torch.stack(ys), self.config.classes, self.config.label_groups)
        return result


def restore_probability(p, shape):
    q = F.interpolate(p, shape, mode='bilinear', align_corners=False)
    return q / q.sum(1, keepdim=True)
