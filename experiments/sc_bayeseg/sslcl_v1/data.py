"""Frozen patient-balanced schedules and two-number domain style statistics."""
from collections import defaultdict
from pathlib import Path
import json
import csv
import math
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset
from lcrseg.data.h5_dataset import load_training_records


class Schedule:
    def __init__(self, data, site, unlabeled=False, seed=163, dataset="mnms", image_size=128, batch_size=2):
        cls = H5UnlabeledDataset if unlabeled else H5LabeledDataset
        self.ds = cls(data, seed=0, dataset=dataset, sites=[site], transform=None)
        assert type(batch_size) is int and 1 <= batch_size <= 4
        self.batch_size = batch_size
        self.image_size = image_size
        self.unlabeled = unlabeled
        self.seed = seed
        self.groups = defaultdict(lambda: defaultdict(list))
        for i, sample in enumerate(self.ds.samples):
            self.groups[sample.row['patient_id']][sample.row['case_id']].append(i)
        self.patients = sorted(self.groups)

    def batch(self, step):
        g = torch.Generator().manual_seed(self.seed + 100003 * step + 1000000007 * self.unlabeled)
        images = []; labels = []; indices = []
        for j in torch.randperm(len(self.patients), generator=g)[:self.batch_size].tolist():
            phases = self.groups[self.patients[j]]; names = sorted(phases)
            phase = names[int(torch.randint(len(names), (), generator=g))]
            ids = phases[phase]; index = ids[int(torch.randint(len(ids), (), generator=g))]
            item = self.ds[index]; indices.append(index)
            key = 'weak_image' if self.unlabeled else 'image'
            x = F.interpolate(item[key][None], (self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
            y = None if self.unlabeled else F.interpolate(item['label'][None, None].float(), (self.image_size, self.image_size), mode='nearest')[0, 0].long()
            for dim in (-1, -2):
                if int(torch.randint(2, (), generator=g)):
                    x = x.flip(dim); y = y.flip(dim) if y is not None else y
            images.append(x)
            if y is not None: labels.append(y)
        return torch.stack(images).cuda(), torch.stack(labels).cuda() if labels else None, indices


@torch.no_grad()
def style(schedule):
    if schedule.ds.records[0]["dataset"] == "fundus":
        from .fundus import rgb_style
        return rgb_style(schedule)
    # Equal patients, equal phases, equal slices. Only labelled training images.
    patients = []
    for patient in schedule.patients:
        phases = []
        for ids in schedule.groups[patient].values():
            values = []
            for i in ids:
                x = schedule.ds._read_image(schedule.ds.samples[i])
                x = F.interpolate(x[None], (128, 128), mode='bilinear', align_corners=False)
                values.append(torch.stack([x.mean(), x.std(unbiased=False)]))
            phases.append(torch.stack(values).mean(0))
        patients.append(torch.stack(phases).mean(0))
    return torch.stack(patients).mean(0).tolist()


def view(x, current, history, step, seed=163, mode="history", history_strength=1.0):
    if len(current) == 6:
        from .fundus import rgb_view
        return rgb_view(x, current, history, step, seed, mode, history_strength)
    g = torch.Generator(device=x.device).manual_seed(seed * 1000 + step)
    params = torch.rand((len(x), 5, 1, 1), device=x.device, generator=g)
    assert mode in ('history', 'wide', 'reverse')
    assert 0 <= history_strength <= 1
    if history is not None and history_strength != 1:
        assert mode == 'history'
        history = [current[0] + history_strength * (history[0] - current[0]),
                   current[1] * math.exp(history_strength * math.log(history[1] / current[1]))]
    if mode == 'wide': history = None
    if mode == 'reverse':
        assert history is not None
        history = [2 * current[0] - history[0], current[1] ** 2 / history[1]]
    old = (params[:, 0:1] < .5) if history is not None or mode == 'wide' else torch.zeros_like(params[:, 0:1], dtype=torch.bool)
    hist = current if history is None else history
    mean = torch.where(old, float(hist[0]), float(current[0]))
    sd = torch.where(old, float(hist[1]), float(current[1])).clamp_min(1e-4)
    if mode == 'wide':
        # Only current-domain statistics; same five random numbers and 50% branch.
        generic_mean = current[0] + .75 * current[1] * (2 * params[:, 4:5] - 1)
        generic_sd = current[1] * torch.exp(math.log(1.5) * (4 * params[:, 0:1] - 1))
        mean = torch.where(old, generic_mean, float(current[0]))
        sd = torch.where(old, generic_sd, float(current[1])).clamp_min(1e-4)
    centered = (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), unbiased=False, keepdim=True).clamp_min(1e-4)
    axis = torch.linspace(0, 2 * torch.pi, x.shape[-1], device=x.device)
    bias = (torch.cos(axis)[None, None, :, None] + torch.cos(axis)[None, None, None, :]) / 2
    return centered * sd * (.8 + .4 * params[:, 1:2]) + mean + sd * (.1 * params[:, 2:3] - .05) + sd * (.1 * params[:, 3:4] - .05) * bias


def audit(s):
    table = json.loads(Path(s['split']).read_text())['records']
    canonical = {r['patient_id']: r for r in table if r['cohort'] == 'canonical320'}
    seen = {}; report = {}; private = {}
    for site in ('Siemens', 'Philips'):
        with (Path(s['data']) / 'manifests/training/lcrseg_v1_seed0.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['dataset'] == 'mnms' and r['site_or_vendor'] == site]
        load_training_records(Path(s['data']), seed=0, dataset='mnms', sites=[site], roles=['train_labeled', 'train_unlabeled', 'val'])
        report[site] = {}; private[site] = rows
        for role in ('train_labeled', 'train_unlabeled', 'val', 'test'):
            selected = [r for r in rows if r['primary_20pct_split'] == role]
            groups = defaultdict(list)
            for r in selected:
                assert r['cohort'] == 'canonical320'
                reference = canonical[r['patient_id']]
                assert reference['site_or_vendor'] == site and reference['primary_20pct_split'] == role
                if role == 'train_unlabeled': assert not r['label_h5_relpath']
                groups[r['patient_id']].append(r)
            for patient, phases in groups.items():
                assert patient not in seen and len(phases) == 2 and len({r['phase'] for r in phases}) == 2
                seen[patient] = (site, role)
            report[site][role] = dict(patients=len(groups), phases=len(selected))
        assert sum(v['patients'] for v in report[site].values()) == (95 if site == 'Siemens' else 125)
    return dict(status='PASS', domains=report, patient_roles_disjoint=True, frozen_split_matches=True, hidden_label_paths_absent=True, test_files_opened=0), private
