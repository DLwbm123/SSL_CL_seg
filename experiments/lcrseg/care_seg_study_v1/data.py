"""Frozen manifest, patient->phase->slice schedules and phase-macro validation."""
from collections import defaultdict
import json
import h5py
import numpy as np
import torch
from lcrseg.data import H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled


def datasets(root, site):
    common = dict(data_root=root, seed=0, dataset='mnms', sites=[site])
    labeled = H5LabeledDataset(**common, transform=LabeledTransform())
    unlabeled = H5UnlabeledDataset(**common, transform=WeakStrongTransform(cutout_probability=0))
    for dataset in (labeled, unlabeled):
        if any(r['cohort'] != 'canonical320' or r['evaluation_eligible'] != 'true' for r in dataset.records):
            raise ValueError('only eligible frozen canonical320 records are admitted')
    return labeled, unlabeled


def volumes(*datasets):
    result = {}
    for dataset in datasets:
        for row in dataset.records:
            with h5py.File(dataset.h5_root / row['image_h5_relpath'], 'r') as f:
                shape = f['image'].shape
                result[row['case_id']] = dict(depth=shape[0], spacing=json.loads(f.attrs['processed_spacing']))
    return result


class PatientBatcher:
    def __init__(self, dataset, patients, labeled, seed):
        self.dataset, self.labeled, self.seed = dataset, labeled, seed
        self.groups = defaultdict(lambda: defaultdict(list))
        for i,s in enumerate(dataset.samples):
            if s.row['patient_id'] in patients:
                self.groups[s.row['patient_id']][s.row['case_id']].append(i)
        self.patients = sorted(self.groups)
        if len(self.patients) < 2:
            raise ValueError('batch needs two distinct patients')
    def batch_at(self, step):
        generator = torch.Generator().manual_seed(self.seed+step*100003+(0 if self.labeled else 700001))
        indices = []
        for pos in torch.randperm(len(self.patients), generator=generator)[:2].tolist():
            cases = self.groups[self.patients[pos]]
            case = sorted(cases)[int(torch.randint(len(cases), (), generator=generator))]
            indices.append(cases[case][int(torch.randint(len(cases[case]), (), generator=generator))])
        # Keep augmentation streams equal across arms even when controller work differs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed+step*100003+(11 if self.labeled else 700013))
            values = [self.dataset[i] for i in indices]
        return (collate_labeled if self.labeled else collate_unlabeled)(values)


@torch.no_grad()
def evaluate(model, root, site):
    dataset = H5LabeledDataset(root, seed=0, dataset='mnms', sites=[site], roles=['val'], transform=None)
    if any(r['cohort'] != 'canonical320' or r['evaluation_eligible'] != 'true' for r in dataset.records):
        raise ValueError('invalid validation cohort')
    model.eval()
    device = next(model.parameters()).device
    counts = defaultdict(lambda: np.zeros((3,3), dtype=np.int64))
    metadata = {}
    for i in range(0, len(dataset), 2):
        samples = [dataset[j] for j in range(i, min(i+2,len(dataset)))]
        batch = collate_labeled(samples)
        pred = model(batch.image.to(device)).logits.argmax(1).cpu().numpy()
        for j,s in enumerate(samples):
            case = s['case_id']; y = s['label'].numpy()
            metadata[case] = s['patient_id']
            for k in range(1,4):
                p,t = pred[j] == k,y == k
                counts[case][k-1] += [int((p&t).sum()),int(p.sum()),int(t.sum())]
    patient_scores = defaultdict(list)
    for case,v in counts.items():
        score = np.divide(2*v[:,0],v[:,1]+v[:,2],out=np.ones(3),where=(v[:,1]+v[:,2])>0)
        patient_scores[metadata[case]].append(score)
    if any(len(v)!=2 for v in patient_scores.values()):
        raise ValueError('each validation patient must have ED and ES')
    rows = {p:np.mean(v,axis=0).tolist() for p,v in patient_scores.items()}
    mean = np.mean(list(rows.values()),axis=0)
    return dict(site=site,patients=len(rows),class_dice=mean.tolist(),macro_dice=float(mean.mean())), rows
