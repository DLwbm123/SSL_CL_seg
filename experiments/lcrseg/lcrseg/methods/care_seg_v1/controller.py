"""Block-frozen, label-anchored controller on canonical reduced-resolution grids."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .core import evidence, geometry_basis, Repair
from .statistics import StateGMM, CandidateRisk, weights, choose


def resize(x, size, mode='bilinear'):
    return F.interpolate(x[None].float(), size=size, mode=mode,
                         **({'align_corners': False} if mode == 'bilinear' else {}))[0]


@torch.no_grad()
def collect(model, historical, dataset, size, labeled):
    rows = []
    device = next(model.parameters()).device
    for sample in dataset:
        if not labeled and 'label' in sample:
            raise ValueError('U exposed labels')
        image = sample['image' if labeled else 'weak_image'][None].to(device)
        first, second = model(image), model(image.flip(-1))
        if first.decoder_features is None or 'dec1' not in first.decoder_features:
            raise ValueError('CARE needs dec1 features')
        p = resize(first.logits[0].softmax(0), size).cpu()
        p2 = resize(second.logits[0].softmax(0).flip(-1), size).cpu()
        feature = F.normalize(resize(first.decoder_features['dec1'][0], size), dim=0).cpu()
        h = resize(historical(image).logits[0].softmax(0), size).cpu() if historical is not None else None
        valid = sample['valid_mask' if labeled else 'strong_valid_mask']
        row = dict(key=(sample['case_id'], sample['slice_index']), patient=sample['patient_id'],
                   shape=tuple(image.shape[-2:]), p=p, p2=p2, feature=feature, h=h,
                   valid=resize(valid, size, 'nearest')[0].bool())
        if labeled:
            row['label'] = resize(sample['label'][None], size, 'nearest')[0].long()
        rows.append(row)
    return rows


def prototypes(rows, tree, node):
    groups = defaultdict(list)
    for row in rows:
        label = tree.labels(row['label'], node)
        for k in range(len(node.children)):
            keep = (label == k) & row['valid']
            if keep.any():
                groups[(row['patient'], row['key'][0], k)].append(row['feature'][:, keep].mean(1))
    result = []
    for k in range(len(node.children)):
        by_patient = defaultdict(list)
        for (patient, case, child), values in groups.items():
            if child == k:
                by_patient[patient].append(torch.stack(values).mean(0))
        values = [torch.stack(v).mean(0) for v in by_patient.values()]
        if not values:
            return None
        result.append(torch.stack(values).mean(0))
    return F.normalize(torch.stack(result), dim=1)


def add_geometry(rows, tree, node, config):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['key'][0]].append(row)
    for case, volume in grouped.items():
        volume.sort(key=lambda r: -1 if r['key'][1] is None else r['key'][1])
        masks = [(tree.read(r['p'], node)[0].argmax(0) == 1).numpy() for r in volume]
        if config['geometry'] == 'physical3d':
            spec = config['volumes'][case]
            if [r['key'][1] for r in volume] != list(range(spec['depth'])):
                raise ValueError('physical geometry requires every volume slice')
            spacing = tuple(spec['spacing'])
            if len(spacing) != 3:
                raise ValueError('physical spacing must be z,y,x')
            scale = (1, volume[0]['shape'][0]/masks[0].shape[0], volume[0]['shape'][1]/masks[0].shape[1])
            taper, basis = geometry_basis(np.stack(masks), spacing=np.array(spacing)*scale, radius=config['repair_radius'])
            for i, row in enumerate(volume):
                row['basis'] = torch.tensor(basis[:, i]*taper[i], dtype=torch.float32)
        else:
            if any(r['key'][1] is not None for r in volume):
                raise ValueError('slice data requires explicit physical3d geometry')
            for row, mask in zip(volume, masks):
                spacing = tuple(2/max(s-1, 1) for s in mask.shape)
                taper, basis = geometry_basis(mask, spacing=spacing, radius=config['repair_radius'])
                row['basis'] = torch.tensor(basis*taper, dtype=torch.float32)


def describe(candidates, p):
    return torch.stack([torch.stack([q.max(0).values, -(q*q.clamp_min(1e-8).log()).sum(0),
                                   (q-p).abs().sum(0)], -1) for q in candidates], -2).reshape(-1, 3, 3).numpy()


def prepare(rows, tree, node, proto, repair):
    result = []
    for row in rows:
        p, mass = tree.read(row['p'], node)
        p2, _ = tree.read(row['p2'], node)
        similarities = torch.einsum('kd,dhw->khw', proto, row['feature'])
        p, pred, semantic, x = evidence(p, p2, similarities)
        r = repair.apply(p, row['basis']) if repair else p.clone()
        h = tree.read(row['h'], node)[0] if row['h'] is not None else p.clone()
        candidates = [p, r, h]
        eligible = np.ones((pred.numel(), 3), bool)
        eligible[:, 1] = repair is not None
        eligible[:, 2] = row['h'] is not None
        eligible &= (mass.flatten().numpy() > .01)[:, None]
        eligible &= row['valid'].flatten().numpy()[:, None]
        entry = dict(row=row, candidates=candidates, x=x.reshape(-1, 2).numpy(),
                     agreement=(pred == semantic).flatten().numpy(), descriptor=describe(candidates, p),
                     eligible=eligible, patients=[row['patient']]*pred.numel())
        if 'label' in row:
            target = tree.labels(row['label'], node)
            support = ((target >= 0) & row['valid']).flatten().numpy()
            entry['support'] = support
            entry['state'] = ((semantic == target).long()+2*(pred == target).long()).flatten().numpy()
            # Bounded Brier risk uses candidate soft targets, including repaired targets.
            onehot = F.one_hot(target.clamp_min(0), len(node.children)).permute(2, 0, 1)
            entry['loss'] = np.stack([(.5*(q-onehot).square().sum(0)).flatten().numpy() for q in candidates], 1)
        result.append(entry)
    return result


def concatenate(entries, key, *, labeled=False):
    values = []
    for entry in entries:
        value = np.asarray(entry[key])
        values.append(value[entry['support']] if labeled else value)
    return np.concatenate(values)


def fitting_subset(entries, limit, labeled=False):
    """Equal deterministic case quotas make ED/ES equal within each patient."""
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry['row']['patient'], entry['row']['key'][0])].append(entry)
    counts = defaultdict(int)
    for patient, case in grouped:
        counts[patient] += 1
    output = []
    for (patient, case), values in grouped.items():
        keys = ['x', 'agreement', 'descriptor', 'eligible', 'patients']
        if labeled:
            keys += ['state', 'loss']
        packed = {key: np.concatenate([np.asarray(v[key])[v['support'] if labeled else v['eligible'][:, 0]] for v in values]) for key in keys}
        n = len(packed['patients'])
        if n == 0:
            continue
        indices = np.linspace(0, n-1, min(n, max(1, limit//counts[patient])), dtype=int)
        packed = {key: value[indices] for key, value in packed.items()}
        packed['support'] = np.ones(len(indices), bool)
        output.append(packed)
    if not output:
        # Preserve correctly shaped empty arrays when the parent class is absent.
        empty = {key: concatenate(entries, key, labeled=labeled)[:0] for key in keys}
        empty['support'] = np.zeros(0, bool)
        output.append(empty)
    return output


def refresh(tree, config, snapshot, historical, datasets, temporal):
    a, b, u = [collect(snapshot, historical, dataset, config['grid'], labeled=i < 2)
               for i, dataset in enumerate(datasets)]
    maps = {r['key']: [] for r in u}
    masses = {r['key']: {frozenset(range(tree.classes)): torch.ones(config['grid'])} for r in u}
    stats = dict(repair_updates=0, em_iterations=0, accepted=0, decisions=0, action_P=0, action_R=0, action_H=0, supported_branches=0)
    fitted = []
    for node in tree.nodes:
        proto = prototypes(a, tree, node)
        repair = None
        if len(node.children) == 2 and config['repair_steps'] > 0:
            for rows in (a, b, u):
                add_geometry(rows, tree, node, config)
            repair = Repair(4 if config['geometry'] == 'physical3d' else 3,
                            config['repair_bound'], config['repair_steps'], config['repair_lr'])
            probabilities, bases, targets, patients = [], [], [], []
            for row in a:
                label = tree.labels(row['label'], node)
                keep = (label >= 0) & row['valid']
                probabilities.append(tree.read(row['p'], node)[0][1][keep])
                bases.append(row['basis'][:, keep].T)
                targets.append(label[keep])
                patients.extend([row['patient']]*int(keep.sum()))
            if patients and len(set(torch.cat(targets).tolist())) == 2:
                repair.fit(torch.cat(probabilities), torch.cat(bases), torch.cat(targets), torch.tensor(weights(patients)))
                stats['repair_updates'] += repair.updates
            else:
                repair = None
        if proto is None:
            for row in u:
                maps[row['key']].append(dict(target=tree.read(row['p'], node)[0], weight=torch.zeros(config['grid']), keep=torch.zeros(config['grid'], dtype=torch.bool), historical=None))
                for group in node.children:
                    masses[row['key']][frozenset(group)] = torch.zeros(config['grid'])
            fitted.append(None)
            continue
        be, ue = [prepare(rows, tree, node, proto, repair) for rows in (b, u)]
        full_ue = ue
        be = fitting_subset(be, config['fit_pixels_per_patient'], labeled=True)
        ue = fitting_subset(ue, config['fit_pixels_per_patient'])
        gmm = StateGMM(config['minimum_patients'], config['em_iterations'])
        gmm.fit(concatenate(be, 'x', labeled=True), concatenate(be, 'state', labeled=True),
                concatenate(be, 'agreement', labeled=True), concatenate(be, 'patients', labeled=True),
                concatenate(ue, 'x'), concatenate(ue, 'agreement'), concatenate(ue, 'patients'), classes=len(node.children))
        post, valid = gmm.predict(concatenate(be, 'x', labeled=True), concatenate(be, 'agreement', labeled=True))
        risk = CandidateRisk(config['minimum_patients'], config['risk_ridge'])
        risk.fit(post, concatenate(be, 'descriptor', labeled=True), concatenate(be, 'loss', labeled=True),
                 concatenate(be, 'eligible', labeled=True) & valid[:, None], concatenate(be, 'patients', labeled=True))
        stats['em_iterations'] += gmm.em_iterations
        stats['supported_branches'] += len(gmm.branches)
        fitted.append(dict(prototypes=proto, repair=repair, gmm=gmm, risk=risk))
        for entry in full_ue:
            row, candidates = entry['row'], entry['candidates']
            post, valid = gmm.predict(entry['x'], entry['agreement'])
            predicted = risk.predict(post, entry['descriptor'])
            eligible = entry['eligible'] & valid[:, None]
            predicted[~eligible] = np.inf
            identity = np.stack([q.argmax(0).flatten().numpy() for q in candidates], 1)
            predicted = temporal.update((node.name, row['key']), predicted, identity)
            action = choose(predicted, eligible, maximum=config['maximum_risk'], margin=config['risk_margin'])
            selected = torch.tensor(action).reshape(config['grid'])
            target = torch.zeros_like(candidates[0])
            for k, q in enumerate(candidates):
                target += q*(selected == k)
            parent = masses[row['key']][frozenset(node.classes)]
            weight = parent*(selected >= 0)
            for k, group in enumerate(node.children):
                masses[row['key']][frozenset(group)] = weight*target[k]
            keep = torch.tensor(eligible[:, 2] & (predicted[:, 2] <= config['maximum_risk'])).reshape(config['grid'])
            maps[row['key']].append(dict(target=target, weight=weight, keep=keep, historical=row['h']))
            stats['accepted'] += int((selected >= 0).sum())
            for k, name in enumerate(('P', 'R', 'H')):
                stats['action_'+name] += int((selected == k).sum())
            stats['decisions'] += selected.numel()
    return maps, fitted, stats
