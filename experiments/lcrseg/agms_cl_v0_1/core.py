"""Parent-set supervision and lagged current-L risk; no patient IO."""
import torch


def weights(risk, adaptive):
    r = risk.detach()
    return (.1 + .7 * torch.softmax(-r / .1, dim=0) if adaptive else torch.full_like(r, 1/3)).detach()


def brier(probabilities, labels):
    """Equal present classes within each image, then equal valid images."""
    valid = labels != 255
    target = (labels == 1) | (labels == 2)
    output = []
    for q in probabilities:
        d = q.detach()[:, 1:3].sum(1)
        images = []
        for b in range(len(labels)):
            classes = []
            for cls in (False, True):
                m = valid[b] & (target[b] == cls)
                if m.any():classes.append((d[b][m] - float(cls)).square().mean())
            if classes:images.append(torch.stack(classes).mean())
        if not images:return None
        output.append(torch.stack(images).mean())
    return torch.stack(output).detach()


def selector(probabilities, geometry, fine, arm, risk):
    q = torch.stack([p.detach() for p in probabilities])
    d = q[:, :, 1:3].sum(2)
    alpha = weights(risk, arm == 'A5')
    mean = (alpha[:, None, None, None] * d).sum(0) if len(q) == 3 else d[0]
    var = (alpha[:, None, None, None] * (d - mean).square()).sum(0) if len(q) == 3 else torch.zeros_like(mean)
    mask = geometry.bool() & ~fine.bool() & (d[0] >= .9)
    if arm in ('A4', 'A5'):mask = mask & (mean >= .9) & (var <= .01)
    return mask.detach(), dict(alpha=alpha, mean=mean.detach(), variance=var.detach(), disc=d.detach())


def parent_loss(logp, mask, geometry):
    denom = geometry.sum((1, 2))
    valid = denom > 0
    if not valid.any():return logp.sum() * 0
    nll = -torch.logsumexp(logp[:, 1:3], dim=1)
    return ((nll * mask.detach()).sum((1, 2)) / denom.clamp_min(1))[valid].mean()


def coverage(fine, coarse, geometry, stats):
    rows = []
    for i in range(len(geometry)):
        n = int(geometry[i].sum());m = coarse[i]
        rows.append(dict(geometry=n, fine=int((fine[i] & geometry[i]).sum()), coarse=int(m.sum()),
                         ignore=int((~geometry[i]).sum()),
                         fine_fraction=None if not n else int((fine[i]&geometry[i]).sum())/n,
                         coarse_fraction=None if not n else int(m.sum())/n,
                         ignore_fraction=float((~geometry[i]).float().mean()),
                         selected_disc_per_scale=[float(d[i][m].mean()) if m.any() else None for d in stats['disc']],
                         selected_disc=float(stats['disc'][0,i][m].mean()) if m.any() else None,
                         selected_variance=float(stats['variance'][i][m].mean()) if m.any() else None))
    return rows


def shadow_counts(probabilities, labels, geometry, fine, coarse):
    # Selection already finished without labels. This independent counter has no
    # return path to the training loss or selector.
    rows = []
    pred = probabilities[0].detach().argmax(1)
    for i in range(len(labels)):
        mask = coarse[i] & geometry[i] & (labels[i] != 255)
        n = int(mask.sum())
        rows.append(dict(selected=n, parent_correct=int(((labels[i] == 1) | (labels[i] == 2))[mask].sum()),
                         fine_correct=int((pred[i] == labels[i])[mask].sum()),
                         parent_accuracy=None if not n else float(((labels[i] > 0) & (labels[i] < 3))[mask].float().mean()),
                         fine_accuracy=None if not n else float((pred[i] == labels[i])[mask].float().mean()),
                         rim_support=int((mask&(labels[i]==1)).sum()),cup_support=int((mask&(labels[i]==2)).sum()),
                         rim_fine_correct=int((mask&(labels[i]==1)&(pred[i]==1)).sum()),
                         cup_fine_correct=int((mask&(labels[i]==2)&(pred[i]==2)).sum()),
                         interpretation='current training L, not independent generalization'))
    return rows
