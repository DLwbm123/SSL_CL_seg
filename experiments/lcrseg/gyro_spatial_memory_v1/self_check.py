"""Zero-update synthetic checks; neither patient access nor optimizer.step."""
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from .module import (SpatialProjection, JointGenerator, unit, sample_positions,
    rows, centroid, anchored_loss, rate_gap, retained_bytes, frame)


def run():
    torch.manual_seed(163)
    torch.set_num_threads(2)
    frames = torch.stack([frame(90163, 10, 32), frame(90164, 10, 32)]).double()
    views = F.normalize(torch.randn(96, 3, 32, dtype=torch.float64), dim=-1)
    c = .5*views[:, 0]+.25*views[:, 1]+.25*views[:, 2]
    loss, info = anchored_loss(c, frames, torch.tensor([0]*32+[1]*64))
    beta = 1/3
    spread = (torch.tensor([.5, .25, .25], dtype=c.dtype)[None, :, None]
        * (views-c[:, None]).square()).sum((1, 2))
    y = info['mixed'];rank = min(y.shape)
    deficit = ((1-beta)*spread.mean() + beta*(1-beta)*(c-info['anchors']).square().sum(1).mean()
        + (rank*y.square().sum()-torch.linalg.svdvals(y).sum().square())/(len(y)*rank))
    assert torch.allclose(1-info['kappa'].square(), deficit, atol=1e-12)
    assert 0 < loss < 1
    z = torch.eye(32, dtype=torch.float64)[:2]
    u = z.flip(0)
    assert abs(float(rate_gap(z, u))) < 1e-12
    matched = (z+z)/2
    mismatched = (z+u)/2
    assert abs(float(torch.linalg.svdvals(matched).sum())-2) < 1e-12
    assert abs(float(torch.linalg.svdvals(mismatched).sum())-1) < 1e-12

    projection = SpatialProjection(128)
    generator = JointGenerator(1)
    for m in (projection, generator):
        m.eval().requires_grad_(False)
    before = copy.deepcopy(generator.state_dict())
    raw = torch.randn(2, 32, 8, 8, requires_grad=True)
    code = unit(raw)
    image, mask = generator(code, (64, 64))
    assert image.shape == (2, 1, 64, 64) and mask.shape == (2, 4, 64, 64)
    assert torch.allclose(mask.sum(1), torch.ones(2, 64, 64), atol=2e-6)
    teacher = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 4, 1))
    query_x = torch.randn(2, 1, 64, 64)
    query_y = torch.randint(0, 4, (2, 64, 64))
    parameters = dict(teacher.named_parameters())
    def hypergradient(detach_image=False, detach_mask=False):
        synthetic_image = image.detach() if detach_image else image
        synthetic_mask = mask.detach() if detach_mask else mask
        logits = functional_call(teacher, parameters, (synthetic_image,))
        inner = -(synthetic_mask*logits.log_softmax(1)).sum(1).mean()
        derivatives = torch.autograd.grad(inner, tuple(parameters.values()), create_graph=True)
        updated = {k: value-.001*g for (k,value),g in zip(parameters.items(), derivatives)}
        outer = F.cross_entropy(functional_call(teacher, updated, (query_x,)), query_y)
        return torch.autograd.grad(outer, raw, retain_graph=True)[0]
    g_image = hypergradient(detach_mask=True)
    g_mask = hypergradient(detach_image=True)
    g_both = hypergradient()
    assert g_image.norm() > 0 and g_mask.norm() > 0
    assert torch.isfinite(g_both).all()
    assert all(p.grad is None for p in generator.parameters())
    assert all(torch.equal(before[k], v) for k, v in generator.state_dict().items())
    positions = sample_positions(2, 7, 'cpu')
    assert rows(code, positions).shape == (8, 32)
    all_codes = torch.empty(32, 32, 8, 8)
    memory_bytes = retained_bytes(projection, generator, code=all_codes) + frames.float().numel()*4
    assert memory_bytes <= 8*1024**2
    return dict(status='PASS', optimizer_step_calls=0, functional_updates_simulated=3,
        persistent_training_updates=0, patient_access=False,
        spectral_identity_absolute_error=float(abs(1-info['kappa'].square()-deficit)),
        permutation_rate_gap=float(rate_gap(z,u)), matched_nuclear=2., permuted_nuclear=1.,
        image_hypergradient_norm=float(g_image.norm()), mask_hypergradient_norm=float(g_mask.norm()),
        proposed_retained_bytes=memory_bytes,
        checks=['weighted_deficit_identity','rate_gap_permutation_blindness','paired_spectral_sensitivity',
            'joint_image_mask_shapes','image_and_mask_hypergradient_paths','frozen_generator_state',
            'spatial_indexing','retained_memory_cap'],
        limitation='Synthetic toy learner only; native second derivatives, data access and GPU qualification untested.')


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
