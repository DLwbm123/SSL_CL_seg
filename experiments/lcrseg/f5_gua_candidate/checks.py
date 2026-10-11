"""Finite generated CPU algebra checks, without any optimizer or model calls."""
import torch
from .alignment import align_displacement


def run_checks():
    torch.set_num_threads(1)
    tensor = lambda x: torch.tensor(x, dtype=torch.double)
    # Positive coordinate scaling can reverse a previously safe dot product.
    gradient = tensor([1., -.5]); reference = tensor([1., 1.])
    metric = tensor([10., .1]); proposal = gradient / metric
    assert gradient @ reference > 0 and proposal @ reference < 0
    corrected, stats = align_displacement([proposal], [[reference]], [metric])
    expected = proposal - (proposal @ reference) / ((reference / metric) @ reference) * reference / metric
    assert torch.allclose(corrected[0], expected, atol=1e-12, rtol=1e-12)
    assert abs(float(corrected[0] @ reference)) < 1e-12 and stats['projected']
    examples = [('anisotropic_scaling', stats)]
    # Historical momentum is represented by a supplied proposal, not another update.
    out, stats = align_displacement([tensor([-1., 1.])], [[tensor([1., 0.])]], [tensor([1., 1.])])
    assert torch.equal(out[0], tensor([0., 1.]))
    examples.append(('historical_direction', stats))
    for refs in ([], [[tensor([0., 0.])]], [[tensor([1., 0.])]]):
        p = [tensor([1., 2.])]
        out, stats = align_displacement(p, refs, [tensor([1., 3.])])
        assert out is p and not stats['projected']
    empty, _ = align_displacement([], [[]], [])
    assert empty == []
    # Unequal blocks model differently scaled parameter groups in displacement units.
    p = [tensor([-.1]), tensor([-2., -3.])]
    refs = [[tensor([1.]), tensor([0., 0.])],
            [tensor([0.]), tensor([1., 0.])],
            [tensor([0.]), tensor([0., 1.])]]
    out, stats = align_displacement(p, refs, [tensor([.01]), tensor([2., 100.])])
    assert all(x.abs().max() < 1e-12 for x in out) and stats['active_constraints'] == 3
    examples.append(('three_active_blocks', stats))
    p = [tensor([-1., 2.])]
    out, _ = align_displacement(p, [[tensor([1., 0.])], [tensor([-1., 0.])]], [tensor([1., 1.])])
    assert torch.allclose(out[0], tensor([0., 2.]), atol=1e-12)
    # Verify the metric KKT conditions, including dtype round-trip, on a fixed set.
    rng = torch.Generator().manual_seed(1601)
    max_negative_cosine = 0.
    for i in range(64):
        dtype = torch.float32 if i % 2 else torch.float64
        p = torch.randn(7, generator=rng, dtype=torch.double).to(dtype)
        refs = [torch.randn(7, generator=rng, dtype=torch.double).to(dtype) for _ in range(3)]
        m = torch.exp(torch.randn(7, generator=rng, dtype=torch.double))
        saved = [x.clone() for x in [p, m] + refs]
        out, stats = align_displacement([p], [[r] for r in refs], [m]); d = out[0].double()
        assert all(torch.equal(a, b) for a, b in zip(saved, [p, m] + refs))
        assert out[0].dtype == dtype
        for r in refs:
            cosine = float(d @ r.double()) / max(float(d.norm() * r.double().norm()), 1e-30)
            max_negative_cosine = min(max_negative_cosine, cosine)
            assert cosine >= -1e-6
        objective = float(((d - p.double()).square() * m).sum())
        assert objective <= float((p.double().square() * m).sum()) + 1e-6
        if dtype == torch.float64:
            active = [r for r in refs if abs(float(d @ r)) < 1e-8]
            if active:
                a = torch.stack(active, 1); rhs = m * (d - p)
                coef = torch.linalg.lstsq(a, rhs).solution
                assert (coef >= -1e-8).all() and torch.allclose(a @ coef, rhs, atol=1e-8, rtol=1e-8)
            else:
                assert torch.equal(out[0], p)
    bad = [([tensor([1.])], [], [tensor([0.])], ValueError),
           ([tensor([1.])], [[tensor([1., 2.])]], [tensor([1.])], ValueError),
           ([tensor([float('nan')])], [], [tensor([1.])], FloatingPointError),
           ([tensor([1.])], [[tensor([1.])]] * 4, [tensor([1.])], ValueError),
           ([tensor([1e308])], [[tensor([1e308])]], [tensor([1.])], FloatingPointError)]
    for p, r, m, error in bad:
        try:
            align_displacement(p, r, m)
        except error:
            pass
        else:
            raise AssertionError('invalid algebra input accepted')
    return dict(status='MATH_PASS_NOT_NATIVE_QUALIFICATION',
                analytic_cases=8, randomized_cases=64, rejected_invalid_cases=len(bad),
                max_negative_cast_cosine=max_negative_cosine, examples=dict(examples),
                costs=dict(cpu_optimizer_updates=0, generated_cuda_updates=0,
                           real_L_updates=0, formal_updates=0, forwards=0, vjps=0),
                limitations=['no native Adam invocation, optimizer state alignment, training integration, teacher, deployment or checkpoint qualification',
                             'constructed mismatch does not establish mismatch frequency or medical performance'])
