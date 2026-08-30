from __future__ import annotations

import torch
from torch.nn import functional as F

from di_dmpa_jascl.modeling import (
    compute_pas_validity, masked_probability_consistency_loss, pas_probability_objective,
    gradient_norm, RepairedMeanTeacher,
)
from .test_model_checkpoint import TinySegNet


def _logits():
    torch.manual_seed(42)
    return torch.randn(2, 3, 4, 4, requires_grad=True), torch.randn(2, 3, 4, 4, requires_grad=True)


def test_pas_probability_loss_matches_manual_formula():
    s, t = _logits()
    mask = torch.arange(32).reshape(2, 4, 4) % 2 == 0
    observed = masked_probability_consistency_loss(s, t, mask)
    manual = ((s.softmax(1) - t.detach().softmax(1)).square().sum(1)[mask]).sum() / mask.sum()
    torch.testing.assert_close(observed, manual, rtol=1e-7, atol=1e-7)


def test_pas_validity_is_student_teacher_intersection():
    class Fixed(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.nn.Parameter(logits)
        def forward(self, images, *, stochastic_classifier):
            assert stochastic_classifier is True
            return self.logits, torch.ones(1, 2, 1, 3)
    s = Fixed(torch.tensor([[[[5., 0., 5.]], [[0., 0., 0.]]]]))
    t = Fixed(torch.tensor([[[[5., 5., 0.]], [[0., 0., 0.]]]]))
    loss, vs, vt, joint = pas_probability_objective(s, t, None, torch.ones(2, 2))
    assert joint.tolist() == [[[True, False, False]]]
    assert torch.equal(joint, vs.valid_mask & vt.valid_mask)
    assert joint.dtype == torch.bool and not joint.requires_grad
    assert loss.requires_grad


def test_invalid_pixels_have_zero_loss_contribution():
    s, t = _logits()
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, 1, 1] = True
    loss = masked_probability_consistency_loss(s, t, mask)
    grad, = torch.autograd.grad(loss, s)
    assert torch.count_nonzero(grad.permute(0, 2, 3, 1)[~mask]) == 0
    mutated = t.detach().clone().permute(0, 2, 3, 1)
    mutated[~mask] = 100
    torch.testing.assert_close(loss, masked_probability_consistency_loss(s, mutated.permute(0, 3, 1, 2), mask))


def test_zero_valid_pixels_returns_graph_connected_zero():
    s, t = _logits()
    loss = masked_probability_consistency_loss(s, t, torch.zeros(2, 4, 4, dtype=torch.bool))
    assert loss.requires_grad and loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    assert s.grad is not None and torch.count_nonzero(s.grad) == 0
    assert t.grad is None


def test_consistency_loss_requires_grad():
    s, t = _logits()
    assert masked_probability_consistency_loss(s, t, torch.ones(2, 4, 4, dtype=torch.bool)).requires_grad


def _model_losses():
    torch.manual_seed(21)
    wrapper = RepairedMeanTeacher(TinySegNet(), TinySegNet())
    x = torch.randn(2, 3, 8, 8)
    s, _ = wrapper.student(x, stochastic_classifier=True)
    with torch.no_grad():
        t, _ = wrapper.teacher(x, stochastic_classifier=True)
    loss_u = masked_probability_consistency_loss(s, t, torch.ones(2, 8, 8, dtype=torch.bool))
    loss_sup = F.cross_entropy(s, torch.randint(0, 3, (2, 8, 8)))
    params = list(wrapper.student.parameters())
    return wrapper, params, loss_sup, loss_u


def test_unsupervised_gradient_is_nonzero_when_valid():
    _, params, _, loss_u = _model_losses()
    grads = torch.autograd.grad(loss_u, params, allow_unused=True)
    assert gradient_norm(grads) > 1e-8


def test_teacher_receives_no_gradient():
    wrapper, _, sup, u = _model_losses()
    (sup + 0.5 * u).backward()
    assert all(parameter.grad is None for parameter in wrapper.teacher.parameters())


def test_prototypes_and_masks_receive_no_gradient():
    logits = torch.tensor([[[[5., 5.]], [[0., 0.]]]], requires_grad=True)
    features = torch.ones(1, 2, 1, 1, requires_grad=True)
    prototypes = torch.ones(2, 2, requires_grad=True)
    validity = compute_pas_validity(logits, features, prototypes, 0.7, 0.7)
    assert validity.valid_mask.shape == (1, 1, 2)
    mask = validity.valid_mask.float().requires_grad_()
    teacher = torch.zeros_like(logits, requires_grad=True)
    masked_probability_consistency_loss(logits, teacher, mask).backward()
    assert prototypes.grad is None and features.grad is None and mask.grad is None
    assert teacher.grad is None
    assert all(not x.requires_grad for x in vars(validity).values())


def test_total_gradient_differs_from_supervised_gradient():
    _, params, sup, u = _model_losses()
    gs = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(sup + 0.5 * u, params, allow_unused=True)
    assert gradient_norm([b-a for a, b in zip(gs, gt) if a is not None and b is not None]) > 1e-8


def test_lambda_zero_removes_only_unsupervised_gradient():
    _, params, sup, u = _model_losses()
    gs = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
    g0 = torch.autograd.grad(sup + 0.0 * u, params, allow_unused=True)
    for a, b in zip(gs, g0):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_pas_thresholds_are_strict_and_classes_are_not_void():
    logits = torch.tensor([[[[0.]], [[5.]]]])
    features = torch.ones(1, 2, 1, 1)
    prototypes = torch.ones(2, 2)
    v = compute_pas_validity(logits, features, prototypes, 0.7, 0.7)
    assert v.predicted_class.item() == 1 and v.valid_mask.item()
    rejected = compute_pas_validity(logits, features, prototypes, float(v.confidence.item()), 0.7)
    assert not rejected.valid_mask.item() and rejected.predicted_class.item() == 1
