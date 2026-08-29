from __future__ import annotations

import torch

from scripts.audit_bprc_x1_feasibility import X1_SCALE, _candidate_objectives


def test_bprc_x1_candidate_is_exact_b2_div_three() -> None:
    b0 = torch.tensor(2.0, requires_grad=True)
    b2 = torch.tensor(6.0, requires_grad=True)
    candidates = _candidate_objectives(
        {
            "current_scores": torch.zeros((1, 3, 1, 1)),
            "relations": {"B0": b0, "B2": b2},
        }
    )
    assert X1_SCALE == 1.0 / 3.0
    assert candidates["X0"] is b0
    assert torch.equal(candidates["X1"], torch.tensor(2.0))
    candidates["X1"].backward()
    assert torch.equal(b2.grad, torch.tensor(1.0 / 3.0))


def test_bprc_x1_rejects_non_three_class_inputs() -> None:
    try:
        _candidate_objectives(
            {
                "current_scores": torch.zeros((1, 2, 1, 1)),
                "relations": {"B0": torch.tensor(0.0), "B2": torch.tensor(0.0)},
            }
        )
    except AssertionError as error:
        assert "fixed C=3" in str(error)
    else:
        raise AssertionError("two-class candidate must be rejected")
