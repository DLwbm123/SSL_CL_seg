from __future__ import annotations

from pathlib import Path

import pytest
import torch

from di_dmpa_jascl.config import load_yaml
from di_dmpa_jascl.runner import Gate0RepairedRunner

from .test_model_checkpoint import TinySegNet


ROOT = Path(__file__).resolve().parents[2]


def _runner(output_dir: Path) -> Gate0RepairedRunner:
    config = load_yaml(ROOT / "configs/gate0_repaired_v2/fundus_pas_probmse.yaml")
    protocol = load_yaml(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml")
    return Gate0RepairedRunner(
        repo_root=ROOT,
        config=config,
        protocol=protocol,
        seed=0,
        output_dir=output_dir,
        device="cpu",
        model_factory=TinySegNet,
    )


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.integration
def test_interrupted_resume_matches_uninterrupted_six_step_trajectory(tmp_path: Path) -> None:
    if not Path("/root/LCRSeg").is_dir():
        pytest.skip("frozen LCRSeg data are available only on the experiment node")
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"

    uninterrupted = _runner(uninterrupted_dir)
    assert uninterrupted.run(stop_after_global_step=6)["status"] == "INTERRUPTED"

    first_leg = _runner(interrupted_dir)
    first_result = first_leg.run(stop_after_global_step=3)
    assert first_result["status"] == "INTERRUPTED"
    resumed = _runner(interrupted_dir)
    second_result = resumed.run(resume_path=first_result["checkpoint"], stop_after_global_step=6)
    assert second_result["status"] == "INTERRUPTED"

    reference = torch.load(uninterrupted_dir / "last.pt", map_location="cpu", weights_only=False)
    candidate = torch.load(interrupted_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in ("student", "ema_teacher", "optimizer", "scheduler", "gas_state", "stage_state", "sampler_state"):
        _assert_nested_equal(reference[key], candidate[key])
