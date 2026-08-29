import torch

from lcrseg.methods.lcrseg_v0_1 import LCRSegV01Method
from lcrseg.methods.lcrseg_v0_2a import LCRSegV02AMethod
from lcrseg.models import UNet2D
from tests.v0_2a_test_utils import batches, previous_checkpoint, routing_config


def test_legacy_continuous_golden_equivalence(tmp_path) -> None:
    parent = previous_checkpoint(tmp_path)
    labeled, unlabeled = batches()
    legacy = LCRSegV01Method(UNet2D(3, 3), config=routing_config(use_learnability=True, use_compatibility=False))
    amended = LCRSegV02AMethod(UNet2D(3, 3), config=routing_config(variant_id="R0", assimilation_mode="legacy_continuous_v01", consolidation_mode="uniform_relation"))
    legacy.begin_site("B", parent, 4)
    amended.begin_site("B", parent, 4)
    first = legacy.training_step(labeled, unlabeled, global_step=1, site_step=0)
    second = amended.training_step(labeled, unlabeled, global_step=1, site_step=0)
    for key in ("loss_sup", "loss_assim", "loss_relation"):
        assert torch.allclose(first.losses[key], second.losses[key], atol=1.0e-7, rtol=0.0)
    assert torch.allclose(first.total_loss, second.total_loss, atol=1.0e-7, rtol=0.0)
    assert first.maps is not None and second.maps is not None
    for key in ("pseudo_labels", "pseudo_valid", "learnability", "current_relation_probability", "old_relation_probability"):
        assert torch.equal(first.maps[key], second.maps[key])
    for left, right in zip(legacy._pending_anchor_updates, amended._pending_anchor_updates, strict=True):
        for first_value, second_value in zip(left[:3], right[:3], strict=True):
            assert torch.equal(first_value, second_value)
