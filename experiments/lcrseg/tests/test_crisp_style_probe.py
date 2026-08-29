from __future__ import annotations

import inspect

import pytest
import torch

from lcrseg.data.transforms import _flip
from lcrseg.representation.style_probe import FrozenStyleProbeTransform, crisp_style_probe_contract


@pytest.mark.parametrize(("dataset", "channels"), [("fundus", 3), ("prostate", 1)])
def test_crisp_style_probe_is_deterministic_same_geometry_and_no_cutout(dataset: str, channels: int) -> None:
    image = torch.linspace(0.0, 1.0, channels * 17 * 19).reshape(channels, 17, 19)
    image_before = image.clone()
    global_state = torch.get_rng_state().clone()
    transform = FrozenStyleProbeTransform(protocol_seed=2)
    first = transform(image=image, dataset=dataset, site_id="current_site", case_id="case_07")
    second = transform(image=image, dataset=dataset, site_id="current_site", case_id="case_07")

    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.equal(image, image_before)
    assert first["geometry_record"] == second["geometry_record"]
    assert first["style_record"] == second["style_record"]
    assert torch.equal(first["clean_image"], second["clean_image"])
    assert torch.equal(first["style_image"], second["style_image"])
    geometry = first["geometry_record"]
    assert torch.equal(
        first["clean_image"],
        _flip(image, hflip=geometry["hflip"], vflip=geometry["vflip"]),
    )
    assert geometry["cutout"] is False and geometry["cutout_box"] is None
    assert bool(first["style_valid_mask"].all())
    assert not torch.equal(first["clean_image"], first["style_image"])


def test_crisp_style_probe_freezes_existing_appearance_contract() -> None:
    contract = crisp_style_probe_contract()
    assert contract["geometry"]["flip_probability"] == 0.5
    assert contract["appearance"]["strong_noise_std"] == 0.03
    assert contract["appearance"]["brightness_delta"] == 0.10
    assert contract["appearance"]["contrast_delta"] == 0.10
    assert contract["appearance"]["operators_in_order"] == ["contrast", "brightness", "gaussian_noise"]
    assert contract["cutout"] is False
    assert contract["new_augmentation"] is False
    assert contract["uses_global_rng"] is False
    assert contract["hidden_gt_usage"] == "none"


def test_crisp_style_probe_has_no_label_or_model_interface() -> None:
    parameters = set(inspect.signature(FrozenStyleProbeTransform.__call__).parameters)
    assert parameters == {"self", "image", "dataset", "site_id", "case_id"}
    result = FrozenStyleProbeTransform(protocol_seed=0)(
        image=torch.ones((3, 8, 8)), dataset="fundus", site_id="REFUGE", case_id="g0001"
    )
    forbidden = {"label", "hidden_label", "pseudo_label", "model", "optimizer"}
    assert not forbidden.intersection(result)


def test_crisp_style_probe_rejects_unknown_dataset_and_wrong_channels() -> None:
    transform = FrozenStyleProbeTransform(protocol_seed=0)
    with pytest.raises(ValueError, match="unsupported"):
        transform(image=torch.ones((3, 8, 8)), dataset="unknown", site_id="s", case_id="c")
    with pytest.raises(ValueError, match="requires 3 channels"):
        transform(image=torch.ones((1, 8, 8)), dataset="fundus", site_id="s", case_id="c")
