import pytest

from scripts.run_v0_4_crossed_pilot import _validate_cross, pilot_run_name


def test_v04_crossed_pilot_registry_is_exact_and_diagnostic_only() -> None:
    valid = []
    for rng in (10, 11, 12):
        _validate_cross("O", 0, rng)
        valid.append(("O", 0, rng))
    for split_seed in (0, 1, 2):
        _validate_cross("S", split_seed, 20)
        valid.append(("S", split_seed, 20))
    names = {
        pilot_run_name(family, split_seed, rng, variant)
        for family, split_seed, rng in valid
        for variant in ("R0", "R1")
    }
    assert len(names) == 12
    assert all("diagnostic_only" in name for name in names)
    assert all(name.endswith("_rimone1000") for name in names)


@pytest.mark.parametrize(
    ("family", "split_seed", "rng"),
    [("O", 1, 10), ("O", 0, 20), ("S", 0, 10), ("S", 3, 20)],
)
def test_v04_crossed_pilot_registry_rejects_unregistered_crosses(
    family: str, split_seed: int, rng: int
) -> None:
    with pytest.raises(ValueError):
        _validate_cross(family, split_seed, rng)
