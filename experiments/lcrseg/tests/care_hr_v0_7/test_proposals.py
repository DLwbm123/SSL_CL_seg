import inspect

import numpy as np

from care_hr_v0_7.proposals import generate_proposals


def test_proposal_api_has_no_hidden_truth_or_source_fields():
    names = {name.lower() for name in inspect.signature(generate_proposals).parameters}
    assert names == {"current_hard", "historical_hard", "minimum_pixels", "maximum_proposals"}


def test_add_and_remove_definitions_are_exact():
    current = np.zeros((8, 8), dtype=int)
    historical = current.copy()
    historical[:2, :4] = 1
    current[6:, 4:] = 2
    proposals = generate_proposals(current, historical)
    assert [(p.target_class, p.direction, p.area) for p in proposals] == [(1, "add", 8), (2, "remove", 8)]


def test_components_are_four_connected():
    current = np.zeros((3, 3), dtype=int)
    historical = current.copy()
    historical[0, 0] = historical[1, 1] = 1
    proposals = generate_proposals(current, historical, minimum_pixels=1)
    assert len(proposals) == 2 and all(p.area == 1 for p in proposals)


def test_sort_is_area_then_centroid_then_class_direction():
    current = np.zeros((12, 12), dtype=int)
    historical = current.copy()
    historical[0:2, 0:5] = 1
    historical[8:10, 8:12] = 2
    proposals = generate_proposals(current, historical)
    assert [p.area for p in proposals] == [10, 8]


def test_ids_and_masks_repeat_exactly():
    current = np.zeros((8, 8), dtype=int)
    historical = current.copy(); historical[:2, :4] = 1
    left = generate_proposals(current, historical)
    right = generate_proposals(current, historical)
    assert [p.proposal_id for p in left] == [p.proposal_id for p in right]
    assert all(np.array_equal(a.mask, b.mask) for a, b in zip(left, right))


def test_overlap_resolution_uses_fixed_sorted_priority():
    current = np.zeros((4, 4), dtype=int); current[:2, :] = 1
    historical = np.zeros((4, 4), dtype=int); historical[:2, :] = 2
    proposals = generate_proposals(current, historical)
    assert [(p.target_class, p.direction) for p in proposals] == [(1, "remove")]


def test_components_below_eight_pixels_are_removed():
    current = np.zeros((4, 4), dtype=int)
    historical = current.copy(); historical[0, :4] = 1
    assert generate_proposals(current, historical) == ()


def test_maximum_twelve_proposals_per_case():
    current = np.zeros((40, 40), dtype=int)
    historical = current.copy()
    for index in range(13):
        row, col = divmod(index, 4)
        historical[row * 3:row * 3 + 2, col * 9:col * 9 + 4] = 1
    assert len(generate_proposals(current, historical)) == 12
