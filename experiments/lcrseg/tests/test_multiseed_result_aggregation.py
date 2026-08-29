from lcrseg.analysis.v0_3 import aggregate_paired_seed_metrics


def test_multiseed_result_aggregation() -> None:
    rows = []
    for seed, delta in ((0, 0.01), (1, -0.01), (2, 0.02)):
        for variant, offset in (("R0", 0.0), ("R1", delta)):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "final_average_dice": 0.6 + offset,
                    "bwt": -0.1 + offset,
                    "incoming_dice": 0.7 + offset,
                    "previous_site_dice": 0.65 + offset,
                }
            )
    result = aggregate_paired_seed_metrics(rows)
    assert len(result["paired"]) == 3
    assert abs(result["summary"]["metrics"]["final_average_dice"]["mean"] - (0.02 / 3.0)) < 1.0e-12
    assert result["summary"]["metrics"]["final_average_dice"]["positive_direction_count"] == 2

