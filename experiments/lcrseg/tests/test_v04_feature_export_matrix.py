from pathlib import Path

from scripts.run_v0_4_feature_export_matrix import _jobs


def test_v04_feature_export_matrix_has_all_r0_r1_seed_site_pairs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    from lcrseg.analysis.v0_4 import RUN_NAMES, SITE_ORDER

    for run_name in RUN_NAMES.values():
        for site_index, site in enumerate(SITE_ORDER):
            path = root / "runs" / run_name / f"checkpoint_final_site{site_index}_{site}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    jobs = _jobs(root, tmp_path / "features")
    assert len(jobs) == 54
    assert len({str(job["output"]) for job in jobs}) == 54
    assert {int(job["seed"]) for job in jobs} == {0, 1, 2}
    assert {str(job["variant"]) for job in jobs} == {"R0", "R1"}
