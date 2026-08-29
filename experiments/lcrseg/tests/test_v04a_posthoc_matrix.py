from pathlib import Path

from lcrseg.analysis.v0_4 import SITE_ORDER, checkpoint_variant
from scripts.run_v0_4a_feature_export_matrix import _jobs


def test_v04a_checkpoint_variant_is_sra() -> None:
    payload = {"config_resolved": {"method": {"variant_id": "SRA"}}, "method_name": "lcrseg_v0_4a"}
    assert checkpoint_variant(payload) == "SRA"


def test_v04a_feature_export_matrix_has_all_seed_checkpoint_evaluation_pairs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for seed in (0, 1, 2):
        run_name = f"fundus_seed{seed}_lcrseg_v0_4a_sra_uniform_full200e"
        for site_index, site in enumerate(SITE_ORDER):
            path = root / "runs" / run_name / f"checkpoint_final_site{site_index}_{site}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    jobs = _jobs(root, tmp_path / "features")
    assert len(jobs) == 27
    assert len({str(job["output"]) for job in jobs}) == 27
    assert {int(job["seed"]) for job in jobs} == {0, 1, 2}
