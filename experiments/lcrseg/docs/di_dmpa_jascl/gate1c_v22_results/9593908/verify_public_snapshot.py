"""Read-only checks of this public snapshot; no data, model, GPU or network use."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
CODE = "1cfd8235293e157afd6b40f0f091ce6bc6df9f9f"
PREREG = "9593908bd36f7f833e385a70b2b772b7a8c84d22"
ORIGINAL_MANIFEST = "1dcbf3199afa0df92a4d5ffcc69940c7b4ef235e7b1dd3d425a31952eff48c0c"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite(value):
    raise ValueError("Non-finite JSON constant: " + value)


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def main():
    if not __debug__:
        raise RuntimeError("Run this assertion-based verifier without Python -O.")
    manifest = read(ROOT / "PUBLIC_COPY_MANIFEST.json")
    rows = manifest["files"]
    names = [row["path"] for row in rows]
    assert len(names) == len(set(names)) == manifest["file_count"]
    objects = list(ROOT.rglob("*"))
    assert not any(p.is_symlink() for p in objects)
    assert all(p.is_file() or p.is_dir() for p in objects)
    files = [p for p in objects if p.is_file()]
    assert {p.relative_to(ROOT).as_posix() for p in files} == set(names) | {"PUBLIC_COPY_MANIFEST.json"}
    for row in rows:
        rel = PurePosixPath(row["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        path = ROOT / rel
        path.resolve().relative_to(ROOT)
        assert path.suffix in {".json", ".md", ".xml", ".sha256", ".py"}
        assert path.stat().st_size == row["public_bytes"] and sha(path) == row["public_sha256"], row["path"]
        if path.suffix == ".json":
            read(path)
    assert sum(row["public_bytes"] for row in rows) == manifest["total_public_bytes"]
    assert sum(row["source_group"] == "integration_archive" for row in rows) == 89
    assert sum(row["source_group"] == "execution_records" for row in rows) == 5
    run = ROOT / "integration_attempt1"
    assert sha(run / "GATE1C_V2_ARTIFACT_MANIFEST.json") == ORIGINAL_MANIFEST
    assert (run / "GATE1C_V2_ARTIFACT_MANIFEST.sha256").read_text().split()[0] == ORIGINAL_MANIFEST
    original = read(run / "GATE1C_V2_ARTIFACT_MANIFEST.json")
    source_hashes = {row["path"]: row["sha256"] for row in original["artifacts"]}
    assert original["file_count"] == 102 and original["total_bytes"] == 126389395
    for row in rows:
        if row["source_group"] == "integration_archive" and row["source_path"] in source_hashes:
            assert row["original_sha256"] == source_hashes[row["source_path"]]
    meta = read(run / "GATE1C_V2_RUN_METADATA.json")
    status = read(run / "GATE1C_V22_STATUS.json")
    numeric = read(run / "NUMERICAL_COMPARISON_AUDIT.json")
    assert meta["diagnostic_code_commit"] == CODE and meta["preregistration_commit"] == PREREG
    assert not any(meta["method_flags"].values())
    assert status["status"] == "PASS_EXACT_CODE_REAL_INTEGRATION"
    assert status["scientific_admission"] is None
    assert status["gate1_overall_status"] == "FAIL_TRANSPORT_NOT_SUPPORTED"
    counts = dict.fromkeys(("native_forwards", "shadow_forwards", "native_autograd", "shadow_autograd"), 0)
    comparison, supervised, components = [], [], []
    alignment_count = 0
    for phase in ("draw0", "noise", "posterior", "poe"):
        barrier = read(run / ("PHASE_" + phase + ".json"))
        assert barrier["status"] == "PASS" and barrier["metadata"] == meta
        assert all(source_hashes[path] == value for path, value in barrier["evidence_sha256"].items())
        for key, value in barrier["counts"].items():
            counts[key] += value
        exits = read(run / ("PROCESS_EXIT_" + phase + ".json"))
        assert exits["exit_codes"] == [0, 0]
        result_paths = sorted((run / "probes" / phase).glob("*/result.json"))
        assert len(result_paths) == 3
        for path in result_paths:
            result = read(path)
            assert result["metadata"] == meta
            comparison.extend(r for r in result["native_precision_comparisons"] if r["block"] == "global")
            supervised.append(result["supervised_precision_comparisons"]["global"])
            components.extend(result["class_contribution"])
            alignment_count += len(result["alignment"])
    assert counts == status["counts"] == numeric["counts"] == dict(native_forwards=51, shadow_forwards=24, native_autograd=276, shadow_autograd=366)
    assert (len(comparison), len(supervised), len(components), alignment_count) == (288, 12, 630, 2016)
    assert all(row["precision_comparable"] for row in comparison)
    for row in comparison + supervised:
        assert (row["native_l2_norm"] == row["reference_l2_norm"] == 0) or (row["relative_l2"] <= .001 and row["cosine"] >= .9999)
    assert all(row["component_sum_pass"] for row in components)
    assert max(r["relative_l2"] for r in comparison) == numeric["maximum_objective_relative_l2"]
    assert min(r["cosine"] for r in comparison) == numeric["minimum_objective_cosine"]
    assert max(r["relative_l2"] for r in supervised) == numeric["maximum_supervised_relative_l2"]
    assert max(r["component_sum_max_abs_error"] for r in components) == numeric["maximum_component_sum_abs_residual"]
    cache = read(run / "CACHE_REUSE_AUDIT.json")
    assert cache["cache_reuse_approved"] and (cache["cases"], cache["pixels"], cache["cache_bytes"]) == (495, 72990720, 4856574421)
    retained = [r for r in cache["references"] if r["retain_private_copy"]]
    assert len(retained) == len({r["path"] for r in retained}) == 567
    assert sum(r["bytes"] for r in retained) == 4932630373
    assert (cache["original_validation_forwards"], cache["new_validation_forwards"]) == (990, 0)
    junit = ET.parse(run / "pytest.xml")
    suites = list(junit.iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 210
    assert all(int(s.attrib[k]) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    props = {p.attrib["name"]: p.attrib["value"] for p in junit.iter("property")}
    assert props["diagnostic_code_commit"] == CODE and props["v22_synthetic_contract"] == PREREG and props["source_clean"] == "true"
    receipt = read(ROOT / "records/INTEGRATION_PROCESS_EXIT.json")
    assert receipt["exit_code"] == 0 and receipt["process_exited"]
    assert receipt["exit_observation"] == "DIRECT_SSH_COMMAND_RETURN" and receipt["manifest_sha256"] == ORIGINAL_MANIFEST
    state = read(ROOT / "MIGRATION_STATE.json")
    interrupted = read(ROOT / "records/FULL_SSH_OBSERVATION_INTERRUPTED_20260831T0430.json")
    assert interrupted["original_ssh_transport"]["exit_code"] == 255 and interrupted["full_controller_exit_code"] is None
    assert state["full"]["controller_exit_code"] is None and state["full"]["completed_new_forwards"] is None
    assert not state["destination_launch_ready"] and not state["method_reproduction_complete"]
    print(json.dumps({"status": "PASS_PUBLIC_SNAPSHOT_HASHES_AND_RECORDED_NUMERICS", "indexed_public_files": len(rows), "indexed_public_bytes": manifest["total_public_bytes"], "integration_new_forwards": 75, "historical_validation_forwards_reused": 990, "full_outcome": "UNKNOWN", "private_payloads_verified": False, "destination_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
