import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import rc_shor_v0_5_posthoc_audit as audit


def rows(order):
    return [{"policy": "C3", "row_index": index, "route": index % 3} for index in order]


def test_fold_order_route_is_recovered_only_by_row_index_sort():
    shuffled = rows([2, 0, 3, 1])
    assert [row["route"] for row in audit.aligned_rows(shuffled, "C3", 4)] == [0, 1, 2, 0]
    with pytest.raises(audit.AuditBlocked, match="global row_index order"):
        audit.aligned_rows(shuffled, "C3", 4, sort=False)


def test_metric_difference_equals_route_gain_and_current_drop_is_zero():
    seeds = np.repeat(np.arange(3), 6)
    domains = np.tile(np.repeat(np.arange(3), 2), 3)
    utility = np.zeros((18, 2)); utility[:, 0] = 0.2
    classes = np.zeros((18, 2, 2))
    route = np.zeros(18, dtype=int); route[domains == 2] = 2
    summary = audit.route_summary(route, utility, classes, seeds, domains)
    c0 = np.full(18, 0.5)
    c3 = c0 + np.where(domains < 2, 0.2, 0.0)
    assert summary["three_domain_gain"] == pytest.approx(
        audit.balanced_mean(c3, seeds, domains) - audit.balanced_mean(c0, seeds, domains))
    assert summary["current_domain_drop"] == 0.0


def test_rho_changes_candidate_route_when_consensus_crosses_threshold():
    lcb = np.zeros((100, 1, 2)); lcb[:85, 0, 0] = 1; lcb[85:, 0, 1] = 1
    feasible = np.ones(100, dtype=bool); ood = np.zeros((1, 2), dtype=bool)
    assert audit.ungated_ensemble_route(lcb, feasible, 0.0, 0.80, ood).tolist() == [0]
    assert audit.ungated_ensemble_route(lcb, feasible, 0.0, 0.90, ood).tolist() == [2]


def test_final_and_realization_share_route_function():
    source = inspect.getsource(audit.full_policy_route)
    assert source.count("route_decisions(") == 2


def test_per_unit_counts_are_not_global_intersection():
    result = audit.feasibility_report([
        np.asarray([True, True, False]), np.asarray([False, True, True])])
    assert result["per_fold"] == [2, 2]
    assert result["global_replicate_index_intersection"] == 1


def test_zero_route_precision_is_null_with_explicit_counts():
    seeds = np.repeat(np.arange(3), 3); domains = np.tile(np.arange(3), 3)
    result = audit.route_summary(np.full(9, 2), np.zeros((9, 2)), np.zeros((9, 2, 2)), seeds, domains)
    assert result["route_precision"] is None
    assert result["route_precision_numerator"] == result["route_precision_denominator"] == 0


def test_private_manifest_or_sealed_file_tamper_blocks(tmp_path):
    sealed = tmp_path / "candidate_seals" / "fold0.json"; sealed.parent.mkdir()
    sealed.write_text('{"ok":true}')
    entry = {"path": "candidate_seals/fold0.json", "bytes": sealed.stat().st_size,
             "sha256": audit.sha256_file(sealed)}
    manifest = {"files": 1, "bytes": entry["bytes"], "entries": [entry],
                "content_sha256": audit.canonical_hash([entry])}
    (tmp_path / "RC_SHOR_V0_5_PRIVATE_MANIFEST.json").write_text(json.dumps(manifest))
    bundle = audit.VerifiedBundle(tmp_path); bundle.verify(entry["path"])
    sealed.write_text('{"ok":false}')
    with pytest.raises(audit.AuditBlocked, match="SHA mismatch"):
        bundle.verify(entry["path"])
    manifest["content_sha256"] = "0" * 64
    (tmp_path / "RC_SHOR_V0_5_PRIVATE_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(audit.AuditBlocked, match="content seal"):
        audit.VerifiedBundle(tmp_path)


def test_audit_has_no_model_or_private_data_api_calls():
    tree = ast.parse(Path(audit.__file__).read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert not imports & {"torch", "h5py", "rc_shor_v0_5", "shor_v0_4_test"}
    assert not calls & {"forward", "read_label", "load_state_dict"}
