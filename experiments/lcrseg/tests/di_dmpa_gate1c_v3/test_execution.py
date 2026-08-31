"""Orchestration rejects missing, extra, misbound and unsuccessful evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from di_dmpa_gate1c_v2 import binding as b, execution as e, precision_pilot as pilot
from di_dmpa_gate1c_v3 import execution as v3
from di_dmpa_gate1c_v3.binding import INTEGRATION_IDS


@pytest.mark.parametrize("phase", pilot.PHASES)
@pytest.mark.parametrize("bad", [None, "count", "exit", "gpu", "guard", "missing", "extra"])
def test_three_pair_phase_requires_complete_owned_evidence(tmp_path, monkeypatch, phase, bad):
    root = Path(__file__).resolve().parents[2]
    p = b.read_json(root / "docs/di_dmpa_jascl/DI_DMPA_GATE1C_V2_PREREGISTRATION.json")
    p["integration_pair_ids"] = INTEGRATION_IDS
    selected = v3.pairs(p, "integration")
    meta = dict(controller_pid=99, purpose="synthetic_orchestration_only")
    args = SimpleNamespace(output=tmp_path, scope="integration")
    values = dict(alignment_rows={"draw0": 56, "noise": 448, "posterior": 56, "poe": 112}[phase],
                  global_comparisons={"draw0": 8, "noise": 64, "posterior": 8, "poe": 16}[phase],
                  class_components={"draw0": 168, "noise": 0, "posterior": 0, "poe": 42}[phase], supervised_global_comparisons=1)
    checked = []
    def numeric(result, pair, requested, context):
        assert result["metadata"] == context and result["pair"] == pair and result["phase"] == requested
        checked.append(pair["batch_id"])
        return values
    monkeypatch.setattr(pilot, "validate_result", numeric)
    for shard, pair in enumerate(selected):
        start = dict(metadata=meta, phase=phase, shard=shard, pid=1000 + shard, parent_pid=99, physical_gpu=4 + shard)
        if bad == "gpu" and shard == 0:
            start["physical_gpu"] = 0
        b.write_json(tmp_path / f"WORKER_{phase}_{shard}_START.json", start)
        counts = dict(zip(pilot.COUNT_KEYS, pilot.COUNTS[phase]))
        if bad == "count" and shard == 0:
            counts["native_forwards"] += 1
        parity = [dict(exact_R1_parity=True, pixels=294912, batch_id=pair["batch_id"])
                  for _ in range({"draw0": 1, "noise": 8, "posterior": 1, "poe": 0}[phase])]
        b.write_json(tmp_path / f"WORKER_{phase}_{shard}.json", dict(start, status="PASS", counts=counts,
            completed_units=[pair["batch_id"]], PAS_parity=parity, all_checkpoints_unchanged=True))
        (tmp_path / f"WORKER_{phase}_{shard}.log").write_text("synthetic\n")
        directory = tmp_path / "probes" / phase / e.pair_name(pair)
        if bad != "missing" or shard != 0:
            result = dict(pair=pair, metadata=meta, phase=phase)
            for key in ("student_logits_sha256", "student_features_sha256", "labeled_logits_sha256", "teacher_features_sha256",
                        "teacher_probability_sha256", "R1_validity_sha256", "native_supervised_gradient_sha256", "supervised_gradient_sha256"):
                result[key] = "synthetic"
            result.update(gradient_hashes={}, student_draw_replay={})
            b.write_json(directory / "result.json", result)
        iso = dict.fromkeys(("teacher_gradients", "prototype_gradients", "history_bank_gradients", "student_parameter_grad_fields"), "None")
        iso.update(metadata=meta, legacy_prototypes_unchanged=True, current_history_banks_unchanged=True,
                   optimizer_constructed=False, backward_called=False)
        b.write_json(directory / "isolation.json", iso)
        before = dict(student="S", ema_teacher="T", gradient_student="D")
        guard = dict(metadata=meta, bitwise_unchanged=True, extraction_completed=True, before=before,
                     after=copy.deepcopy(before), status="PASS", checkpoint_id=pair["checkpoint_id"],
                     checkpoint_sha256_before=pair["checkpoint_sha256"], checkpoint_sha256_after=pair["checkpoint_sha256"])
        if bad == "guard" and shard == 0:
            guard["after"]["student"] = "changed"
        b.write_json(tmp_path / "probe_models" / phase / e.pair_name(pair) / "immutability" /
                     f"B0_seed{pair['seed']}_stage{pair['stage_index']}.json", guard)
    b.write_json(tmp_path / f"PROCESS_EXIT_{phase}.json", dict(actual_child_exit_codes=[0, 1, 0] if bad == "exit" else [0, 0, 0],
        parent_pid=99, worker_pids=[1000, 1001, 1002]))
    if bad == "extra":
        (tmp_path / "probes" / phase / "unregistered_pair").mkdir()
    if bad:
        with pytest.raises((b.ProtocolError, FileNotFoundError)):
            v3.probe_barrier(args, p, meta, phase)
        assert not (tmp_path / f"PHASE_{phase}.json").exists()
    else:
        v3.probe_barrier(args, p, meta, phase)
        assert checked == [q["batch_id"] for q in selected]
        receipt = b.read_json(tmp_path / f"PHASE_{phase}.json")
        assert receipt["coverage"] == {k: 3 * v for k, v in values.items()}
        assert (tmp_path / f"PHASE_{phase}_MANIFEST.json").is_file()


def test_future_r2_candidate_does_not_rescue_r3_scientific_status():
    status = dict(reliability_status="FAIL_IDENTITY_HISTORY_RELIABILITY_NOT_SUPPORTED",
                  R0_R1_R2_R3_results={"R2": {"pixel_normalized": {"all_pass": True}}})
    before = copy.deepcopy(status)
    assert v3.reduced_candidate(status) == "CURRENT_ONLY_MPA_JASCL" and status == before
    status["R0_R1_R2_R3_results"]["R2"]["pixel_normalized"]["all_pass"] = False
    assert v3.reduced_candidate(status) == "NONE"
