"""The restricted replication cannot admit from background, controls, or lost rows."""
import copy
from pathlib import Path

import pytest
import torch

from di_dmpa_gate1c_v3.k2_replication import adjudicate
from tests.di_dmpa_gate1_v2.test_core import fake_rows


def test_restricted_replication_admission_and_null_integrity():
    rows = [r for r in fake_rows() if r["panel_id"] == "B0-EMA" and r["K"] in (1, 2)]
    result = adjudicate(rows)
    assert result["status"] == "K2_REPLICATION_PASS" and result["selected_K"] == 2
    assert result["null_identity_checks"] == 54 and not result["new_K_selection"]
    assert result["statistics"]["A4_matched_slot_count"] == 180
    assert result["gate1_overall_status"] == "FAIL_TRANSPORT_NOT_SUPPORTED"

    for variant in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
        with pytest.raises(RuntimeError, match="54 unique"):
            adjudicate(variant)
    for field, replacement in (("panel_id", "B0-student"), ("K", 3)):
        variant = copy.deepcopy(rows)
        variant[0][field] = replacement
        with pytest.raises(RuntimeError, match="54 unique"):
            adjudicate(variant)

    variant = copy.deepcopy(rows)
    variant[0]["metrics"]["val"]["full_uid_count_used"] -= 1
    with pytest.raises(RuntimeError, match="lost UID"):
        adjudicate(variant)
    variant = copy.deepcopy(rows)
    variant[1]["metrics"]["val"]["null_mass"] = 0.1
    with pytest.raises(RuntimeError, match="null mass"):
        adjudicate(variant)

    # Background may improve perfectly, but no foreground gain means rejection.
    variant = copy.deepcopy(rows)
    for row in variant:
        if row["class_id"] in (1, 2) and row["K"] == 2:
            for metric in row["metrics"].values():
                metric["R95_null_worst_case"] = 1.0
    result = adjudicate(variant)
    assert result["status"] == "K2_REPLICATION_FAIL" and result["selected_K"] is None
    assert not result["conditions"]["A1"] and not result["conditions"]["A2"]

    # Stability is an independent gate even when all radius conditions pass.
    variant = copy.deepcopy(rows)
    for row in variant:
        if row["class_id"] in (1, 2):
            for draw in row["bootstrap"]:
                draw["matched_cosines"] = [0.84] * row["K"]
    result = adjudicate(variant)
    assert result["status"] == "K2_REPLICATION_FAIL" and not result["conditions"]["A4"]


def test_direct_input_loader_rejects_old_and_changed_checkpoints(tmp_path):
    from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model
    from di_dmpa_gate1.feature_extraction import state_hash
    from di_dmpa_gate1c_v3.baseline import bind_payload, state_digest
    from di_dmpa_gate1c_v3.durable import sha256
    from di_dmpa_gate1c_v3.inputs import load_models

    root = Path(__file__).resolve().parents[2]
    model = build_lcrseg_unet_jascl_model(root / "third_party/JASCL_REFERENCE",
        upstream_path="Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT", input_channels=3, num_classes=3)
    binding = dict(code_commit="synthetic_fixture", config_sha256="synthetic_fixture", seed=0)
    bank, counts, mask = torch.ones(3, 16), torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.bool)
    capture = dict(bank=bank, counts=counts, valid_mask=mask, metadata=dict(binding=binding, stage_index=0,
        domain="REFUGE", bank_sha256=state_digest(bank), counts_sha256=state_digest(counts), valid_mask_sha256=state_digest(mask)))
    payload = dict(student=model.state_dict(), ema_teacher=model.state_dict(), optimizer={}, scheduler={},
        rng_state={}, sampler_state={}, best_metric=0.5, stage_state=dict(stage_index=0), schema_version=2,
        git_commit="synthetic_fixture", config_hash="synthetic_fixture",
        gas_state=dict(grad_update=model.state_dict()["decoder.conv_logit.grad_update"]))
    bind_payload(payload, binding, capture, stage_best=True)
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    cp = dict(baseline="B0", path=str(path), sha256=sha256(path), seed=0, stage_index=0, domain="REFUGE",
        legacy_pas_capture=capture["metadata"])
    models, loaded = load_models(root, cp, device="cpu", sources=("ema_teacher",))
    assert set(models) == {"ema_teacher"} and torch.equal(loaded["prototypes"], bank)
    assert state_hash(models["ema_teacher"].state_dict()) == state_hash(payload["ema_teacher"])
    assert not models["ema_teacher"].training and all(not p.requires_grad and p.grad is None for p in models["ema_teacher"].parameters())
    cp["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="changed regenerated B0"):
        load_models(root, cp, device="cpu")
    payload["v3"]["protocol"] = "old_private_checkpoint"
    torch.save(payload, path)
    cp["sha256"] = sha256(path)
    with pytest.raises(RuntimeError, match="wrong or reconstructed"):
        load_models(root, cp, device="cpu")
