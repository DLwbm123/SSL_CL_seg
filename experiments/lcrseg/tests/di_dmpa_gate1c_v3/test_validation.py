"""Fresh cache observations preserve numerical values and reject contract drift."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b, execution as e
from di_dmpa_gate1c_v3 import PROTOCOL
from di_dmpa_gate1c_v3.binding import BUDGET, INTEGRATION_IDS, validate_contract
from di_dmpa_gate1c_v3.validation import capture, validate_raw


def test_new_contract_only_rebinds_checkpoint_identity():
    root = Path(__file__).resolve().parents[2]
    old = b.read_json(root / "docs/di_dmpa_jascl/DI_DMPA_GATE1C_V2_PREREGISTRATION.json")
    p = copy.deepcopy(old)
    p.update(protocol=PROTOCOL, budget=BUDGET, integration_pair_ids=INTEGRATION_IDS,
             input_contract_version="v3", diagnostic_precision="float64_shadow", allowed_gpus=[4, 5, 6, 7])
    for cp in p["immutable_baseline"]["checkpoint_inputs"]:
        cp["sha256"] = b.H(["synthetic-new-checkpoint", cp["checkpoint_id"]])
    for pair in p["gradient_diagnostic"]["batch_pairs"]:
        pair["checkpoint_sha256"] = b.checkpoint(p, pair["seed"], pair["stage_index"])["sha256"]
    p["fixed_batch_pairs_sha256"] = b.H(p["gradient_diagnostic"]["batch_pairs"])
    validate_contract(p, old)
    changed = copy.deepcopy(p)
    changed["gradient_diagnostic"]["batch_pairs"][0]["teacher_draw_seeds"][0] += 1
    changed["fixed_batch_pairs_sha256"] = b.H(changed["gradient_diagnostic"]["batch_pairs"])
    with pytest.raises(b.ProtocolError, match="fixed pairs/seeds"):
        validate_contract(changed, old)
    changed = copy.deepcopy(p)
    changed["primary"]["historical_transform"] = "learned"
    with pytest.raises(b.ProtocolError, match="frozen numerical"):
        validate_contract(changed, old)
    changed = copy.deepcopy(p)
    changed["legacy_prototype_reconstruction"] = {}
    with pytest.raises(b.ProtocolError, match="reconstruction"):
        validate_contract(changed, old)


def test_capture_preserves_native_scores_pas_nulls_and_raw_values(tmp_path):
    torch.manual_seed(7)
    sl, tl = (torch.randn(1, 3, 3, 4) for _ in range(2))
    sf, tf = (torch.rand(1, 16, 3, 4) for _ in range(2))
    tf[:, :, 0, 0] = 0
    legacy = torch.nn.functional.normalize(torch.rand(3, 16), dim=1)
    raw = np.random.default_rng(5).uniform(0.01, 1, (3, 2, 16))
    current = raw / np.linalg.norm(raw, axis=2, keepdims=True)
    history = current.copy()
    expected = e.build(sl, sf, tl, tf, legacy, current, history)
    rng = torch.get_rng_state().clone()
    with capture(tmp_path, maximum_forwards=0) as count:
        actual = e.build(sl, sf, tl, tf, legacy, current, history)
        desc = e.save_arrays(tmp_path / "validation_cache/seed0_stage1/synthetic.npz", {k: actual[k] for k in e.CACHE_FIELDS})
    assert count == dict(native_forwards=0, cases=1, original_PAS_calls=2)
    assert torch.equal(rng, torch.get_rng_state())
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    case = dict(arrays=desc, **{name + "_sha256": b.tensor_hash(value) for name, value in
        (("student_logits", sl), ("teacher_logits", tl), ("student_features", sf), ("teacher_features", tf))})
    assert validate_raw(case, 1, height=3, width=4)["direct_PAS_parity"]
    arrays = b.read_arrays(desc["raw_values"])
    assert arrays["null_mask"].sum() == 1 and arrays["pixel_yx"].shape == (12, 2)
    arrays["teacher_features"][0, 0, 0, 1] += 1
    np.savez_compressed(desc["raw_values"]["path"], **arrays)
    with pytest.raises(b.ProtocolError, match="SHA mismatch"):
        validate_raw(case, 1, height=3, width=4)


def test_full_size_synthetic_unit_preserves_original_forward_and_guard(tmp_path, monkeypatch):
    from tests.di_dmpa_gate1c_v2.test_core import Tiny
    from di_dmpa_gate1c_v3 import validation

    torch.manual_seed(31)
    model = Tiny().eval().requires_grad_(False)
    image = torch.rand(1, 3, 384, 384)
    legacy = torch.nn.functional.normalize(torch.rand(3, 16), dim=1)
    current = np.repeat(np.eye(16)[:3, None], 2, axis=1)
    checkpoint = tmp_path / "synthetic_checkpoint.bin"
    checkpoint.write_bytes(b"synthetic-only; no real checkpoint or data")
    cp = dict(checkpoint_id="B0/seed0/stage0", path=str(checkpoint), sha256=b.sha256(checkpoint))
    row = dict(case_id="synthetic", image_sha256=b.tensor_hash(image))
    p = dict(validation=dict(plans=[dict(seed=0, stage_index=0, cases=[dict(case_id="synthetic", teacher_draw0_seed=43, student_seed=47)])]))
    monkeypatch.setattr(e, "checkpoint", lambda *a: cp)
    monkeypatch.setattr(e, "records", lambda *a: [row])
    monkeypatch.setattr(e, "image_only", lambda r: r)
    monkeypatch.setattr(e, "_images", lambda *a: image.clone())
    monkeypatch.setattr(e, "load_b0", lambda *a: ({k: copy.deepcopy(model) for k in ("student", "ema_teacher")}, legacy.clone()))
    monkeypatch.setattr(e, "banks", lambda *a: (current.copy(), np.empty((3, 0, 16))))
    monkeypatch.setattr(e, "bank_identity", lambda *a: "synthetic K2 only")
    original = e.validation_unit(tmp_path, tmp_path, p, {}, {}, 0, 0, tmp_path / "original", "cpu")
    expected_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(validation, "LCRSegUNet2DJASCL", Tiny)
    with capture(tmp_path / "captured", maximum_forwards=2) as count:
        actual = e.validation_unit(tmp_path, tmp_path, p, {}, {}, 0, 0, tmp_path / "captured", "cpu")
        with pytest.raises(b.ProtocolError, match="budget/dtype exceeded"):
            model(image, stochastic_classifier=True)
    assert count == dict(native_forwards=2, cases=1, original_PAS_calls=2)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    a, z = actual["cases"][0], original["cases"][0]
    for key, value in b.read_arrays(z["arrays"]).items():
        np.testing.assert_array_equal(b.read_arrays(a["arrays"])[key], value)
    assert {k: v for k, v in a.items() if k != "arrays"} == {k: v for k, v in z.items() if k != "arrays"}
    assert validate_raw(a, 0)["native_values_verified"]
    guard = b.read_json(tmp_path / "captured/validation_models/seed0_stage0/immutability/B0_seed0_stage0.json")
    assert guard["status"] == "PASS" and guard["extraction_completed"] and guard["before"] == guard["after"]
