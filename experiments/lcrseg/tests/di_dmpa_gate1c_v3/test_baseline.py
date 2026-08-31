"""The evidence adapter must leave the original trajectory and RNG unchanged."""
import os
from pathlib import Path

import pytest
import torch

from di_dmpa_jascl.runner import Gate0RepairedRunner
from di_dmpa_gate1c_v3.baseline import RegeneratedB0Runner, state_digest, verify_payload
from di_dmpa_gate1c_v3.baseline_execution import audit_seed
from di_dmpa_gate1c_v3.durable import write_new
from tests.gate0.test_resume_v2 import synthetic_bundle, TinySegNet, ROOT


def test_live_pas_capture_resume_and_early_best_sealing(tmp_path):
    torch.set_num_threads(1)
    config, protocol = synthetic_bundle(tmp_path / "data")
    actual = os.environ.get("V3_TEST_ACTUAL_MODEL") == "1"
    device = os.environ.get("V3_TEST_DEVICE", "cpu") if actual else "cpu"

    def runner(output, regenerated):
        cls = RegeneratedB0Runner if regenerated else Gate0RepairedRunner
        kwargs = dict(provenance={"purpose": "synthetic_equivalence"}) if regenerated else {}
        result = cls(repo_root=ROOT, config=config, protocol=protocol, seed=0,
                     output_dir=tmp_path / output, device=device,
                     model_factory=None if actual else TinySegNet, **kwargs)
        # Test-only selection fixture: force the pre-PAS checkpoint to stay best.
        evaluate = result.evaluate_domain
        def fixed_val(domain, role):
            report = evaluate(domain, role)
            if role == "val":
                report["mean_iou"] = 0.5
            return report
        result.evaluate_domain = fixed_val
        return result

    reference = runner("reference", False)
    expected_counts = None
    prototype_batches = reference._prototype_batches
    def observe_batches(*args):
        nonlocal expected_counts
        batches = prototype_batches(*args)
        expected_counts = torch.tensor([sum(int((batch["label"] == c).sum()) for batch in batches)
                                        for c in range(3)])
        return batches
    reference._prototype_batches = observe_batches
    reference.run(stop_at_event="before_stage_transition")
    expected = torch.load(tmp_path / "reference/last.pt", map_location="cpu", weights_only=False)
    candidate = runner("candidate", True)
    first = candidate.run(stop_after_global_step=53)
    middle = torch.load(first["checkpoint"], map_location="cpu", weights_only=False)
    assert middle["sampler_state"]["phase"] == "unlabeled"
    assert middle["v3"]["legacy_pas_capture"]["additional_model_forwards"] == 0
    candidate = runner("candidate", True)
    candidate.run(resume_path=first["checkpoint"], stop_at_event="before_stage_transition")
    observed = torch.load(tmp_path / "candidate/last.pt", map_location="cpu", weights_only=False)
    for key in expected:
        assert state_digest(expected[key]) == state_digest(observed[key]), key
    best_path = tmp_path / "candidate/stage_0_REFUGE/best.pt"
    early_best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert early_best["stage_state"]["epoch"] == 1 and early_best["prototypes"] is None
    captured = torch.load(tmp_path / "candidate/legacy_pas_captures/stage0.pt", weights_only=False)
    candidate.run(resume_path=tmp_path / "candidate/last.pt")
    sealed = torch.load(best_path, map_location="cpu", weights_only=False)
    verify_payload(sealed)
    for key in expected:
        if key != "prototypes":
            assert state_digest(early_best[key]) == state_digest(sealed[key]), key
    assert state_digest(captured["bank"]) == state_digest(sealed["prototypes"])
    assert torch.equal(sealed["prototype_counts"], expected_counts)
    assert 0 < int(expected_counts.sum()) <= 3 * 16 * 16
    assert sealed["v3"]["bank_captured_after_selected_epoch"]
    independent = runner("independent_load", True)
    independent._load_best_models(best_path)
    assert state_digest(independent.wrapper.student.state_dict()) == state_digest(sealed["student"])
    assert state_digest(independent.wrapper.teacher.state_dict()) == state_digest(sealed["ema_teacher"])
    with pytest.raises(RuntimeError, match="resume requires last.pt"):
        independent.resume(best_path)
    sealed["prototype_counts"][0] += 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_payload(sealed)
    audit_root = tmp_path / "completed_seed_audit"
    audit_root.mkdir()
    report = audit_seed(candidate, audit_root, config, protocol, {"purpose": "synthetic_equivalence"})
    assert report["status"] == "PASS_SEED_ENGINEERING" and report["observed_steps"] == 690
    if os.environ.get("V3_TEST_REPORT_DIR"):
        out = Path(os.environ["V3_TEST_REPORT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        write_new(out / "V3_BASELINE_ADAPTER_EQUIVALENCE.json", dict(
            status="PASS", model="production_UNet_JASCL" if actual else "TinySegNet",
            device=device, data="synthetic_hashed_HDF5", original_fields_bitwise_equal=list(expected),
            interrupted_global_step=53, observed_stage_epochs=100, forced_best_epoch=1,
            genuine_pas_capture_epoch=25, no_extra_stochastic_draws=True,
            full_classifier_independent_load=True, tamper_rejected=True))


def test_two_case_overfit_and_fixed_batch_golden(tmp_path):
    """Engineering fixture only; no real C0/B0 run or hidden/test labels."""
    from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, update_gas_from_supervised_gradient
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(17)
    device = os.environ.get("V3_TEST_DEVICE", "cpu")
    model = build_lcrseg_unet_jascl_model(ROOT / "third_party/JASCL_REFERENCE",
        upstream_path="Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT", input_channels=3, num_classes=3).to(device)
    label = (torch.arange(32).view(1, 1, 32).expand(2, 32, 32) // 11).clamp_max(2).to(device)
    image = F.one_hot(label, 3).permute(0, 3, 1, 2).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00004)
    model.eval()
    logits, features = model(image, stochastic_classifier=False)
    repeated, repeated_features = model(image, stochastic_classifier=False)
    assert torch.equal(logits, repeated) and torch.equal(features, repeated_features)
    initial = float(F.cross_entropy(logits, label).detach())
    curve = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(image, stochastic_classifier=False)
        loss = F.cross_entropy(logits, label)
        assert torch.isfinite(loss)
        loss.backward()
        update_gas_from_supervised_gradient(model)
        optimizer.step()
        curve.append(float(loss.detach()))
    with torch.no_grad():
        final_logits, _ = model(image, stochastic_classifier=False)
        final = float(F.cross_entropy(final_logits, label))
        accuracy = float((final_logits.argmax(1) == label).float().mean())
    assert final < 0.2 * initial and accuracy >= 0.98
    write_new(tmp_path / "TWO_CASE_AND_GOLDEN.json", dict(status="PASS", data="two_synthetic_32x32_cases",
        model="production_UNet_JASCL", device=device, steps=100, initial_ce=initial, final_ce=final,
        accuracy=accuracy, deterministic_forward_bitwise_equal=True, backward_and_gas_update=True,
        image_sha256=state_digest(image), label_sha256=state_digest(label), loss_curve=curve))
