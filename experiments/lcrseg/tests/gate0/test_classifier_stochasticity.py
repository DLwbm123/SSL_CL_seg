import ast
from pathlib import Path

import torch

from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model
from .test_official_model_contract import REFERENCE_ROOT, UPSTREAM_PATH


def _model():
    return build_lcrseg_unet_jascl_model(REFERENCE_ROOT, upstream_path=UPSTREAM_PATH,
                                      input_channels=3, num_classes=3).eval()


def test_deterministic_evaluation_is_repeatable():
    model = _model()
    image = torch.randn(1, 3, 16, 16)
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        first, _ = model(image, stochastic_classifier=False)
        middle = torch.get_rng_state().clone()
        second, _ = model(image, stochastic_classifier=False)
    assert torch.equal(first, second)
    assert torch.equal(state, middle) and torch.equal(state, torch.get_rng_state())


def test_stochastic_classifier_draws_different_weights():
    model = _model()
    # Upstream float32 cancellation makes all-zero GAS a degenerate noise scale.
    # Exercise sampling with a nondegenerate, already-trained GAS state; do not repair upstream.
    gas = model.decoder.conv_logit.grad_update
    with torch.no_grad():
        gas.copy_(torch.linspace(0.01, 1.0, gas.numel()).reshape_as(gas))
    image = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        a, _ = model(image, stochastic_classifier=True)
        b, _ = model(image, stochastic_classifier=True)
    assert not torch.equal(a, b)


def test_production_student_teacher_calls_are_explicit():
    root = Path(__file__).resolve().parents[2]
    for relative in ("di_dmpa_jascl/runner.py", "di_dmpa_jascl/modeling.py"):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"student", "teacher"}:
                assert "stochastic_classifier" in {key.arg for key in node.keywords}, (relative, node.lineno)
    tree = ast.parse((root / "di_dmpa_jascl/runner.py").read_text())
    assert not any(isinstance(n, ast.Name) and n.id == "upstream_pas_labels" for n in ast.walk(tree))


def test_control_configs_differ_only_in_lambda():
    from di_dmpa_jascl.config import load_yaml
    root = Path(__file__).resolve().parents[2]
    a = load_yaml(root / "configs/gate0_repaired_v2/fundus_lambda_u0.yaml")
    b = load_yaml(root / "configs/gate0_repaired_v2/fundus_pas_probmse.yaml")
    assert a["training"].pop("lambda_u") == 0.0
    assert b["training"].pop("lambda_u") == 0.5
    assert a == b
