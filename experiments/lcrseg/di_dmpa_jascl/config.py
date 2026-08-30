from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import METHOD_REGISTERED, UPSTREAM_COMMIT


ALL_NEW_MODULE_SWITCHES = (
    "use_multi_prototype",
    "use_domain_indexed_bank",
    "use_transport",
    "use_soft_proto_fusion",
    "use_history_gate",
    "use_multi_proto_loss",
    "use_proto_inference",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate0_config(config: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if resolved.get("name") != "gate0_repaired":
        raise ValueError("only the gate0_repaired runner is authorized")
    if resolved.get("semantic_version") != 2:
        raise ValueError("v1 is archived; formal execution requires semantic_version=2")
    if resolved.get("objective_name") != "probability_mse_on_joint_pas_validity":
        raise ValueError("only the reviewed PAS probability objective is authorized")
    if resolved.get("evaluation_classifier") != "posterior_mean":
        raise ValueError("formal evaluation must use posterior-mean classifier")
    if resolved.get("benchmark") != "fundus":
        raise ValueError("only Fundus is authorized for Gate 0 v2")
    if resolved.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("gate0_repaired must remain pinned to official commit 3c93ca7")
    if resolved.get("method_registered") is not False or METHOD_REGISTERED:
        raise RuntimeError("DI-DMPA method registration is forbidden before Gate 0 passes")

    method = resolved.setdefault("method", {})
    for switch in ALL_NEW_MODULE_SWITCHES:
        if method.get(switch) is not False:
            raise ValueError(f"{switch} must remain false during Gate 0")
    if method.get("use_constant_patch_classifier_regularization") is not False:
        raise ValueError("constant-patch classifier regularization is forbidden during Gate 0")
    if method.get("constant_patch_regularization_in_gas") is not False:
        raise ValueError("constant-patch regularization must never enter GAS")

    benchmark_name = resolved["benchmark"]
    benchmarks = protocol.get("benchmarks", {})
    if benchmark_name not in benchmarks:
        raise ValueError(f"benchmark is absent from DOMAIN_PROTOCOL.yaml: {benchmark_name}")
    benchmark = benchmarks[benchmark_name]
    configured_order = resolved["data"]["domain_order"]
    if configured_order != benchmark["domain_order"]:
        raise ValueError("configured domain order differs from the frozen protocol")
    if int(resolved["model"]["num_classes"]) != int(benchmark["class_count"]):
        raise ValueError("configured class count differs from the frozen protocol")
    if int(resolved["model"]["input_channels"]) != int(benchmark["input_channels"]):
        raise ValueError("configured input channels differ from the frozen protocol")
    model = resolved["model"]
    expected_model = {
        "implementation": "lcrseg_unet2d_jascl_3x3_stochastic_head",
        "base_channels": 16,
        "normalization": "groupnorm",
        "classifier_kernel_size": 3,
        "classifier_padding": 1,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"Gate 0 model contract requires {key}={expected!r}")
    reference_root = model.get("reference_root")
    if not isinstance(reference_root, str) or not reference_root.strip():
        raise ValueError("Gate 0 requires an explicit official JASCL reference_root")
    if resolved["data"].get("test_gt_policy") != "evaluator_only":
        raise ValueError("final test GT must be evaluator-only")

    expected_training = {
        "optimizer": "adam",
        "lr": 1.0e-3,
        "weight_decay": 4.0e-5,
        "epochs_per_stage": 100,
        "scheduler": "polynomial",
        "scheduler_power": 0.9,
        "teacher_ema_alpha": 0.99,
        "teacher_mode": "upstream_eval",
        "prototype_start_epoch": 25,
        "pseudo_label_interval_epochs": 5,
        "confidence_threshold": 0.7,
        "similarity_threshold": 0.7,
        "labeled_batch_size": 2,
        "unlabeled_batch_size": 2,
        "ignore_label": 255,
    }
    training = resolved["training"]
    if training.get("lambda_u") not in (0.0, 0.5):
        raise ValueError("only compute/RNG-matched C0=0.0 and B0=0.5 are authorized")
    if "unsupervised_consistency_weight" in training:
        raise ValueError("legacy loss-weight field is forbidden in v2")
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Gate 0 may not change upstream {key}: {training.get(key)!r} != {expected!r}")
    if training.get("weak_strong_augmentation") is not False:
        raise ValueError("weak/strong augmentation is not authorized for gate0_repaired")
    return resolved


def resolved_config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
