#!/usr/bin/env python3
"""Run one seed of the preregistered CRISP-Seg V0.1 feasibility audit.

Both phases are read-only with respect to models, anchors, data, and optimizer
state.  ``roles`` uses only current-site training-visible records.  ``functional``
uses the frozen role state plus registered update/validation batches and performs
stateless virtual parameter updates; it never calls an optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.func import functional_call
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.analysis.tarc_v0_1 import SITES, TRANSITIONS, checkpoint_path, labeled_loader  # noqa: E402
from lcrseg.analysis.v0_4 import load_frozen_method, stable_seed  # noqa: E402
from lcrseg.common import canonical_json, sha256_bytes, sha256_path, write_csv, write_json  # noqa: E402
from lcrseg.data import (  # noqa: E402
    DeterministicBatcher,
    H5LabeledDataset,
    H5UnlabeledDataset,
    LabeledTransform,
    WeakStrongTransform,
    collate_labeled,
    collate_unlabeled,
)
from lcrseg.engine.metrics import masked_cross_entropy, multiclass_dice_loss  # noqa: E402
from lcrseg.losses.channel_role_consistency import (  # noqa: E402
    invariant_feature_consolidation,
    plastic_feature_consistency,
)
from lcrseg.methods.base import relation_supervision_loss  # noqa: E402
from lcrseg.methods.components.compatibility import zero_compatibility  # noqa: E402
from lcrseg.methods.components.learnability import compute_learnability  # noqa: E402
from lcrseg.methods.components.pseudo_label import build_pseudo_labels  # noqa: E402
from lcrseg.methods.components.routing import relation_consolidation_loss, weighted_mean  # noqa: E402
from lcrseg.methods.lcrseg_v0_1 import _uniform_compatibility  # noqa: E402
from lcrseg.representation.channel_roles import (  # noqa: E402
    FEATURE_LAYERS,
    ChannelRoleState,
    build_channel_role_state,
    case_equal_mean,
    content_relevance_case,
    effective_sample_size,
    hard_rank_roles,
    jaccard,
    quartile_indices,
    spearman_correlation,
    stable_half_assignment,
    style_sensitivity_case,
    uniform_half_roles,
)
from lcrseg.representation.style_probe import FrozenStyleProbeTransform, crisp_style_probe_contract  # noqa: E402
from scripts.audit_aspr_relation_space import _workspace_hash  # noqa: E402


UPDATE_BATCHES = 32
VAL_BATCHES = 16
VIRTUAL_STEP_NORM = 1.0e-3
VARIANTS = ("C0", "C1", "C2", "C3", "C4", "C5")
NUM_CLASSES = 3
FOREGROUND_IDS = (1, 2)


def _environment(physical_gpu: int, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "physical_gpu": int(physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    uuid, name, driver = (part.strip() for part in query.stdout.strip().split(",", maxsplit=2))
    result.update(uuid=uuid, name=name, driver=driver)
    return result


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite CRISP artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    os.replace(temporary, path)


def _assert_frozen_previous(method: Any, old_checkpoint: Path) -> None:
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("CRISP transition lacks a frozen previous model/anchor state")
    old_payload = torch.load(old_checkpoint, map_location="cpu", weights_only=False)
    if _model_hash(method.old_model) != _state_hash(old_payload["current_model_state"]):
        raise RuntimeError("frozen previous model does not match registered old checkpoint")
    for parameter in method.old_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    method.old_model.eval()


def _dataset_records(dataset: Any) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for sample in dataset.samples:
        case_id = str(sample.row["case_id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        values.append((str(sample.row.get("patient_id") or case_id), case_id))
    return values


def _semantic_case_statistics(
    feature: torch.Tensor,
    label: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    resized_label = F.interpolate(label[:, None].float(), size=feature.shape[-2:], mode="nearest")[:, 0].long()
    resized_valid = F.interpolate(valid_mask.float(), size=feature.shape[-2:], mode="nearest")[:, 0].bool()
    cells = feature.detach().float().permute(0, 2, 3, 1)
    count = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total = torch.zeros((NUM_CLASSES, feature.shape[1]), dtype=torch.float64)
    square = torch.zeros_like(total)
    absolute = torch.zeros_like(total)
    for class_id in range(NUM_CLASSES):
        selected = cells[(resized_label == class_id) & resized_valid]
        count[class_id] = selected.shape[0]
        if selected.numel():
            value = selected.double().cpu()
            total[class_id] = value.sum(dim=0)
            square[class_id] = value.square().sum(dim=0)
            absolute[class_id] = value.abs().sum(dim=0)
    return {"count": count, "sum": total, "square": square, "absolute": absolute}


def _combine_semantic_statistics(
    case_statistics: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    case_ids: Sequence[str],
    layer: str,
) -> dict[str, torch.Tensor]:
    if not case_ids:
        raise ValueError("held-out semantic fold is empty")
    result: dict[str, torch.Tensor] = {}
    for field in ("count", "sum", "square", "absolute"):
        result[field] = sum((case_statistics[case_id][layer][field] for case_id in case_ids), start=torch.tensor(0.0))
    return result


def _fisher_ratio(statistics: Mapping[str, torch.Tensor]) -> torch.Tensor:
    count = statistics["count"].double()
    total = statistics["sum"].double()
    square = statistics["square"].double()
    valid = count.gt(0)
    if int(valid.sum()) < 2:
        raise ValueError("Fisher validation requires at least two observed classes")
    means = total / count[:, None].clamp_min(1.0)
    global_mean = total[valid].sum(dim=0) / count[valid].sum()
    between = (count[valid, None] * (means[valid] - global_mean).square()).sum(dim=0)
    within = (square[valid] - total[valid].square() / count[valid, None].clamp_min(1.0)).sum(dim=0)
    result = between / (within.clamp_min(0.0) + 1.0e-8)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite Fisher ratio")
    return result.float()


def _role_probe_cases(
    *,
    method: Any,
    data_root: Path,
    seed: int,
    site_id: str,
    device: torch.device,
) -> dict[str, Any]:
    labeled_dataset = H5LabeledDataset(
        data_root, seed=seed, dataset="fundus", sites=(site_id,), roles=("train_labeled",), transform=None
    )
    unlabeled_dataset = H5UnlabeledDataset(
        data_root, seed=seed, dataset="fundus", sites=(site_id,), transform=None
    )
    labeled_pairs = _dataset_records(labeled_dataset)
    unlabeled_pairs = _dataset_records(unlabeled_dataset)
    labeled_half = stable_half_assignment(labeled_pairs)
    unlabeled_half = stable_half_assignment(unlabeled_pairs)
    content_by_case: dict[str, dict[str, torch.Tensor]] = {}
    style_by_case: dict[str, dict[str, torch.Tensor]] = {}
    activation_by_case: dict[str, dict[str, torch.Tensor]] = {}
    semantic_by_case: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    feature_shapes: dict[str, tuple[int, ...]] = {}
    old_model = method.old_model
    if old_model is None:
        raise RuntimeError("missing frozen role model")
    old_model.eval()

    for index in range(len(labeled_dataset)):
        sample = labeled_dataset[index]
        case_id = str(sample["case_id"])
        image = sample["image"][None].to(device).detach().requires_grad_(True)
        label = sample["label"][None].to(device)
        valid = sample["valid_mask"][None].to(device)
        output = old_model(image)
        if output.decoder_features is None:
            raise RuntimeError("CRISP content probe decoder path is absent")
        loss = masked_cross_entropy(output.logits, label, valid) + multiclass_dice_loss(output.logits, label, valid)
        features = tuple(output.decoder_features[layer] for layer in FEATURE_LAYERS)
        gradients = torch.autograd.grad(loss, features, retain_graph=False, create_graph=False)
        content_by_case[case_id] = {
            layer: content_relevance_case(feature, gradient).cpu()
            for layer, feature, gradient in zip(FEATURE_LAYERS, features, gradients, strict=True)
        }
        semantic_by_case[case_id] = {
            layer: _semantic_case_statistics(feature, label, valid)
            for layer, feature in zip(FEATURE_LAYERS, features, strict=True)
        }
        for layer, feature in zip(FEATURE_LAYERS, features, strict=True):
            feature_shapes.setdefault(layer, tuple(feature.shape))
        if any(parameter.grad is not None for parameter in old_model.parameters()):
            raise AssertionError("content probe populated frozen previous-model .grad")
        del image, label, valid, output, loss, features, gradients

    style_transform = FrozenStyleProbeTransform(protocol_seed=seed)
    for index in range(len(unlabeled_dataset)):
        sample = unlabeled_dataset[index]
        case_id = str(sample["case_id"])
        views = style_transform(
            image=sample["weak_image"], dataset="fundus", site_id=site_id, case_id=case_id
        )
        paired = torch.stack([views["clean_image"], views["style_image"]], dim=0).to(device)
        valid = views["style_valid_mask"][None].to(device)
        with torch.inference_mode():
            output = old_model(paired)
        if output.decoder_features is None:
            raise RuntimeError("CRISP style probe decoder path is absent")
        style_by_case[case_id] = {}
        activation_by_case[case_id] = {}
        for layer in FEATURE_LAYERS:
            clean = output.decoder_features[layer][:1]
            style = output.decoder_features[layer][1:]
            style_by_case[case_id][layer] = style_sensitivity_case(clean, style, valid).cpu()
            activation_by_case[case_id][layer] = clean.float().square().sum(dim=(0, 2, 3)).sqrt().cpu()
            feature_shapes.setdefault(layer, tuple(clean.shape))
        if any(parameter.grad is not None for parameter in old_model.parameters()):
            raise AssertionError("style probe populated frozen previous-model .grad")
        del paired, valid, output

    labeled_ids = [case_id for _, case_id in labeled_pairs]
    unlabeled_ids = [case_id for _, case_id in unlabeled_pairs]
    return {
        "labeled_ids": labeled_ids,
        "unlabeled_ids": unlabeled_ids,
        "labeled_half": labeled_half,
        "unlabeled_half": unlabeled_half,
        "content_by_case": content_by_case,
        "style_by_case": style_by_case,
        "activation_by_case": activation_by_case,
        "semantic_by_case": semantic_by_case,
        "feature_shapes": feature_shapes,
    }


def _aggregate_role(
    probe: Mapping[str, Any],
    *,
    half: str | None,
    site_id: str,
    checkpoint_sha256: str,
    style_probe_sha256: str,
) -> ChannelRoleState:
    labeled_ids = [
        case_id
        for case_id in probe["labeled_ids"]
        if half is None or probe["labeled_half"][case_id] == half
    ]
    unlabeled_ids = [
        case_id
        for case_id in probe["unlabeled_ids"]
        if half is None or probe["unlabeled_half"][case_id] == half
    ]
    content = {
        layer: case_equal_mean([probe["content_by_case"][case_id][layer] for case_id in labeled_ids])
        for layer in FEATURE_LAYERS
    }
    style = {
        layer: case_equal_mean([probe["style_by_case"][case_id][layer] for case_id in unlabeled_ids])
        for layer in FEATURE_LAYERS
    }
    return build_channel_role_state(
        site_id=site_id,
        source_checkpoint_sha256=checkpoint_sha256,
        labeled_case_ids=labeled_ids,
        unlabeled_case_ids=unlabeled_ids,
        style_probe_sha256=style_probe_sha256,
        content_scores=content,
        style_scores=style,
        feature_shapes=probe["feature_shapes"],
    )


def _heldout_validation(
    *,
    probe: Mapping[str, Any],
    role_a: ChannelRoleState,
    role_b: ChannelRoleState,
    seed: int,
    transition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in FEATURE_LAYERS:
        folds: list[dict[str, Any]] = []
        for source_half, heldout_half, role in (("A", "B", role_a), ("B", "A", role_b)):
            heldout_labeled = [
                case_id for case_id in probe["labeled_ids"] if probe["labeled_half"][case_id] == heldout_half
            ]
            heldout_unlabeled = [
                case_id for case_id in probe["unlabeled_ids"] if probe["unlabeled_half"][case_id] == heldout_half
            ]
            statistics = _combine_semantic_statistics(probe["semantic_by_case"], heldout_labeled, layer)
            fisher = _fisher_ratio(statistics)
            alpha = role.invariant_weights[layer].cpu()
            beta = role.plastic_weights[layer].cpu()
            top_alpha = quartile_indices(alpha, largest=True)
            bottom_alpha = quartile_indices(alpha, largest=False)
            top_beta = quartile_indices(beta, largest=True)
            fisher_top = float(fisher[list(top_alpha)].median())
            fisher_bottom = float(fisher[list(bottom_alpha)].median())
            fisher_ratio = fisher_top / (fisher_bottom + 1.0e-8)
            style = case_equal_mean(
                [probe["style_by_case"][case_id][layer] for case_id in heldout_unlabeled]
            )
            style_plastic = float(style[list(top_beta)].median())
            style_stable = float(style[list(top_alpha)].median())
            style_ratio = style_plastic / (style_stable + 1.0e-8)
            foreground_nonzero = {
                str(class_id): float(statistics["absolute"][class_id, list(top_alpha)].sum()) > 0.0
                for class_id in FOREGROUND_IDS
            }
            activation = case_equal_mean(
                [probe["activation_by_case"][case_id][layer] for case_id in heldout_unlabeled]
            )
            plastic_alive_fraction = float(activation[list(top_beta)].gt(1.0e-6).float().mean())
            folds.append(
                {
                    "source_half": source_half,
                    "heldout_half": heldout_half,
                    "fisher_ratio": fisher_ratio,
                    "style_ratio": style_ratio,
                    "foreground_nonzero": foreground_nonzero,
                    "plastic_alive_fraction": plastic_alive_fraction,
                    "heldout_labeled_cases": len(heldout_labeled),
                    "heldout_unlabeled_cases": len(heldout_unlabeled),
                }
            )
        rows.append(
            {
                "seed": seed,
                "transition": transition,
                "layer": layer,
                "fisher_top_bottom_ratio_fold_a_to_b": folds[0]["fisher_ratio"],
                "fisher_top_bottom_ratio_fold_b_to_a": folds[1]["fisher_ratio"],
                "fisher_top_bottom_ratio": sum(item["fisher_ratio"] for item in folds) / 2.0,
                "style_plastic_stable_ratio_fold_a_to_b": folds[0]["style_ratio"],
                "style_plastic_stable_ratio_fold_b_to_a": folds[1]["style_ratio"],
                "style_plastic_stable_ratio": sum(item["style_ratio"] for item in folds) / 2.0,
                "foreground_class_1_nonzero": all(item["foreground_nonzero"]["1"] for item in folds),
                "foreground_class_2_nonzero": all(item["foreground_nonzero"]["2"] for item in folds),
                "plastic_alive_fraction": min(item["plastic_alive_fraction"] for item in folds),
                "plastic_alive_pass": min(item["plastic_alive_fraction"] for item in folds) >= 0.90,
                "fold_details_json": json.dumps(folds, sort_keys=True, separators=(",", ":")),
                "hidden_gt_usage": "none_training_visible_labeled_gt_only",
            }
        )
    return rows


def roles_phase(args: argparse.Namespace) -> int:
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    if seed_dir.exists():
        raise FileExistsError(f"refusing to overwrite CRISP feasibility seed: {seed_dir}")
    seed_dir.mkdir(parents=True)
    device = torch.device(args.device)
    style_audit = ROOT / "reports" / "experiment_status" / "CRISP_STYLE_PROBE_AUDIT.json"
    if not style_audit.is_file() or json.loads(style_audit.read_text())["status"] != "CRISP_STYLE_PROBE_AUDIT_PASSED":
        raise RuntimeError("CRISP style-probe prerequisite did not pass")
    style_probe_sha256 = crisp_style_probe_contract()["contract_sha256"]
    role_rows: list[dict[str, Any]] = []
    reproducibility_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    case_manifest: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        old_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, old_index)
        current_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, current_index)
        method, _ = load_frozen_method(current_checkpoint, device)
        _assert_frozen_previous(method, old_checkpoint)
        transition = f"{SITES[old_index]}->{SITES[current_index]}"
        old_before = _model_hash(method.old_model)
        probe = _role_probe_cases(
            method=method,
            data_root=args.data_root.resolve(),
            seed=args.seed,
            site_id=SITES[current_index],
            device=device,
        )
        full = _aggregate_role(
            probe,
            half=None,
            site_id=SITES[current_index],
            checkpoint_sha256=sha256_path(old_checkpoint),
            style_probe_sha256=style_probe_sha256,
        )
        role_a = _aggregate_role(
            probe,
            half="A",
            site_id=SITES[current_index],
            checkpoint_sha256=sha256_path(old_checkpoint),
            style_probe_sha256=style_probe_sha256,
        )
        role_b = _aggregate_role(
            probe,
            half="B",
            site_id=SITES[current_index],
            checkpoint_sha256=sha256_path(old_checkpoint),
            style_probe_sha256=style_probe_sha256,
        )
        role_path = seed_dir / f"channel_roles_{old_index}_{current_index}.pt"
        _atomic_torch_save(
            role_path,
            {
                "protocol_id": "crispseg_v0_1",
                "seed": args.seed,
                "transition": transition,
                "full": full.state_dict(),
                "half_a": role_a.state_dict(),
                "half_b": role_b.state_dict(),
                "labeled_half_assignment": probe["labeled_half"],
                "unlabeled_half_assignment": probe["unlabeled_half"],
                "optimizer_steps": 0,
                "hidden_gt_usage": "none",
            },
        )
        for data_role, case_ids, assignment in (
            ("current_train_labeled", probe["labeled_ids"], probe["labeled_half"]),
            ("current_train_unlabeled", probe["unlabeled_ids"], probe["unlabeled_half"]),
        ):
            for case_id in case_ids:
                case_manifest.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "site_id": SITES[current_index],
                        "data_role": data_role,
                        "case_id": case_id,
                        "split_half": assignment[case_id],
                    }
                )
        for layer in FEATURE_LAYERS:
            alpha, beta = full.invariant_weights[layer], full.plastic_weights[layer]
            q25, q75 = torch.quantile(alpha.float(), torch.tensor([0.25, 0.75])).tolist()
            role_rows.append(
                {
                    "seed": args.seed,
                    "transition": transition,
                    "layer": layer,
                    "channels": alpha.numel(),
                    "mean_alpha": float(alpha.mean()),
                    "alpha_q25": q25,
                    "alpha_q75": q75,
                    "alpha_iqr": q75 - q25,
                    "ess_alpha": effective_sample_size(alpha),
                    "ess_alpha_over_d": effective_sample_size(alpha) / alpha.numel(),
                    "ess_beta": effective_sample_size(beta),
                    "ess_beta_over_d": effective_sample_size(beta) / beta.numel(),
                    "zero_evidence_count": int(full.zero_evidence_masks[layer].sum()),
                    "content_mean": float(full.content_scores[layer].mean()),
                    "style_mean": float(full.style_scores[layer].mean()),
                }
            )
            alpha_a, alpha_b = role_a.invariant_weights[layer], role_b.invariant_weights[layer]
            reproducibility_rows.append(
                {
                    "seed": args.seed,
                    "transition": transition,
                    "layer": layer,
                    "spearman_alpha_a_b": spearman_correlation(alpha_a, alpha_b),
                    "top_quartile_jaccard": jaccard(
                        quartile_indices(alpha_a, largest=True), quartile_indices(alpha_b, largest=True)
                    ),
                    "bottom_quartile_jaccard": jaccard(
                        quartile_indices(alpha_a, largest=False), quartile_indices(alpha_b, largest=False)
                    ),
                    "labeled_a": sum(value == "A" for value in probe["labeled_half"].values()),
                    "labeled_b": sum(value == "B" for value in probe["labeled_half"].values()),
                    "unlabeled_a": sum(value == "A" for value in probe["unlabeled_half"].values()),
                    "unlabeled_b": sum(value == "B" for value in probe["unlabeled_half"].values()),
                }
            )
        validation_rows.extend(
            _heldout_validation(
                probe=probe, role_a=role_a, role_b=role_b, seed=args.seed, transition=transition
            )
        )
        old_after = _model_hash(method.old_model)
        if old_before != old_after:
            raise AssertionError("CRISP role audit mutated frozen previous model")
        if any(parameter.grad is not None for parameter in method.old_model.parameters()):
            raise AssertionError("CRISP role audit left previous-model gradients")
        transitions.append(
            {
                "transition": transition,
                "old_checkpoint": str(old_checkpoint),
                "old_checkpoint_sha256": sha256_path(old_checkpoint),
                "current_checkpoint": str(current_checkpoint),
                "current_checkpoint_sha256": sha256_path(current_checkpoint),
                "role_state": str(role_path),
                "role_state_sha256": sha256_path(role_path),
                "labeled_cases": len(probe["labeled_ids"]),
                "unlabeled_cases": len(probe["unlabeled_ids"]),
                "old_model_sha256_before": old_before,
                "old_model_sha256_after": old_after,
            }
        )
        del method, probe, full, role_a, role_b
        torch.cuda.empty_cache()
    for row in case_manifest:
        row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
    write_csv(seed_dir / "role_non_degeneracy.csv", role_rows)
    write_csv(seed_dir / "role_reproducibility.csv", reproducibility_rows)
    write_csv(seed_dir / "semantic_style_validation.csv", validation_rows)
    write_json(seed_dir / "role_case_manifest.json", case_manifest)
    summary = {
        "protocol_id": "crispseg_v0_1",
        "seed": args.seed,
        "status": "CRISP_ROLE_PHASE_COMPLETE",
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "role_model": "frozen_previous_site_model",
        "role_data": ["current_train_labeled", "current_train_unlabeled"],
        "transitions": transitions,
        "role_rows": len(role_rows),
        "reproducibility_rows": len(reproducibility_rows),
        "validation_rows": len(validation_rows),
        "case_manifest_sha256": sha256_bytes(canonical_json(case_manifest).encode("utf-8")),
        "style_probe_contract_sha256": style_probe_sha256,
        "style_probe_audit_sha256": sha256_path(style_audit),
        "training_manifest_sha256": sha256_path(
            args.data_root.resolve() / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv"
        ),
        "split_sha256": sha256_path(args.data_root.resolve() / "splits" / f"fundus_seed{args.seed}.json"),
        "environment": _environment(args.physical_gpu, device),
        "workspace_hash": _workspace_hash(),
    }
    write_json(seed_dir / "roles_summary.json", summary)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "output": str(seed_dir)}, indent=2))
    return 0


def _fixed_updates(
    data_root: Path,
    seed: int,
    site_id: str,
    device: torch.device,
) -> tuple[list[tuple[Any, Any]], list[dict[str, Any]]]:
    labeled_dataset = H5LabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        roles=("train_labeled",),
        transform=LabeledTransform(flip_probability=0.5),
    )
    unlabeled_dataset = H5UnlabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        transform=WeakStrongTransform(
            flip_probability=0.5,
            strong_noise_std=0.03,
            brightness_delta=0.10,
            contrast_delta=0.10,
            cutout_probability=0.5,
            cutout_fraction=0.20,
        ),
    )
    labeled_batcher = DeterministicBatcher(
        labeled_dataset,
        batch_size=2,
        seed=stable_seed("crisp-v0.1-update-order", seed, site_id),
        namespace=f"crisp-v0.1:{seed}:{site_id}:labeled",
        collate=collate_labeled,
        shuffle=False,
    )
    unlabeled_batcher = DeterministicBatcher(
        unlabeled_dataset,
        batch_size=4,
        seed=stable_seed("crisp-v0.1-update-order", seed, site_id),
        namespace=f"crisp-v0.1:{seed}:{site_id}:unlabeled",
        collate=collate_unlabeled,
        shuffle=False,
    )
    batches: list[tuple[Any, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index in range(UPDATE_BATCHES):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stable_seed("crisp-v0.1-update-augmentation", seed, site_id, index))
            labeled = labeled_batcher.batch_at(index)
            unlabeled = unlabeled_batcher.batch_at(index)
        row = {
            "batch_index": index,
            "labeled_case_ids": labeled.case_id,
            "unlabeled_case_ids": unlabeled.case_id,
            "unlabeled_geometry": unlabeled.geometry_record,
            "strong_valid_pixels": int(unlabeled.strong_valid_mask.sum()),
            "strong_total_pixels": unlabeled.strong_valid_mask.numel(),
        }
        row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
        manifest.append(row)
        batches.append((labeled.to(device, non_blocking=True), unlabeled.to(device, non_blocking=True)))
    return batches, manifest


def _take_cycling(loader: Iterable[Any], count: int, device: torch.device) -> list[Any]:
    result: list[Any] = []
    iterator = iter(loader)
    while len(result) < count:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        result.append(batch.to(device, non_blocking=True))
    return result


def _gated_assimilation(
    strong_logits: torch.Tensor,
    pseudo: Any,
    learnability: Any,
    strong_valid_mask: torch.Tensor,
) -> torch.Tensor:
    target = F.interpolate(pseudo.labels[:, None].float(), size=strong_logits.shape[-2:], mode="nearest")[:, 0].long()
    pseudo_valid = F.interpolate(pseudo.valid.float(), size=strong_logits.shape[-2:], mode="nearest").bool()
    weights = F.interpolate(
        learnability.score.detach(), size=strong_logits.shape[-2:], mode="bilinear", align_corners=False
    )
    valid = pseudo_valid & strong_valid_mask.detach().bool()
    weights = weights * valid.float()
    pixel = F.cross_entropy(strong_logits, target, ignore_index=-100, reduction="none")[:, None]
    return weighted_mean(pixel, weights, reference=strong_logits)


def _role_variants(state: ChannelRoleState, device: torch.device) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    continuous_alpha = {layer: state.invariant_weights[layer].to(device) for layer in FEATURE_LAYERS}
    continuous_beta = {layer: state.plastic_weights[layer].to(device) for layer in FEATURE_LAYERS}
    hard_alpha: dict[str, torch.Tensor] = {}
    hard_beta: dict[str, torch.Tensor] = {}
    uniform_alpha: dict[str, torch.Tensor] = {}
    uniform_beta: dict[str, torch.Tensor] = {}
    for layer in FEATURE_LAYERS:
        hard_alpha[layer], hard_beta[layer] = hard_rank_roles(continuous_alpha[layer])
        uniform_alpha[layer], uniform_beta[layer] = uniform_half_roles(continuous_alpha[layer])
    return {
        "continuous": (continuous_alpha, continuous_beta),
        "hard": (hard_alpha, hard_beta),
        "uniform": (uniform_alpha, uniform_beta),
    }


def _role_objective(
    method: Any,
    payload: Mapping[str, Any],
    labeled: Any,
    unlabeled: Any,
    roles: Mapping[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> dict[str, Any]:
    if method.old_model is None or method.old_anchor_bank is None:
        raise RuntimeError("functional audit lacks frozen previous state")
    method.model.train()
    method.old_model.eval()
    labeled_output = method.model(labeled.image)
    labeled_relation = method._relation(labeled_output.relation_features, method.current_anchor_bank)
    supervised = method._supervised_losses(
        labeled_output,
        labeled,
        relation_anchor_loss=relation_supervision_loss(labeled_relation.logits, labeled.label, labeled.valid_mask),
    )["loss_sup"]
    current_weak = method.model(unlabeled.weak_image)
    strong_output = method.model(unlabeled.strong_image)
    if current_weak.decoder_features is None or strong_output.decoder_features is None:
        raise RuntimeError("functional audit decoder path is absent")
    current_relation = method._relation(current_weak.relation_features, method.current_anchor_bank)
    strong_relation = method._relation(strong_output.relation_features, method.current_anchor_bank)
    with torch.no_grad():
        old_weak = method.old_model(unlabeled.weak_image)
        if old_weak.decoder_features is None:
            raise RuntimeError("functional audit old decoder path is absent")
        old_relation = method._relation(old_weak.relation_features, method.old_anchor_bank)
    pseudo = build_pseudo_labels(
        current_weak.logits.detach().softmax(dim=1),
        current_relation,
        tau_cls=float(method.config["tau_cls"]),
        tau_anchor=float(method.config["tau_anchor"]),
        delta_anchor=float(method.config["delta_anchor"]),
        tau_spatial=float(method.config["tau_spatial"]),
        temperature_cls=float(method.config["temperature_cls"]),
        temperature_anchor=float(method.config["temperature_anchor"]),
        spatial_floor=float(method.config["spatial_floor"]),
    )
    site_step = max(0, int(payload["site_step"]) - 1)
    learnability = compute_learnability(
        current_weak.logits.detach(),
        current_relation,
        pseudo,
        site_step=site_step,
        total_steps=max(1, int(method.total_steps)),
        rank_start=float(method.config["rank_start"]),
        rank_end=float(method.config["rank_end"]),
        rank_temperature=float(method.config["rank_temperature"]),
        relation_margin_center=float(method.config["relation_margin_center"]),
        relation_margin_temperature=float(method.config["relation_margin_temperature"]),
        min_rank_pixels=int(method.config["min_rank_pixels"]),
    )
    assimilation = _gated_assimilation(strong_output.logits, pseudo, learnability, unlabeled.strong_valid_mask)
    compatibility = _uniform_compatibility(zero_compatibility(old_relation.probabilities))
    relation = relation_consolidation_loss(
        strong_relation,
        old_relation,
        compatibility,
        unlabeled.strong_valid_mask,
        distill_temperature=float(method.config["distill_temperature"]),
    )
    auxiliary: dict[str, dict[str, Any]] = {}
    for role_name, (alpha, beta) in roles.items():
        ifc = invariant_feature_consolidation(
            current_weak.decoder_features, old_weak.decoder_features, alpha, valid_mask=None
        )
        pfc = plastic_feature_consistency(
            current_weak.decoder_features,
            strong_output.decoder_features,
            beta,
            unlabeled.strong_valid_mask,
        )
        kappa_i = 0.5 * (float(alpha["dec3"].mean()) + float(alpha["dec1"].mean()))
        kappa_p = 1.0 - kappa_i
        auxiliary[role_name] = {"ifc": ifc, "pfc": pfc, "kappa_i": kappa_i, "kappa_p": kappa_p}
    bootstrap_at = int(method.bootstrap_state.get("completed_at_site_step", -1))
    lambda_assim = float(method.config["lambda_assim"]) * method._assimilation_ramp(
        site_step, bootstrap_complete_at=bootstrap_at
    )
    lambda_relation = float(method.config["lambda_relation"]) * method._relation_ramp(site_step)
    base = supervised + lambda_assim * assimilation + lambda_relation * relation
    continuous = auxiliary["continuous"]
    hard = auxiliary["hard"]
    uniform = auxiliary["uniform"]
    totals = {
        "C0": base,
        "C1": base + lambda_relation * continuous["kappa_i"] * continuous["ifc"].loss,
        "C2": base + lambda_assim * continuous["kappa_p"] * continuous["pfc"].loss,
        "C3": base
        + lambda_relation * continuous["kappa_i"] * continuous["ifc"].loss
        + lambda_assim * continuous["kappa_p"] * continuous["pfc"].loss,
        "C4": base
        + lambda_relation * hard["kappa_i"] * hard["ifc"].loss
        + lambda_assim * hard["kappa_p"] * hard["pfc"].loss,
        "C5": base
        + lambda_relation * uniform["kappa_i"] * uniform["ifc"].loss
        + lambda_assim * uniform["kappa_p"] * uniform["pfc"].loss,
    }
    return {
        "totals": totals,
        "supervised": supervised,
        "assimilation": assimilation,
        "relation": relation,
        "auxiliary": auxiliary,
        "lambda_assim": lambda_assim,
        "lambda_relation": lambda_relation,
        "pseudo_valid_pixels": int(pseudo.valid.sum()),
    }


def _gradients(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    raw = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, raw, strict=True)
    )


def _grad_norm(gradients: tuple[torch.Tensor, ...]) -> float:
    return float(torch.sqrt(sum(value.float().square().sum() for value in gradients)))


def _grad_cosine(first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]) -> float:
    first_norm, second_norm = _grad_norm(first), _grad_norm(second)
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 0.0
    dot = sum((left.float() * right.float()).sum() for left, right in zip(first, second, strict=True))
    return float(dot / (first_norm * second_norm))


def _supervised_validation_loss(output: Any, batch: Any, method: Any) -> torch.Tensor:
    relation = method._relation(output.relation_features, method.current_anchor_bank)
    return (
        masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
        + float(method.config["lambda_dice"]) * multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
        + float(method.config["lambda_anchor_sup"])
        * relation_supervision_loss(relation.logits, batch.label, batch.valid_mask)
    )


@torch.no_grad()
def _evaluate_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor] | None,
    batches: list[Any],
    method: Any,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    for batch in batches:
        output = model(batch.image) if state is None else functional_call(model, state, (batch.image,))
        losses.append(float(_supervised_validation_loss(output, batch, method)))
        dices.append(float(1.0 - multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)))
    model.train(was_training)
    return sum(losses) / len(losses), sum(dices) / len(dices)


def functional_phase(args: argparse.Namespace) -> int:
    seed_dir = args.output_dir.resolve() / f"seed{args.seed}"
    roles_summary = seed_dir / "roles_summary.json"
    if not roles_summary.is_file():
        raise FileNotFoundError(f"role phase is incomplete: {roles_summary}")
    for name in ("gradient_scale.csv", "virtual_steps.csv", "functional_batch_manifest.json", "functional_summary.json"):
        if (seed_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite CRISP functional artifact: {seed_dir / name}")
    device = torch.device(args.device)
    gradient_rows: list[dict[str, Any]] = []
    virtual_rows: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for old_index, current_index in TRANSITIONS:
        transition = f"{SITES[old_index]}->{SITES[current_index]}"
        old_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, old_index)
        current_checkpoint = checkpoint_path(args.run_root.resolve(), args.seed, current_index)
        method, payload = load_frozen_method(current_checkpoint, device)
        _assert_frozen_previous(method, old_checkpoint)
        role_path = seed_dir / f"channel_roles_{old_index}_{current_index}.pt"
        role_bundle = torch.load(role_path, map_location="cpu", weights_only=False)
        role_state = ChannelRoleState.from_state_dict(role_bundle["full"], device=device)
        if role_state.source_checkpoint_sha256 != sha256_path(old_checkpoint):
            raise RuntimeError("role state source checkpoint does not match transition")
        roles = _role_variants(role_state, device)
        updates, update_manifest = _fixed_updates(
            args.data_root.resolve(), args.seed, SITES[current_index], device
        )
        for row in update_manifest:
            batch_manifest.append({"seed": args.seed, "transition": transition, "role": "current_update", **row})
        previous_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(),
                seed=args.seed,
                site_id=SITES[old_index],
                roles=("val",),
                batch_size=4,
                workers=args.workers,
            ),
            VAL_BATCHES,
            device,
        )
        current_batches = _take_cycling(
            labeled_loader(
                args.data_root.resolve(),
                seed=args.seed,
                site_id=SITES[current_index],
                roles=("val",),
                batch_size=4,
                workers=args.workers,
            ),
            VAL_BATCHES,
            device,
        )
        for role_name, batches in (("previous_val", previous_batches), ("current_val", current_batches)):
            for index, batch in enumerate(batches):
                row = {
                    "seed": args.seed,
                    "transition": transition,
                    "role": role_name,
                    "batch_index": index,
                    "case_ids": batch.case_id,
                }
                row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
                batch_manifest.append(row)
        model_before = _model_hash(method.model)
        old_before = _model_hash(method.old_model)
        anchors_before = method.old_anchor_bank.state_dict()
        previous_before_loss, previous_before_dice = _evaluate_state(
            method.model, None, previous_batches, method
        )
        current_before_loss, current_before_dice = _evaluate_state(method.model, None, current_batches, method)
        named_parameters = dict(method.model.named_parameters())
        named_buffers = dict(method.model.named_buffers())
        parameter_names = tuple(named_parameters)
        parameters = tuple(named_parameters.values())
        for update_index, (labeled, unlabeled) in enumerate(updates):
            objectives = _role_objective(method, payload, labeled, unlabeled, roles)
            continuous = objectives["auxiliary"]["continuous"]
            lambda_relation = float(objectives["lambda_relation"])
            lambda_assim = float(objectives["lambda_assim"])
            supervised_grad = _gradients(objectives["supervised"], parameters, retain_graph=True)
            relation_grad = _gradients(lambda_relation * objectives["relation"], parameters, retain_graph=True)
            assimilation_grad = _gradients(lambda_assim * objectives["assimilation"], parameters, retain_graph=True)
            ifc_grad = _gradients(
                lambda_relation * continuous["kappa_i"] * continuous["ifc"].loss,
                parameters,
                retain_graph=True,
            )
            pfc_grad = _gradients(
                lambda_assim * continuous["kappa_p"] * continuous["pfc"].loss,
                parameters,
                retain_graph=True,
            )
            variant_gradients: dict[str, tuple[torch.Tensor, ...]] = {}
            for variant in VARIANTS:
                variant_gradients[variant] = _gradients(
                    objectives["totals"][variant], parameters, retain_graph=variant != VARIANTS[-1]
                )
            relation_norm = _grad_norm(relation_grad)
            assimilation_norm = _grad_norm(assimilation_grad)
            ifc_norm = _grad_norm(ifc_grad)
            pfc_norm = _grad_norm(pfc_grad)
            c0_norm = _grad_norm(variant_gradients["C0"])
            c3_norm = _grad_norm(variant_gradients["C3"])
            ifc_ratio = ifc_norm / relation_norm if relation_norm > 0 else float("inf")
            pfc_ratio = pfc_norm / assimilation_norm if assimilation_norm > 0 else float("inf")
            total_ratio = c3_norm / c0_norm if c0_norm > 0 else float("inf")
            finite = all(
                math.isfinite(value)
                for value in (
                    relation_norm,
                    assimilation_norm,
                    ifc_norm,
                    pfc_norm,
                    c0_norm,
                    c3_norm,
                    ifc_ratio,
                    pfc_ratio,
                    total_ratio,
                )
            )
            gradient_rows.append(
                {
                    "seed": args.seed,
                    "transition": transition,
                    "update_batch": update_index,
                    "lambda_relation": lambda_relation,
                    "lambda_assim": lambda_assim,
                    "kappa_i": continuous["kappa_i"],
                    "kappa_p": continuous["kappa_p"],
                    "relation_gradient_norm": relation_norm,
                    "assimilation_gradient_norm": assimilation_norm,
                    "kappa_ifc_gradient_norm": ifc_norm,
                    "kappa_pfc_gradient_norm": pfc_norm,
                    "c0_total_gradient_norm": c0_norm,
                    "c3_total_gradient_norm": c3_norm,
                    "ifc_to_relation_ratio": ifc_ratio,
                    "pfc_to_assimilation_ratio": pfc_ratio,
                    "c3_to_c0_total_ratio": total_ratio,
                    "ifc_supervised_cosine": _grad_cosine(ifc_grad, supervised_grad),
                    "ifc_relation_cosine": _grad_cosine(ifc_grad, relation_grad),
                    "ifc_assimilation_cosine": _grad_cosine(ifc_grad, assimilation_grad),
                    "pfc_supervised_cosine": _grad_cosine(pfc_grad, supervised_grad),
                    "pfc_relation_cosine": _grad_cosine(pfc_grad, relation_grad),
                    "pfc_assimilation_cosine": _grad_cosine(pfc_grad, assimilation_grad),
                    "c3_supervised_cosine": _grad_cosine(variant_gradients["C3"], supervised_grad),
                    "c3_relation_cosine": _grad_cosine(variant_gradients["C3"], relation_grad),
                    "c3_assimilation_cosine": _grad_cosine(variant_gradients["C3"], assimilation_grad),
                    "finite": finite,
                    "old_model_gradient_nonnull": sum(
                        parameter.grad is not None for parameter in method.old_model.parameters()
                    ),
                    "pseudo_valid_pixels": objectives["pseudo_valid_pixels"],
                }
            )
            for variant in VARIANTS:
                gradients = variant_gradients[variant]
                raw_norm = _grad_norm(gradients)
                if not math.isfinite(raw_norm) or raw_norm <= 0.0:
                    raise FloatingPointError(f"invalid {variant} virtual gradient norm: {raw_norm}")
                scale = VIRTUAL_STEP_NORM / raw_norm
                updated = {
                    name: named_parameters[name] - scale * gradient
                    for name, gradient in zip(parameter_names, gradients, strict=True)
                }
                state = {**named_buffers, **updated}
                previous_after_loss, previous_after_dice = _evaluate_state(
                    method.model, state, previous_batches, method
                )
                current_after_loss, current_after_dice = _evaluate_state(
                    method.model, state, current_batches, method
                )
                virtual_rows.append(
                    {
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "virtual_step_norm": VIRTUAL_STEP_NORM,
                        "raw_gradient_norm": raw_norm,
                        "objective": float(objectives["totals"][variant].detach()),
                        "loss_sup": float(objectives["supervised"].detach()),
                        "loss_assim_r0": float(objectives["assimilation"].detach()),
                        "loss_relation_r0": float(objectives["relation"].detach()),
                        "loss_ifc_continuous": float(continuous["ifc"].loss.detach()),
                        "loss_pfc_continuous": float(continuous["pfc"].loss.detach()),
                        "kappa_i": continuous["kappa_i"],
                        "kappa_p": continuous["kappa_p"],
                        "previous_val_loss_before": previous_before_loss,
                        "previous_val_loss_after": previous_after_loss,
                        "previous_val_loss_delta": previous_after_loss - previous_before_loss,
                        "previous_val_dice_before": previous_before_dice,
                        "previous_val_dice_after": previous_after_dice,
                        "previous_val_dice_delta": previous_after_dice - previous_before_dice,
                        "current_val_loss_before": current_before_loss,
                        "current_val_loss_after": current_after_loss,
                        "current_val_loss_delta": current_after_loss - current_before_loss,
                        "current_val_dice_before": current_before_dice,
                        "current_val_dice_after": current_after_dice,
                        "current_val_dice_delta": current_after_dice - current_before_dice,
                        "checkpoint_or_model_mutated": False,
                    }
                )
            del (
                objectives,
                supervised_grad,
                relation_grad,
                assimilation_grad,
                ifc_grad,
                pfc_grad,
                variant_gradients,
            )
        model_after = _model_hash(method.model)
        old_after = _model_hash(method.old_model)
        anchors_after = method.old_anchor_bank.state_dict()
        anchors_equal = all(
            torch.equal(anchors_before[key].cpu(), anchors_after[key].cpu())
            for key in anchors_before
            if isinstance(anchors_before[key], torch.Tensor)
        )
        if model_before != model_after or old_before != old_after or not anchors_equal:
            raise AssertionError("CRISP functional audit mutated frozen model/anchor state")
        transitions.append(
            {
                "transition": transition,
                "old_checkpoint": str(old_checkpoint),
                "old_checkpoint_sha256": sha256_path(old_checkpoint),
                "current_checkpoint": str(current_checkpoint),
                "current_checkpoint_sha256": sha256_path(current_checkpoint),
                "role_state": str(role_path),
                "role_state_sha256": sha256_path(role_path),
                "model_sha256_before": model_before,
                "model_sha256_after": model_after,
                "old_model_sha256_before": old_before,
                "old_model_sha256_after": old_after,
                "historical_anchor_immutable": anchors_equal,
                "update_batches": UPDATE_BATCHES,
                "previous_val_batches": VAL_BATCHES,
                "current_val_batches": VAL_BATCHES,
            }
        )
        del method, role_state, roles, updates, previous_batches, current_batches
        torch.cuda.empty_cache()
    write_csv(seed_dir / "gradient_scale.csv", gradient_rows)
    write_csv(seed_dir / "virtual_steps.csv", virtual_rows)
    write_json(seed_dir / "functional_batch_manifest.json", batch_manifest)
    summary = {
        "protocol_id": "crispseg_v0_1",
        "seed": args.seed,
        "status": "CRISP_FUNCTIONAL_PHASE_COMPLETE",
        "optimizer_steps": 0,
        "hidden_gt_usage": "none",
        "update_batches_per_pair": UPDATE_BATCHES,
        "previous_val_batches_per_pair": VAL_BATCHES,
        "current_val_batches_per_pair": VAL_BATCHES,
        "virtual_step_norm": VIRTUAL_STEP_NORM,
        "variants": list(VARIANTS),
        "transitions": transitions,
        "gradient_rows": len(gradient_rows),
        "virtual_rows": len(virtual_rows),
        "batch_manifest_sha256": sha256_bytes(canonical_json(batch_manifest).encode("utf-8")),
        "environment": _environment(args.physical_gpu, device),
        "workspace_hash": _workspace_hash(),
    }
    write_json(seed_dir / "functional_summary.json", summary)
    print(json.dumps({"status": summary["status"], "seed": args.seed, "output": str(seed_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("roles", "functional"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_seed = stable_seed("crisp-v0.1-feasibility", args.phase, args.seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return roles_phase(args) if args.phase == "roles" else functional_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
