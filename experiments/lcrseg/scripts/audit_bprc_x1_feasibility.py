#!/usr/bin/env python3
"""Run one seed of the user-authorized BPRC-X1 exploratory diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcrseg.common import canonical_json, sha256_bytes, sha256_path, write_csv, write_json  # noqa: E402
from scripts import audit_bprc_feasibility as base  # noqa: E402


PROTOCOL_ID = "bprcseg_x1_exploratory"
VARIANTS = ("X0", "X1")
X1_SCALE = 1.0 / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tarc-analysis-dir", type=Path, required=True)
    parser.add_argument("--bprc-v01-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _candidate_objectives(objectives: dict[str, Any]) -> dict[str, torch.Tensor]:
    num_classes = int(objectives["current_scores"].shape[1])
    if num_classes != 3:
        raise AssertionError(f"BPRC-X1 fixed C=3, observed C={num_classes}")
    return {
        "X0": objectives["relations"]["B0"],
        "X1": objectives["relations"]["B2"] / float(num_classes),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    seed_dir = output_dir / f"seed{args.seed}"
    if seed_dir.exists():
        raise FileExistsError(f"refusing to overwrite BPRC-X1 seed: {seed_dir}")
    seed_dir.mkdir(parents=True)
    device = torch.device(args.device)
    gradient_rows: list[dict[str, Any]] = []
    virtual_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    transition_summaries: list[dict[str, Any]] = []

    for old_index, current_index in base.TRANSITIONS:
        tarc_seed_dir = args.tarc_analysis_dir.resolve() / f"seed{args.seed}"
        bundle_path = tarc_seed_dir / f"transport_{old_index}_{current_index}.pt"
        bundle = torch.load(bundle_path, map_location="cpu")
        method, payload = base.load_frozen_method(Path(bundle["current_checkpoint"]), device)
        method.train()
        if method.old_model is None or method.old_anchor_bank is None:
            raise RuntimeError("BPRC-X1 requires the frozen historical teacher")
        method.old_model.eval()
        transition = f"{bundle['old_site_id']}->{bundle['current_site_id']}"
        before_model_hash = base._model_hash(method.model)
        before_old_hash = base._model_hash(method.old_model)

        update_batches, update_manifest = base._fixed_updates(
            args.data_root.resolve(), args.seed, bundle["current_site_id"]
        )
        previous_batches = base._take_cycling(
            base.labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=bundle["old_site_id"],
                roles=("val",), batch_size=4, workers=args.workers,
            ),
            base.VAL_BATCHES,
            device,
        )
        current_batches = base._take_cycling(
            base.labeled_loader(
                args.data_root.resolve(), seed=args.seed, site_id=bundle["current_site_id"],
                roles=("val",), batch_size=4, workers=args.workers,
            ),
            base.VAL_BATCHES,
            device,
        )
        for row in update_manifest:
            batch_manifest.append({"seed": args.seed, "transition": transition, **row})
        batch_manifest.extend(
            {"seed": args.seed, "transition": transition, **row}
            for row in base._validation_manifest("previous_val", previous_batches)
        )
        batch_manifest.extend(
            {"seed": args.seed, "transition": transition, **row}
            for row in base._validation_manifest("current_val", current_batches)
        )

        native_anchors = method.current_anchor_bank.anchors.detach()
        old_probabilities = base._old_probabilities(method, previous_batches)
        method.model.eval()
        previous_before = base._evaluate(
            model=method.model,
            state=None,
            batches=previous_batches,
            native_anchors=native_anchors,
            old_probabilities=old_probabilities,
        )
        current_before = base._evaluate(
            model=method.model,
            state=None,
            batches=current_batches,
            native_anchors=native_anchors,
        )
        exact_previous_before = base._baseline_loss(method.model, previous_batches, native_anchors)
        exact_current_before = base._baseline_loss(method.model, current_batches, native_anchors)
        if (
            abs(previous_before["loss"] - exact_previous_before) > 1.0e-12
            or abs(current_before["loss"] - exact_current_before) > 1.0e-12
        ):
            raise AssertionError("BPRC-X1 evaluator diverges from exact TARC baseline")

        named_parameters = dict(method.model.named_parameters())
        named_buffers = {name: value.detach() for name, value in method.model.named_buffers()}
        parameter_names = tuple(named_parameters)
        parameters = tuple(named_parameters.values())

        for update_index, (labeled_cpu, unlabeled_cpu) in enumerate(update_batches):
            labeled = labeled_cpu.to(device, non_blocking=True)
            unlabeled = unlabeled_cpu.to(device, non_blocking=True)
            objectives = base._prepare_objectives(method, payload, labeled, unlabeled)
            if objectives["b0_abs_error"] > 1.0e-7:
                raise AssertionError(f"X0/B0 exact R0 mismatch: {objectives['b0_abs_error']}")
            candidates = _candidate_objectives(objectives)
            sup_gradient = base._gradients(objectives["sup"], parameters)
            assim_gradient = base._gradients(objectives["assim"], parameters)
            relation_gradients = {
                variant: base._gradients(loss, parameters)
                for variant, loss in candidates.items()
            }
            x0_norm = base._norm(relation_gradients["X0"])
            if not math.isfinite(x0_norm) or x0_norm <= 0:
                raise FloatingPointError("invalid X0 relation gradient norm")

            for variant in VARIANTS:
                relation_loss = candidates[variant]
                relation_gradient = relation_gradients[variant]
                relation_norm = base._norm(relation_gradient)
                total_loss = (
                    objectives["sup"]
                    + objectives["lambda_assim"] * objectives["assim"]
                    + objectives["lambda_relation"] * relation_loss
                )
                total_gradient = base._gradients(total_loss, parameters)
                total_norm = base._norm(total_gradient)
                if not math.isfinite(total_norm) or total_norm <= 0:
                    raise FloatingPointError("invalid BPRC-X1 total virtual gradient")
                step_scale = base.VIRTUAL_STEP_NORM / total_norm
                updated_parameters = {
                    name: named_parameters[name].detach() - step_scale * gradient
                    for name, gradient in zip(parameter_names, total_gradient, strict=True)
                }
                functional_state = {**named_buffers, **updated_parameters}
                method.model.eval()
                previous_after = base._evaluate(
                    model=method.model,
                    state=functional_state,
                    batches=previous_batches,
                    native_anchors=native_anchors,
                    old_probabilities=old_probabilities,
                )
                current_after = base._evaluate(
                    model=method.model,
                    state=functional_state,
                    batches=current_batches,
                    native_anchors=native_anchors,
                )
                if update_index == 0 and variant == "X0":
                    exact_previous = base._functional_val_loss(
                        method.model, functional_state, previous_batches, native_anchors
                    )
                    exact_current = base._functional_val_loss(
                        method.model, functional_state, current_batches, native_anchors
                    )
                    if (
                        abs(previous_after["loss"] - exact_previous) > 1.0e-12
                        or abs(current_after["loss"] - exact_current) > 1.0e-12
                    ):
                        raise AssertionError("BPRC-X1 evaluator diverges from exact functional loss")

                gradient_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "relation_form": (
                            "categorical_pixel_mean"
                            if variant == "X0"
                            else "top2_pairwise_class_balanced_div_num_classes"
                        ),
                        "fixed_relation_scale": 1.0 if variant == "X0" else X1_SCALE,
                        "relation_loss": float(relation_loss.detach()),
                        "relation_gradient_norm": relation_norm,
                        "relation_gradient_norm_ratio_to_x0": relation_norm / x0_norm,
                        "cos_relation_with_sup": base._cosine(relation_gradient, sup_gradient),
                        "cos_relation_with_assim": base._cosine(relation_gradient, assim_gradient),
                        "sup_gradient_norm": base._norm(sup_gradient),
                        "assim_gradient_norm": base._norm(assim_gradient),
                        "total_gradient_norm": total_norm,
                        "b0_r0_loss_abs_error": objectives["b0_abs_error"],
                        "old_model_gradient_nonnull": sum(
                            parameter.grad is not None for parameter in method.old_model.parameters()
                        ),
                        "optimizer_step_called": False,
                        "all_finite": all(
                            math.isfinite(value)
                            for value in (relation_norm, total_norm, float(relation_loss.detach()))
                        ),
                        "hidden_gt_usage": "none",
                    }
                )
                virtual_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "seed": args.seed,
                        "transition": transition,
                        "update_batch": update_index,
                        "variant": variant,
                        "virtual_step_norm": base.VIRTUAL_STEP_NORM,
                        "previous_val_loss_before": previous_before["loss"],
                        "previous_val_loss_after": previous_after["loss"],
                        "previous_val_loss_delta": previous_after["loss"] - previous_before["loss"],
                        "previous_val_dice_before": previous_before["dice"],
                        "previous_val_dice_after": previous_after["dice"],
                        "previous_val_dice_delta": previous_after["dice"] - previous_before["dice"],
                        "current_val_loss_before": current_before["loss"],
                        "current_val_loss_after": current_after["loss"],
                        "current_val_loss_delta": current_after["loss"] - current_before["loss"],
                        "current_val_dice_before": current_before["dice"],
                        "current_val_dice_after": current_after["dice"],
                        "current_val_dice_delta": current_after["dice"] - current_before["dice"],
                        "old_model_gradient_nonnull": sum(
                            parameter.grad is not None for parameter in method.old_model.parameters()
                        ),
                        "checkpoint_or_model_mutated": False,
                        "hidden_gt_usage": "none_visible_update_and_val_only",
                    }
                )
                for class_id in range(3):
                    for region in ("all", "boundary", "interior"):
                        key = f"class{class_id}_{region}"
                        if key not in previous_after["margins"]:
                            continue
                        margin_rows.append(
                            {
                                "protocol_id": PROTOCOL_ID,
                                "seed": args.seed,
                                "transition": transition,
                                "update_batch": update_index,
                                "variant": variant,
                                "class_id": class_id,
                                "region": region,
                                "margin_agreement_before": previous_before["margins"][key],
                                "margin_agreement_after": previous_after["margins"][key],
                                "margin_agreement_delta": previous_after["margins"][key]
                                - previous_before["margins"][key],
                                "margin_abs_error_after": 1.0 - previous_after["margins"][key],
                                "metric_function": "scripts.audit_tarc_relation_fidelity._margin",
                                "hidden_gt_usage": "post_hoc_visible_previous_val_only",
                            }
                        )
                method.model.train()
            del objectives, candidates, sup_gradient, assim_gradient, relation_gradients

        after_model_hash = base._model_hash(method.model)
        after_old_hash = base._model_hash(method.old_model)
        if before_model_hash != after_model_hash or before_old_hash != after_old_hash:
            raise AssertionError("BPRC-X1 diagnostic mutated a frozen model")
        transition_summaries.append(
            {
                "transition": transition,
                "current_model_sha256_before": before_model_hash,
                "current_model_sha256_after": after_model_hash,
                "old_model_sha256_before": before_old_hash,
                "old_model_sha256_after": after_old_hash,
                "models_unchanged": True,
                "update_batches": base.UPDATE_BATCHES,
                "previous_val_batches": base.VAL_BATCHES,
                "current_val_batches": base.VAL_BATCHES,
                "checkpoint_sha256": sha256_path(Path(bundle["current_checkpoint"])),
                "transport_bundle_sha256": sha256_path(bundle_path),
            }
        )
        del method, update_batches, previous_batches, current_batches, old_probabilities
        torch.cuda.empty_cache()

    manifest_payload = {
        "protocol_id": PROTOCOL_ID,
        "seed": args.seed,
        "entries": batch_manifest,
    }
    manifest_payload["combined_sha256"] = sha256_bytes(
        canonical_json(batch_manifest).encode("utf-8")
    )
    frozen_manifest_path = (
        args.bprc_v01_dir.resolve() / f"seed{args.seed}" / "fixed_batch_lists.json"
    )
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_sha = frozen_manifest["combined_sha256"]
    if manifest_payload["combined_sha256"] != frozen_sha:
        raise AssertionError("BPRC-X1 did not reuse the exact BPRC V0.1 fixed batches")

    write_csv(seed_dir / "gradient_scale.csv", gradient_rows)
    write_csv(seed_dir / "virtual_steps.csv", virtual_rows)
    write_csv(seed_dir / "margin_analysis.csv", margin_rows)
    write_json(seed_dir / "fixed_batch_lists.json", manifest_payload)
    write_json(
        seed_dir / "summary.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "BPRC_X1_SEED_DIAGNOSTIC_COMPLETE",
            "seed": args.seed,
            "variants": list(VARIANTS),
            "x1_fixed_scale": X1_SCALE,
            "optimizer_steps": 0,
            "gradient_rows": len(gradient_rows),
            "virtual_rows": len(virtual_rows),
            "margin_rows": len(margin_rows),
            "batch_list_sha256": manifest_payload["combined_sha256"],
            "reused_bprc_v0_1_batch_list_sha256": frozen_sha,
            "transitions": transition_summaries,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "seed": args.seed,
                "gradient_rows": len(gradient_rows),
                "virtual_rows": len(virtual_rows),
                "margin_rows": len(margin_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
