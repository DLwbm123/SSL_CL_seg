#!/usr/bin/env python3
"""Formal two-case supervised overfit gate for the LCR-Seg training stack."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lcrseg.data import H5LabeledDataset, collate_labeled  # noqa: E402
from lcrseg.engine.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint  # noqa: E402
from lcrseg.engine.metrics import masked_cross_entropy, multiclass_dice, multiclass_dice_loss  # noqa: E402
from lcrseg.models import UNet2D  # noqa: E402

_DATASET_SPECS = {
    "fundus": {"channels": 3, "classes": 3, "default_site": "REFUGE"},
    "prostate": {"channels": 1, "classes": 2, "default_site": "RUNMC"},
    "mnms": {"channels": 1, "classes": 4, "default_site": "Siemens"},
}
_PALETTE = np.asarray(((0, 0, 0), (47, 166, 84), (244, 180, 0), (216, 75, 75)), dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary JSON: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty curve")
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_run_root(data_root: Path, run_root: Path) -> None:
    frozen_roots = (data_root / "h5" / "v1", data_root / "manifests", data_root / "splits", data_root / "checksums")
    resolved_run = run_root.resolve()
    for frozen in frozen_roots:
        resolved_frozen = frozen.resolve()
        if resolved_run == resolved_frozen or resolved_frozen in resolved_run.parents:
            raise ValueError(f"run root may not be inside frozen input: {resolved_frozen}")


def _select_two_case_indices(dataset: H5LabeledDataset) -> list[int]:
    by_case: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        by_case.setdefault(sample.row["case_id"], []).append(index)
    if len(by_case) < 2:
        raise RuntimeError("two-case overfit requires two labeled patient records")
    selected: list[int] = []
    for indices in list(by_case.values())[:2]:
        selected.append(indices[len(indices) // 2])
    return selected


def _make_montage(image: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    panels: list[Image.Image] = []
    for source, pred, truth in zip(image.cpu(), prediction.cpu(), target.cpu()):
        if source.shape[0] == 3:
            rgb = (source.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        else:
            normalized = source[0].float()
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min()).clamp_min(1e-6)
            rgb = np.repeat((normalized.numpy() * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        pred_rgb = _PALETTE[pred.numpy().clip(0, len(_PALETTE) - 1)]
        target_rgb = _PALETTE[truth.numpy().clip(0, len(_PALETTE) - 1)]
        panels.append(Image.fromarray(np.concatenate((rgb, pred_rgb, target_rgb), axis=1)))
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height))
    top = 0
    for panel in panels:
        canvas.paste(panel, (0, top))
        top += panel.height
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("LCRSEG_RUN_ROOT", "/home/jiangsuiyang/SSL_CL/runs")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=tuple(_DATASET_SPECS), default="fundus")
    parser.add_argument("--site", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    data_root = args.root.resolve()
    run_root = args.run_root.resolve()
    _validate_run_root(data_root, run_root)
    if not (data_root / "h5" / "v1" / "FROZEN").is_file():
        raise RuntimeError(f"frozen marker is missing: {data_root / 'h5/v1/FROZEN'}")
    if not (data_root / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv").is_file():
        raise FileNotFoundError("training manifest is missing")
    site = args.site or _DATASET_SPECS[args.dataset]["default_site"]
    output_dir = args.output_dir or run_root / "m0" / f"two_case_overfit_{args.dataset}_{site}_seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    source = H5LabeledDataset(data_root, seed=args.seed, dataset=args.dataset, sites=(site,))
    selected_indices = _select_two_case_indices(source)
    batch = collate_labeled([source[index] for index in selected_indices]).to(device)
    spec = _DATASET_SPECS[args.dataset]
    model = UNet2D(spec["channels"], spec["classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loss_rows: list[dict[str, float]] = []
    dice_rows: list[dict[str, float]] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        context = torch.cuda.amp.autocast(enabled=True) if device.type == "cuda" else nullcontext()
        with context:
            output = model(batch.image)
            ce = masked_cross_entropy(output.logits, batch.label, batch.valid_mask)
            dice_loss = multiclass_dice_loss(output.logits, batch.label, batch.valid_mask)
            loss = ce + dice_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            dice = multiclass_dice(output.logits, batch.label, batch.valid_mask).detach().cpu()
        loss_rows.append({"step": float(step + 1), "loss_total": float(loss.detach()), "loss_ce": float(ce.detach()), "loss_dice": float(dice_loss.detach())})
        dice_row = {"step": float(step + 1), "mean_foreground_dice": float(dice.mean())}
        dice_row.update({f"dice_class_{index + 1}": float(value) for index, value in enumerate(dice)})
        dice_rows.append(dice_row)

    model.eval()
    with torch.no_grad():
        final_output = model(batch.image)
        final_prediction = final_output.logits.argmax(dim=1)
        final_dice = multiclass_dice(final_output.logits, batch.label, batch.valid_mask).detach().cpu()
    manifest = data_root / "manifests" / "training" / f"lcrseg_v1_seed{args.seed}.csv"
    split = data_root / "splits" / f"{args.dataset}_seed{args.seed}.json"
    resolved_config = {
        "seed": args.seed,
        "dataset": args.dataset,
        "site": site,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "model": {"name": "unet2d", "base_channels": 16, "relation_dim": 128, **spec},
        "data_root": str(data_root),
        "run_root": str(run_root),
        "device": str(device),
    }
    checkpoint = checkpoint_payload(
        method_name="m0_supervised_overfit",
        method_version="0.1",
        git_commit="NO_GIT_WORKTREE",
        config_resolved=resolved_config,
        site_id=site,
        site_index=0,
        epoch=0,
        site_step=args.steps,
        global_step=args.steps,
        current_model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state={},
        scaler_state=scaler.state_dict(),
        current_anchor_state={},
        historical_anchor_state={},
        bootstrap_state={"complete": False},
        method_statistics={},
        data_split_hash=_sha256(split),
        manifest_hash=_sha256(manifest),
    )
    checkpoint_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(checkpoint_path, checkpoint)
    restored = load_checkpoint(checkpoint_path, map_location=device)
    restored_model = UNet2D(spec["channels"], spec["classes"]).to(device)
    restored_model.load_state_dict(restored["current_model_state"])
    restored_model.eval()
    with torch.no_grad():
        restored_logits = restored_model(batch.image).logits
    checkpoint_max_abs_error = float((restored_logits - final_output.logits).abs().max().cpu())
    if checkpoint_max_abs_error > 1e-6:
        raise RuntimeError(f"checkpoint restore differs by {checkpoint_max_abs_error}")

    _write_curve(output_dir / "loss_curve.csv", loss_rows)
    _write_curve(output_dir / "dice_curve.csv", dice_rows)
    _make_montage(batch.image, final_prediction, batch.label, output_dir / "final_prediction_montage.png")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output_dir / "environment.txt").write_text(
        "\n".join(
            (
                f"python={platform.python_version()}",
                f"torch={torch.__version__}",
                f"cuda={torch.version.cuda}",
                f"cudnn={torch.backends.cudnn.version()}",
                f"device={device}",
            )
        )
        + "\n"
    )
    _write_json(output_dir / "config.json", resolved_config)
    final_mean_dice = float(final_dice.mean())
    final_min_dice = float(final_dice.min())
    acceptance = {
        "loss_decreased": loss_rows[-1]["loss_total"] < loss_rows[0]["loss_total"],
        "mean_foreground_dice_ge_0_95": final_mean_dice >= 0.95,
        "minimum_foreground_dice_ge_0_85": final_min_dice >= 0.85,
        "checkpoint_restored": checkpoint_max_abs_error <= 1e-6,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": batch.case_id,
        "patient_ids": batch.patient_id,
        "slice_indices": batch.slice_index,
        "initial_loss": loss_rows[0]["loss_total"],
        "final_loss": loss_rows[-1]["loss_total"],
        "final_mean_foreground_dice": final_mean_dice,
        "final_minimum_foreground_dice": final_min_dice,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "checkpoint_max_abs_error": checkpoint_max_abs_error,
        "acceptance": acceptance,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "two_case_overfit.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not all(acceptance.values()):
        raise SystemExit("M0 two-case overfit acceptance failed; inspect the retained run bundle")


if __name__ == "__main__":
    main()
