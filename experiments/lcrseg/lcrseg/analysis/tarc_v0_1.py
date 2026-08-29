"""Read-only TARC V0.1 feasibility helpers for frozen R0 checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data import H5LabeledDataset, collate_labeled
from ..transport import AllClassTransport, CasePrototypeBatch, build_case_prototypes, estimate_all_class_transport, transport_anchors
from .v0_4 import load_frozen_method


RUN_NAMES = {
    0: "fundus_seed0_lcrseg_uniform_relation_kd_full200e",
    1: "fundus_seed1_lcrseg_v0_3_r0_legacy_uniform_full200e",
    2: "fundus_seed2_lcrseg_v0_3_r0_legacy_uniform_full200e",
}
SITES = ("REFUGE", "RIM_ONE_r3", "Drishti_GS")
TRANSITIONS = ((0, 1), (1, 2))
NUM_CLASSES = 3
MINIMUM_PIXELS = 32


@dataclass(frozen=True)
class TransitionBundle:
    seed: int
    old_site_index: int
    current_site_index: int
    old_site_id: str
    current_site_id: str
    old_checkpoint: Path
    current_checkpoint: Path
    old_anchors: torch.Tensor
    native_current_anchors: torch.Tensor
    global_anchors: torch.Tensor
    class_anchors: torch.Tensor
    transport: AllClassTransport
    source_case_ids: tuple[str, ...]
    source_pixel_counts: torch.Tensor
    historical_anchor_equal: bool


def checkpoint_path(run_root: Path, seed: int, site_index: int) -> Path:
    return Path(run_root) / RUN_NAMES[int(seed)] / f"checkpoint_final_site{site_index}_{SITES[site_index]}.pt"


def labeled_loader(
    data_root: Path,
    *,
    seed: int,
    site_id: str,
    roles: Iterable[str],
    batch_size: int = 4,
    workers: int = 0,
) -> DataLoader[Any]:
    dataset = H5LabeledDataset(
        data_root,
        seed=seed,
        dataset="fundus",
        sites=(site_id,),
        roles=tuple(roles),
        transform=None,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_labeled,
        generator=torch.Generator().manual_seed(0),
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def paired_case_prototypes(
    old_model: torch.nn.Module,
    current_model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[CasePrototypeBatch, CasePrototypeBatch, tuple[str, ...]]:
    old_parts: list[CasePrototypeBatch] = []
    current_parts: list[CasePrototypeBatch] = []
    case_ids: list[str] = []
    old_model.eval()
    current_model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        old_features = old_model(batch.image).relation_features
        current_features = current_model(batch.image).relation_features
        labels = F.interpolate(batch.label[:, None].float(), size=old_features.shape[-2:], mode="nearest")[:, 0].long()
        old_parts.append(build_case_prototypes(old_features, labels, num_classes=NUM_CLASSES, minimum_pixels=MINIMUM_PIXELS))
        current_parts.append(build_case_prototypes(current_features, labels, num_classes=NUM_CLASSES, minimum_pixels=MINIMUM_PIXELS))
        case_ids.extend(batch.case_id)

    def combine(parts: list[CasePrototypeBatch]) -> CasePrototypeBatch:
        return CasePrototypeBatch(
            prototypes=torch.cat([item.prototypes for item in parts], dim=0).detach(),
            valid=torch.cat([item.valid for item in parts], dim=0).detach(),
            pixel_counts=torch.cat([item.pixel_counts for item in parts], dim=0).detach(),
        )

    return combine(old_parts), combine(current_parts), tuple(case_ids)


@torch.no_grad()
def current_frame_oracle(
    current_model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts: list[CasePrototypeBatch] = []
    current_model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        features = current_model(batch.image).relation_features
        labels = F.interpolate(batch.label[:, None].float(), size=features.shape[-2:], mode="nearest")[:, 0].long()
        parts.append(build_case_prototypes(features, labels, num_classes=NUM_CLASSES, minimum_pixels=MINIMUM_PIXELS))
    prototypes = torch.cat([item.prototypes for item in parts], dim=0)
    valid = torch.cat([item.valid for item in parts], dim=0)
    oracle = torch.zeros((NUM_CLASSES, prototypes.shape[-1]), dtype=torch.float32, device=device)
    counts = valid.sum(dim=0)
    for class_id in range(NUM_CLASSES):
        if int(counts[class_id]) < 1:
            continue
        oracle[class_id] = F.normalize(prototypes[valid[:, class_id], class_id].mean(dim=0)[None], dim=1)[0]
    return oracle.detach(), counts.detach()


def _anchors(payload: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    state = payload[key]
    anchors = state["anchors"].detach().float().to(device)
    valid = state["valid"].detach().to(device)
    if anchors.shape[0] != NUM_CLASSES or not bool(valid.all()):
        raise RuntimeError(f"invalid all-class anchor state: {key}")
    return anchors


@torch.no_grad()
def build_transition_bundle(
    *,
    data_root: Path,
    run_root: Path,
    seed: int,
    old_site_index: int,
    current_site_index: int,
    device: torch.device,
    workers: int,
) -> tuple[TransitionBundle, torch.nn.Module, torch.nn.Module]:
    old_checkpoint = checkpoint_path(run_root, seed, old_site_index)
    current_checkpoint = checkpoint_path(run_root, seed, current_site_index)
    old_method, old_payload = load_frozen_method(old_checkpoint, device)
    current_method, current_payload = load_frozen_method(current_checkpoint, device)
    old_anchors = _anchors(old_payload, "current_anchor_state", device)
    historical = _anchors(current_payload, "historical_anchor_state", device)
    native_current = _anchors(current_payload, "current_anchor_state", device)
    historical_equal = bool(torch.equal(old_anchors.cpu(), historical.cpu()))
    if not historical_equal:
        raise RuntimeError("current checkpoint historical anchors are not byte-identical to the old site anchors")
    loader = labeled_loader(
        data_root,
        seed=seed,
        site_id=SITES[current_site_index],
        roles=("train_labeled",),
        workers=workers,
    )
    old_case, current_case, case_ids = paired_case_prototypes(old_method.model, current_method.model, loader, device=device)
    estimate = estimate_all_class_transport(old_case, current_case)
    global_anchors = transport_anchors(old_anchors, estimate.global_delta)
    class_anchors = transport_anchors(old_anchors, estimate.class_deltas)
    bundle = TransitionBundle(
        seed=seed,
        old_site_index=old_site_index,
        current_site_index=current_site_index,
        old_site_id=SITES[old_site_index],
        current_site_id=SITES[current_site_index],
        old_checkpoint=old_checkpoint,
        current_checkpoint=current_checkpoint,
        old_anchors=old_anchors.detach(),
        native_current_anchors=native_current.detach(),
        global_anchors=global_anchors.detach(),
        class_anchors=class_anchors.detach(),
        transport=estimate,
        source_case_ids=case_ids,
        source_pixel_counts=current_case.pixel_counts.detach(),
        historical_anchor_equal=historical_equal,
    )
    return bundle, old_method.model, current_method.model


def relation_probabilities(features: torch.Tensor, anchors: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    view = anchors[:, 0] if anchors.ndim == 3 else anchors
    feature_unit = F.normalize(features.float(), p=2, dim=1, eps=1.0e-8)
    anchor_unit = F.normalize(view.float(), p=2, dim=1, eps=1.0e-8)
    logits = torch.einsum("bdhw,cd->bchw", feature_unit, anchor_unit) / float(temperature)
    probability = logits.softmax(dim=1)
    if not torch.isfinite(probability).all():
        raise FloatingPointError("non-finite TARC relation probability")
    return probability


def tensor_bundle(bundle: TransitionBundle) -> dict[str, Any]:
    return {
        "protocol_id": "tarcseg_v0_1",
        "seed": bundle.seed,
        "old_site_index": bundle.old_site_index,
        "current_site_index": bundle.current_site_index,
        "old_site_id": bundle.old_site_id,
        "current_site_id": bundle.current_site_id,
        "old_checkpoint": str(bundle.old_checkpoint),
        "current_checkpoint": str(bundle.current_checkpoint),
        "old_anchors": bundle.old_anchors.detach().cpu(),
        "native_current_anchors": bundle.native_current_anchors.detach().cpu(),
        "global_anchors": bundle.global_anchors.detach().cpu(),
        "class_anchors": bundle.class_anchors.detach().cpu(),
        "class_deltas": bundle.transport.class_deltas.detach().cpu(),
        "global_delta": bundle.transport.global_delta.detach().cpu(),
        "class_shrinkage": torch.tensor([item.shrinkage for item in bundle.transport.class_estimates]),
        "global_shrinkage": bundle.transport.global_estimate.shrinkage,
        "paired_case_counts": bundle.transport.paired_case_counts.detach().cpu(),
        "global_case_count": bundle.transport.global_case_count,
        "source_case_ids": bundle.source_case_ids,
        "source_pixel_counts": bundle.source_pixel_counts.detach().cpu(),
        "historical_anchor_equal": bundle.historical_anchor_equal,
        "minimum_relation_pixels_per_case_class": MINIMUM_PIXELS,
        "transport_classes": [0, 1, 2],
        "hidden_gt_usage": "none",
    }
