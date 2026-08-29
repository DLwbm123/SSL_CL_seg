from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _official_probabilistic_classifier(reference_root: str | Path, *, upstream_path: str):
    source_root = (Path(reference_root) / upstream_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    deeplab = importlib.import_module("models.deeplabv3.deeplab")
    return deeplab.ProbabilisticClassifier


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """The frozen LCRSeg UNet2D double-convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UpBlock(nn.Module):
    """The frozen LCRSeg UNet2D transposed-convolution decoder block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.merge = ConvNormAct(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            raise ValueError(f"U-Net geometry mismatch: up={value.shape[-2:]}, skip={skip.shape[-2:]}")
        return self.merge(torch.cat((value, skip), dim=1))


class LCRSegUNetDecoder(nn.Module):
    """LCRSeg decoder with the official JASCL stochastic 3x3 classifier."""

    def __init__(self, classifier_type, num_classes: int) -> None:
        super().__init__()
        self.dec3 = UpBlock(128, 64, 64)
        self.dec2 = UpBlock(64, 32, 32)
        self.dec1 = UpBlock(32, 16, 16)
        self.conv_logit = classifier_type(16, int(num_classes), kernel_size=3, padding=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        enc3: torch.Tensor,
        enc2: torch.Tensor,
        enc1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        return self.conv_logit(dec1), dec1


class LCRSegUNet2DJASCL(nn.Module):
    """Frozen medical UNet2D body plus the official JASCL 3x3 GAS head.

    The body is the LCRSeg UNet2D (channels 16/32/64/128, GroupNorm). The
    classifier is instantiated directly from the official pinned JASCL source.
    The official classifier omits padding in its functional convolution, so its
    low-resolution logits are interpolated back to the input geometry exactly as
    the official DeepLab wrapper does.
    """

    def __init__(self, in_channels: int, num_classes: int, classifier_type) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.enc1 = ConvNormAct(self.in_channels, 16)
        self.enc2 = ConvNormAct(16, 32)
        self.enc3 = ConvNormAct(32, 64)
        self.bottleneck = ConvNormAct(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = LCRSegUNetDecoder(classifier_type, self.num_classes)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {image.shape[1]}")
        height, width = image.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(f"input H/W must be divisible by 8, got {(height, width)}")
        enc1 = self.enc1(image)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        logits, features = self.decoder(bottleneck, enc3, enc2, enc1)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=True)
        return logits, features


def build_lcrseg_unet_jascl_model(
    reference_root: str | Path,
    *,
    upstream_path: str,
    input_channels: int,
    num_classes: int,
) -> nn.Module:
    classifier_type = _official_probabilistic_classifier(reference_root, upstream_path=upstream_path)
    return LCRSegUNet2DJASCL(int(input_channels), int(num_classes), classifier_type)


class RepairedMeanTeacher(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.freeze_teacher()
        self.teacher.eval()

    def freeze_teacher(self) -> None:
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, alpha: float) -> None:
        for student_parameter, teacher_parameter in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_parameter.mul_(alpha).add_(student_parameter, alpha=1.0 - alpha)

    def assert_optimizer_excludes_teacher(self, optimizer: torch.optim.Optimizer) -> None:
        teacher_ids = {id(parameter) for parameter in self.teacher.parameters()}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        overlap = teacher_ids.intersection(optimizer_ids)
        if overlap:
            raise RuntimeError("teacher parameter entered optimizer")


def build_mean_teacher(
    reference_root: str | Path,
    *,
    upstream_path: str,
    input_channels: int,
    num_classes: int,
    device: torch.device,
    factory: Callable[[], nn.Module] | None = None,
) -> RepairedMeanTeacher:
    builder = factory or (
        lambda: build_lcrseg_unet_jascl_model(
            reference_root,
            upstream_path=upstream_path,
            input_channels=input_channels,
            num_classes=num_classes,
        )
    )
    student = builder().to(device)
    teacher = builder().to(device)
    wrapper = RepairedMeanTeacher(student, teacher).to(device)
    return wrapper


def classifier_gas_state(student: nn.Module) -> dict[str, torch.Tensor]:
    classifier = student.decoder.conv_logit
    if not hasattr(classifier, "grad_update"):
        raise RuntimeError("official stochastic 3x3 classifier has no GAS grad_update state")
    return {"grad_update": classifier.grad_update.detach().cpu().clone()}


@torch.no_grad()
def update_gas_from_supervised_gradient(student: nn.Module) -> None:
    classifier = student.decoder.conv_logit
    if not hasattr(classifier, "mu") or classifier.mu.weight.grad is None:
        raise RuntimeError("classifier gradient is unavailable for GAS update")
    classifier.grad_update.copy_(classifier.mu.weight.grad.detach().square())


@torch.no_grad()
def restore_gas_state(student: nn.Module, gas_state: dict[str, torch.Tensor]) -> None:
    classifier = student.decoder.conv_logit
    classifier.grad_update.copy_(gas_state["grad_update"].to(classifier.grad_update.device))


@torch.no_grad()
def compute_single_prototypes(
    student: nn.Module,
    batches: list[dict],
    *,
    num_classes: int,
    device: torch.device,
    ignore_label: int,
) -> torch.Tensor:
    was_training = student.training
    feature_sums: torch.Tensor | None = None
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for batch in batches:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        _, features = student(images)
        labels_down = F.interpolate(labels[:, None].float(), features.shape[-2:], mode="nearest").squeeze(1).long()
        flattened = F.normalize(features.float(), dim=1).permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        labels_flat = labels_down.reshape(-1)
        if feature_sums is None:
            feature_sums = torch.zeros(num_classes, features.shape[1], dtype=torch.float64, device=device)
        for class_id in range(num_classes):
            mask = (labels_flat == class_id) & (labels_flat != ignore_label)
            if mask.any():
                feature_sums[class_id].add_(flattened[mask].double().sum(dim=0))
                counts[class_id].add_(int(mask.sum()))
    if was_training:
        student.train()
    if feature_sums is None or (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"cannot build JASCL PAS prototype for empty classes: {missing}")
    return F.normalize((feature_sums / counts[:, None]).float(), dim=1)


@torch.no_grad()
def upstream_pas_labels(
    logits: torch.Tensor,
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    confidence_threshold: float,
    similarity_threshold: float,
    invalid_token: int,
) -> torch.Tensor:
    low_logits = F.interpolate(logits, features.shape[-2:], mode="bilinear", align_corners=False)
    probabilities = F.softmax(low_logits.float(), dim=1)
    confidence, labels = probabilities.max(dim=1)
    normalized_features = F.normalize(features.float(), dim=1)
    similarity = torch.zeros_like(confidence)
    for class_id in range(logits.shape[1]):
        mask = labels == class_id
        if mask.any():
            pixels = normalized_features.permute(0, 2, 3, 1)[mask]
            similarity[mask] = F.cosine_similarity(pixels, prototypes[class_id][None], dim=1)
    keep = (confidence > confidence_threshold) & (similarity > similarity_threshold)
    filtered = labels.clone()
    filtered[~keep] = invalid_token
    return F.interpolate(filtered[:, None].float(), logits.shape[-2:], mode="nearest").squeeze(1)


def assert_complete_classifier_load(state_dict: dict[str, torch.Tensor], student: nn.Module) -> None:
    expected = {key for key in student.state_dict() if "decoder.conv_logit" in key}
    supplied = {key for key in state_dict if "decoder.conv_logit" in key}
    if expected != supplied:
        raise RuntimeError(f"classifier state is incomplete: missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}")
