"""Reusable site-by-site runner for LCR-Seg and every baseline.

No method owns a training loop.  This runner builds the same frozen-manifest
datasets, deterministic batch schedules, optimizer/scheduler, evaluator, and
checkpoint payload for every method.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from ..common import canonical_json, sha256_path, write_csv, write_json, write_text
from ..data import DeterministicBatcher, H5LabeledDataset, H5UnlabeledDataset, LabeledTransform, WeakStrongTransform, collate_labeled, collate_unlabeled
from ..methods import build_method, resolve_method_config
from ..methods.base import ContinualSegMethod
from ..models import UNet2D
from .checkpoint import checkpoint_payload, load_checkpoint, restore_rng_state, save_checkpoint
from .evaluator import EvaluationResult, evaluate_sites
from .trainer import Trainer, TrainerState, build_optimizer, build_scheduler, seed_everything


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "fundus": {
        "channels": 3,
        "classes": 3,
        "sites": ("REFUGE", "RIM_ONE_r3", "Drishti_GS"),
    },
    "prostate": {
        "channels": 1,
        "classes": 2,
        "sites": ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"),
        "default_train_sites": ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC"),
    },
    "mnms": {
        "channels": 1,
        "classes": 4,
        "sites": ("Siemens", "Philips", "GE", "Canon"),
    },
}

_SUPERVISED_METHODS = {"static_sup", "static_supervised", "finetune_sup", "fine_tune_sup", "joint_sup"}
_STATIC_METHODS = {"static_sup", "static_supervised", "static_ssl"}
_JOINT_METHODS = {"joint_sup", "joint_ssl"}


def normalized_method_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def git_commit_or_sentinel(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "NO_GIT_WORKTREE"
    return result.stdout.strip() or "NO_GIT_WORKTREE"


def _runtime_environment(device: torch.device) -> str:
    gpu = "cpu"
    gpu_uuid = ""
    driver = ""
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ]
            if physical:
                command.extend(["--id", physical])
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            first = result.stdout.strip().splitlines()[0]
            gpu_uuid, driver = (item.strip() for item in first.split(",", maxsplit=1))
        except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
            gpu_uuid = "unavailable"
            driver = "unavailable"
    return "\n".join(
        (
            f"python={platform.python_version()}",
            f"torch={torch.__version__}",
            f"cuda={torch.version.cuda}",
            f"cudnn={torch.backends.cudnn.version()}",
            f"device={device}",
            f"gpu={gpu}",
            f"gpu_uuid={gpu_uuid}",
            f"driver={driver}",
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        )
    ) + "\n"


def _json_yaml(value: Mapping[str, Any]) -> str:
    """JSON is valid YAML 1.2 and avoids a runtime PyYAML dependency."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _log_scalar(value: Any) -> Any:
    """Keep V0.2's structured, non-patient routing diagnostics in CSV rows."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("training scalar tensor must contain exactly one value")
        return float(value.detach())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
        return value
    raise TypeError(f"unsupported training-log scalar type: {type(value)!r}")


@dataclass
class AnalysisAccumulator:
    bins: int = 20
    learnability_hist: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.int64))
    compatibility_hist: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.int64))
    quadrants: dict[str, int] = field(default_factory=lambda: {"high_l_high_c": 0, "high_l_low_c": 0, "low_l_high_c": 0, "low_l_low_c": 0})
    pseudo_sources: dict[str, int] = field(default_factory=lambda: {"classifier": 0, "anchor": 0, "deferred": 0})

    def add_maps(self, maps: Mapping[str, torch.Tensor] | None) -> None:
        if not maps:
            return
        if "learnability" in maps:
            learnability = maps["learnability"].detach().float().cpu().numpy().reshape(-1)
            self.learnability_hist += np.histogram(learnability, bins=self.bins, range=(0.0, 1.0))[0]
        else:
            learnability = None
        if "compatibility" in maps:
            compatibility = maps["compatibility"].detach().float().cpu().numpy().reshape(-1)
            self.compatibility_hist += np.histogram(compatibility, bins=self.bins, range=(0.0, 1.0))[0]
        else:
            compatibility = None
        if learnability is not None and compatibility is not None and learnability.shape == compatibility.shape:
            high_l = learnability >= 0.5
            high_c = compatibility >= 0.5
            self.quadrants["high_l_high_c"] += int((high_l & high_c).sum())
            self.quadrants["high_l_low_c"] += int((high_l & ~high_c).sum())
            self.quadrants["low_l_high_c"] += int((~high_l & high_c).sum())
            self.quadrants["low_l_low_c"] += int((~high_l & ~high_c).sum())
        if "pseudo_source" in maps:
            source = maps["pseudo_source"].detach()
            self.pseudo_sources["classifier"] += int(source.eq(1).sum())
            self.pseudo_sources["anchor"] += int(source.eq(2).sum())
            self.pseudo_sources["deferred"] += int(source.eq(0).sum())

    def write(self, output_dir: Path) -> None:
        analysis = output_dir / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(self.bins):
            rows.append(
                {
                    "bin_start": index / self.bins,
                    "bin_end": (index + 1) / self.bins,
                    "learnability_count": int(self.learnability_hist[index]),
                    "compatibility_count": int(self.compatibility_hist[index]),
                }
            )
        write_csv(analysis / "reliability_histograms.csv", rows)
        write_csv(analysis / "quadrant_stats.csv", [{"quadrant": name, "pixel_count": count} for name, count in self.quadrants.items()])
        write_csv(analysis / "pseudo_source_counts.csv", [{"source": name, "pixel_count": count} for name, count in self.pseudo_sources.items()])


@dataclass
class SiteRunResult:
    site_id: str
    checkpoint_final: Path
    completed_steps: int
    evaluation: EvaluationResult | None


class ContinualRunner:
    """Single source of truth for baseline and proposed experiment execution."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = json.loads(json.dumps(dict(config)))
        self.dataset = str(self.config["data"]["dataset"])
        if self.dataset not in DATASET_SPECS:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        self.data_root = Path(self.config["data"]["data_root"]).resolve()
        self.run_root = Path(self.config["experiment"]["run_root"]).resolve()
        self.method_name = normalized_method_name(str(self.config["method"]["name"]))
        # Persist fully expanded method defaults in config.yaml and every
        # checkpoint.  This is intentionally done before provenance is
        # written, not only inside the method constructor.
        resolved_method = resolve_method_config(self.method_name, self.config.get("method", {}))
        resolved_method["name"] = self.method_name
        resolved_method["version"] = str(self.config["method"].get("version", "0.1"))
        self.config["method"] = resolved_method
        initial_previous = self.config["experiment"].get("initial_previous_checkpoint")
        self.initial_previous_checkpoint = Path(initial_previous).resolve() if initial_previous else None
        if self.initial_previous_checkpoint is not None and not self.initial_previous_checkpoint.is_file():
            raise FileNotFoundError(f"initial previous checkpoint is missing: {self.initial_previous_checkpoint}")
        self.site_index_offset = int(self.config["data"].get("site_index_offset", 0))
        if self.site_index_offset < 0:
            raise ValueError("site_index_offset must be nonnegative")
        self.seed = int(self.config["experiment"].get("seed", 0))
        # Diagnostic protocols may keep the frozen data split fixed while
        # varying only initialization, ordering, and augmentation randomness.
        # Formal configurations omit this field and therefore retain their
        # historical seed behavior exactly.
        self.optimization_seed = int(self.config["experiment"].get("optimization_seed", self.seed))
        self.device = torch.device(self.config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.spec = DATASET_SPECS[self.dataset]
        configured_sites = tuple(self.config["data"].get("site_order") or self.spec.get("default_train_sites") or self.spec["sites"])
        self.site_order = tuple(str(site) for site in configured_sites)
        if not self.site_order:
            raise ValueError("site order cannot be empty")
        self.evaluation_sites = tuple(str(site) for site in self.config["data"].get("evaluation_sites") or self.spec["sites"])
        unknown = set(self.site_order).difference(self.spec["sites"])
        if unknown:
            raise ValueError(f"unknown sites for {self.dataset}: {sorted(unknown)}")
        self.project_root = Path(__file__).resolve().parents[2]
        self.manifest_path = self.data_root / "manifests" / "training" / f"lcrseg_v1_seed{self.seed}.csv"
        self.split_path = self.data_root / "splits" / f"{self.dataset}_seed{self.seed}.json"
        if not self.manifest_path.is_file() or not self.split_path.is_file():
            raise FileNotFoundError("frozen manifest or split file is missing")
        self._verify_frozen_boundaries()

    def _verify_frozen_boundaries(self) -> None:
        frozen = (
            self.data_root / "h5" / "v1",
            self.data_root / "manifests",
            self.data_root / "splits",
            self.data_root / "checksums",
        )
        if not (frozen[0] / "FROZEN").is_file():
            raise RuntimeError("frozen HDF5 marker is missing")
        resolved_run_root = self.run_root.resolve()
        for path in frozen:
            resolved = path.resolve()
            if resolved_run_root == resolved or resolved in resolved_run_root.parents:
                raise ValueError(f"run root may not be located inside frozen input: {resolved}")
        if bool(self.config["data"].get("require_readonly", False)):
            writable = [
                str(path)
                for path in frozen
                if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            ]
            if writable:
                raise RuntimeError(f"frozen inputs are writable: {writable}")

    @staticmethod
    def default_config(
        *,
        data_root: Path,
        run_root: Path,
        dataset: str,
        method_name: str,
        seed: int,
        site_order: Iterable[str] | None = None,
        run_name: str,
        device: str | None = None,
    ) -> dict[str, Any]:
        spec = DATASET_SPECS[dataset]
        return {
            "experiment": {
                "run_name": run_name,
                "run_root": str(run_root),
                "seed": int(seed),
                "deterministic": True,
                "final_checkpoint_only": True,
                "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            },
            "data": {
                "data_root": str(data_root),
                "dataset": dataset,
                "site_order": list(site_order or spec.get("default_train_sites") or spec["sites"]),
                "evaluation_sites": list(spec["sites"]),
                "preprocess_version": "v1",
                "evaluation_role": "test",
                "require_readonly": True,
            },
            "model": {
                "name": "unet2d",
                "base_channels": 16,
                "relation_dim": 128,
                "in_channels": spec["channels"],
                "num_classes": spec["classes"],
            },
            "method": {
                "name": method_name,
                "version": (
                    "0.2"
                    if normalized_method_name(method_name) in {"srgas_v0_2", "lw_srgas"}
                    else
                    "0.1a"
                    if normalized_method_name(method_name) in {"srgas_v0_1", "srgas", "sr_gas"}
                    else
                    "0.3"
                    if normalized_method_name(method_name) in {"lcrseg_v0_3", "lcr_v0_3"}
                    else "0.2a"
                    if normalized_method_name(method_name) in {"lcrseg_v0_2a", "lcr_v0_2a"}
                    else "0.2"
                    if normalized_method_name(method_name) in {"lcrseg_v0_2", "lcr_v0_2"}
                    else "0.1"
                ),
            },
            "training": {
                "epochs_per_site": 5,
                "steps_per_site": None,
                "labeled_batch_size": 2,
                "unlabeled_batch_size": 4,
                "lr": 5.0e-4,
                "weight_decay": 1.0e-5,
                "amp": True,
                "amp_init_scale": 1024.0,
                "grad_clip_norm": None,
                "checkpoint_interval_steps": 0,
                "gradient_cosine_interval": 100,
                "max_steps_this_invocation": None,
                "evaluation_batch_size": 4,
            },
        }

    def _run_dir(self) -> Path:
        run_name = str(self.config["experiment"]["run_name"])
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be a simple directory name")
        return self.run_root / run_name

    def _write_provenance(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_text(run_dir / "config.yaml", _json_yaml(self.config))
        write_text(run_dir / "git_commit.txt", git_commit_or_sentinel(self.project_root) + "\n")
        write_text(run_dir / "environment.txt", _runtime_environment(self.device))
        write_text(run_dir / "manifest_hash.txt", sha256_path(self.manifest_path) + "\n")
        write_text(run_dir / "split_hash.txt", sha256_path(self.split_path) + "\n")
        write_text(run_dir / "command.txt", " ".join(sys.argv) + "\n")
        protocol_id = self.config["method"].get("protocol_id")
        if protocol_id in {"lcrseg_v0_3", "lcrseg_v0_4a", "srgas_v0_1", "srgas_v0_1a", "srgas_v0_2"}:
            write_text(run_dir / "manifest_sha256.txt", sha256_path(self.manifest_path) + "\n")
            write_text(run_dir / "split_sha256.txt", sha256_path(self.split_path) + "\n")
        if protocol_id in {"lcrseg_v0_2a", "lcrseg_v0_3", "lcrseg_v0_4a", "srgas_v0_1", "srgas_v0_1a", "srgas_v0_2"}:
            method = self.config["method"]
            keys = (
                (
                    "protocol_id",
                    "variant_id",
                    "assimilation_mode",
                    "consolidation_mode",
                    "learnability_formula_version",
                    "teacher_validity_formula_version",
                    "calibrator_version",
                    "progressive_schedule",
                    "rejection_threshold",
                    "rejection_floor",
                    "rejection_cap",
                    "formal_r0_run",
                    "auxiliary_u0_run",
                    "u0_registered_as_formal_r0",
                )
                if protocol_id == "lcrseg_v0_2a"
                else (
                    "protocol_id",
                    "variant_id",
                    "srgas_variant",
                    "srgas_behavior_variant",
                    "relation_conditioning",
                    "cosine_temperature",
                    "cosine_eps",
                    "gas_epsilon",
                    "noise_variance_max",
                    "noise_warmup_fraction",
                    "noise_warm_start",
                    "sensitivity_timing",
                    "shared_noise_stream",
                    "protocol_seed",
                    "split_seed",
                    "r2c_formula_version",
                    "r2c_temperature",
                    "r2c_resize_mode",
                    "r2c_align_corners",
                    "r2c_reduction",
                    "r2c_source_weight",
                    "supervised_source_weight",
                    "r2c_added_to_training_objective",
                    "channel_mapping",
                    "architecture_change",
                    "shuffle_r2c_target",
                )
                if protocol_id == "srgas_v0_2"
                else (
                    "protocol_id",
                    "variant_id",
                    "assimilation_mode",
                    "consolidation_mode",
                    "lambda_relation",
                    "learnability_formula_version",
                    "progressive_schedule",
                    "teacher_rejection_enabled",
                    "multi_agent",
                    "ric",
                    "v02a_r1_semantic_reference",
                )
                if protocol_id == "lcrseg_v0_3"
                else (
                    "protocol_id",
                    "variant_id",
                    "assimilation_mode",
                    "consolidation_mode",
                    "single_anchor",
                    "multi_agent",
                    "ric",
                    "teacher_rejection",
                    "lambda_assim",
                    "lambda_relation",
                    "learnability_formula_version",
                    "rank_schedule",
                    "soft_allocation",
                    "historical_relation_path",
                )
                if protocol_id == "lcrseg_v0_4a"
                else (
                    "protocol_id",
                    "variant_id",
                    "srgas_variant",
                    "relation_conditioning",
                    "cosine_temperature",
                    "cosine_eps",
                    "gas_epsilon",
                    "noise_variance",
                    "noise_distribution",
                    "noise_scope",
                    "scale_normalization",
                    "same_step_sensitivity",
                    "r2c_temperature",
                    "r2c_resize_mode",
                    "r2c_align_corners",
                    "r2c_reduction",
                    "r2c_source_weight",
                    "supervised_source_weight",
                    "r2c_added_to_training_objective",
                    "channel_mapping",
                    "architecture_change",
                    "shuffle_r2c_target",
                )
            )
            protocol_fields = {key: method[key] for key in keys}
            write_json(run_dir / "protocol.json", protocol_fields)
            parent = {
                "initial_previous_checkpoint": str(self.initial_previous_checkpoint or ""),
                "initial_previous_checkpoint_sha256": (
                    sha256_path(self.initial_previous_checkpoint) if self.initial_previous_checkpoint is not None else ""
                ),
                "formal_r0_run": method.get("formal_r0_run", ""),
                "auxiliary_u0_run": method.get("auxiliary_u0_run", ""),
                "initial_previous_run": str(self.config["experiment"].get("initial_previous_run", "")),
                "completed_parent_steps": int(self.config["experiment"].get("initial_global_step", 0)),
            }
            write_json(run_dir / "parent_artifact.json", parent)

    def _inherit_initial_artifacts(self, run_dir: Path) -> None:
        """Inherit only proven-equivalent completed-site evidence.

        This is used by V0.3 P0 after its exact REFUGE bridge.  It does not
        copy training rows or mutate the parent; it preserves the parent
        checkpoint by hard-link and copies only frozen evaluation/coverage
        records for site indices preceding ``site_index_offset``.
        """

        if not bool(self.config["experiment"].get("inherit_completed_site_artifacts", False)):
            return
        parent_value = self.config["experiment"].get("initial_previous_run")
        if not parent_value or self.initial_previous_checkpoint is None or self.site_index_offset < 1:
            raise ValueError("artifact inheritance requires a parent run, checkpoint, and positive site offset")
        parent = Path(parent_value).resolve()
        if not parent.is_dir() or self.initial_previous_checkpoint.parent.resolve() != parent:
            raise ValueError("initial checkpoint must belong to the declared parent run")
        inherited: dict[str, Any] = {
            "parent_run": str(parent),
            "parent_checkpoint": str(self.initial_previous_checkpoint),
            "parent_checkpoint_sha256": sha256_path(self.initial_previous_checkpoint),
            "site_index_offset": self.site_index_offset,
            "files": [],
        }

        def filtered_rows(filename: str, index_field: str) -> list[dict[str, Any]]:
            source = parent / filename
            if not source.is_file():
                raise FileNotFoundError(f"parent artifact is missing: {source}")
            rows = list(csv.DictReader(source.open()))
            selected = [row for row in rows if int(row[index_field]) < self.site_index_offset]
            if not selected:
                raise RuntimeError(f"parent artifact has no inheritable rows: {source}")
            write_csv(run_dir / filename, selected, fieldnames=list(selected[0]))
            inherited["files"].append({"path": filename, "rows": len(selected), "source_sha256": sha256_path(source)})
            return selected

        filtered_rows("site_matrix_long.csv", "trained_site_index")
        filtered_rows("per_case_metrics.csv", "trained_site_index")
        branch = parent / "branch_coverage.csv"
        if branch.is_file():
            rows = [
                row
                for row in csv.DictReader(branch.open())
                if int(row["site_index"]) < self.site_index_offset
            ]
            if rows:
                write_csv(run_dir / "branch_coverage.csv", rows, fieldnames=list(rows[0]))
                inherited["files"].append(
                    {"path": "branch_coverage.csv", "rows": len(rows), "source_sha256": sha256_path(branch)}
                )
        checkpoint_target = run_dir / self.initial_previous_checkpoint.name
        if checkpoint_target.exists():
            raise FileExistsError(checkpoint_target)
        os.link(self.initial_previous_checkpoint, checkpoint_target)
        inherited["files"].append(
            {"path": checkpoint_target.name, "hardlink": True, "source_sha256": sha256_path(self.initial_previous_checkpoint)}
        )
        checkpoint_alias = run_dir / f"checkpoint_site_{self.site_index_offset - 1}_{self.initial_previous_checkpoint.stem.split('_', maxsplit=3)[-1]}.pt"
        if checkpoint_alias.exists():
            raise FileExistsError(checkpoint_alias)
        os.link(self.initial_previous_checkpoint, checkpoint_alias)
        inherited["files"].append(
            {"path": checkpoint_alias.name, "hardlink": True, "source_sha256": sha256_path(self.initial_previous_checkpoint)}
        )
        for summary in sorted(parent.glob("site_summary_*.json")):
            try:
                index = int(summary.name.split("_", maxsplit=2)[2].split("_", maxsplit=1)[0])
            except (IndexError, ValueError):
                continue
            if index >= self.site_index_offset:
                continue
            destination = run_dir / summary.name
            destination.write_bytes(summary.read_bytes())
            inherited["files"].append({"path": summary.name, "source_sha256": sha256_path(summary)})
        write_json(run_dir / "inherited_artifacts.json", inherited)

    def _build_method(self) -> ContinualSegMethod:
        model = UNet2D(
            int(self.config["model"]["in_channels"]),
            int(self.config["model"]["num_classes"]),
            base_channels=int(self.config["model"].get("base_channels", 16)),
            relation_dim=int(self.config["model"].get("relation_dim", 128)),
        ).to(self.device)
        return build_method(self.method_name, model, config=self.config.get("method", {})).to(self.device)

    def _method_needs_unlabeled(self) -> bool:
        return self.method_name not in _SUPERVISED_METHODS

    def _site_data_scope(self, site_index: int, site_id: str) -> tuple[str, ...]:
        if self.method_name in _JOINT_METHODS:
            scope = str(self.config["training"].get("joint_scope", "all"))
            return self.site_order if scope == "all" else self.site_order[: site_index + 1]
        return (site_id,)

    def _datasets(self, scope: tuple[str, ...]):
        transforms = self.config.get("transforms", {})
        labeled = H5LabeledDataset(
            self.data_root,
            seed=self.seed,
            dataset=self.dataset,
            sites=scope,
            transform=LabeledTransform(flip_probability=float(transforms.get("labeled_flip_probability", 0.5))),
        )
        unlabeled = None
        if self._method_needs_unlabeled():
            unlabeled = H5UnlabeledDataset(
                self.data_root,
                seed=self.seed,
                dataset=self.dataset,
                sites=scope,
                transform=WeakStrongTransform(
                    flip_probability=float(transforms.get("flip_probability", 0.5)),
                    strong_noise_std=float(transforms.get("strong_noise_std", 0.03)),
                    brightness_delta=float(transforms.get("brightness_delta", 0.10)),
                    contrast_delta=float(transforms.get("contrast_delta", 0.10)),
                    cutout_probability=float(transforms.get("cutout_probability", 0.5)),
                    cutout_fraction=float(transforms.get("cutout_fraction", 0.20)),
                ),
            )
        return labeled, unlabeled

    def _reference_unlabeled_steps(self, scope: tuple[str, ...]) -> int:
        """Count the image-only SSL schedule without yielding a batch.

        Supervised controls never receive this dataset in their training loop.
        Its length is used solely to give them the same number of optimizer
        updates as an SSL method under the frozen protocol.
        """

        reference = H5UnlabeledDataset(
            self.data_root,
            seed=self.seed,
            dataset=self.dataset,
            sites=scope,
            transform=None,
        )
        return int(math.ceil(len(reference) / int(self.config["training"]["unlabeled_batch_size"])))

    def _joint_equivalent_total_steps(self) -> int:
        """Return the exact sum of sequential per-site training budgets.

        The merged joint dataset can have fewer batches than the sum of the
        individual schedules because the final partial batches are combined.
        JointTrain must nevertheless receive the same *optimization-step*
        budget as the corresponding sequential protocol, including those
        per-site rounding effects.
        """

        training = self.config["training"]
        scope = self._site_data_scope(0, "JOINT_ALL")
        override = training.get("steps_per_site")
        if override not in (None, "", 0):
            total = int(override) * len(scope)
            if total < 1:
                raise ValueError("configured joint training has zero steps")
            return total
        labeled_batch_size = int(training["labeled_batch_size"])
        unlabeled_batch_size = int(training["unlabeled_batch_size"])
        total = 0
        for site_id in scope:
            labeled, unlabeled = self._datasets((site_id,))
            labeled_steps = int(math.ceil(len(labeled) / labeled_batch_size))
            unlabeled_steps = (
                int(math.ceil(len(unlabeled) / unlabeled_batch_size))
                if unlabeled is not None
                else self._reference_unlabeled_steps((site_id,))
            )
            total += int(training["epochs_per_site"]) * max(labeled_steps, unlabeled_steps)
        if total < 1:
            raise ValueError("configured joint training has zero steps")
        return total

    def _total_steps(
        self,
        labeled_batcher: DeterministicBatcher,
        unlabeled_batcher: DeterministicBatcher | None,
        *,
        scope: tuple[str, ...],
    ) -> tuple[int, int]:
        training = self.config["training"]
        reference_steps = self._reference_unlabeled_steps(scope) if unlabeled_batcher is None and self.method_name in _SUPERVISED_METHODS else 0
        steps_per_epoch = max(labeled_batcher.steps_per_epoch, unlabeled_batcher.steps_per_epoch if unlabeled_batcher else 0, reference_steps)
        override = training.get("steps_per_site")
        total = int(override) if override not in (None, "", 0) else int(training["epochs_per_site"]) * steps_per_epoch
        if total < 1:
            raise ValueError("configured site training has zero steps")
        if self.method_name in _JOINT_METHODS:
            total = self._joint_equivalent_total_steps()
        return total, steps_per_epoch

    def _checkpoint(
        self,
        path: Path,
        *,
        method: ContinualSegMethod,
        trainer: Trainer,
        site_id: str,
        site_index: int,
        epoch: int,
        completed_site_steps: int,
        completed_global_steps: int,
    ) -> None:
        method_state = method.method_state_dict()
        payload = checkpoint_payload(
            method_name=method.method_name,
            method_version=method.method_version,
            git_commit=git_commit_or_sentinel(self.project_root),
            config_resolved=self.config,
            site_id=site_id,
            site_index=site_index,
            epoch=epoch,
            site_step=completed_site_steps,
            global_step=completed_global_steps,
            current_model_state=method.model.state_dict(),
            optimizer_state=trainer.optimizer.state_dict(),
            scheduler_state=trainer.scheduler.state_dict(),
            scaler_state=trainer.scaler.state_dict(),
            current_anchor_state=method_state["current_anchor_state"],
            historical_anchor_state=method_state["historical_anchor_state"],
            bootstrap_state=method_state["bootstrap_state"],
            method_statistics=method_state["method_statistics"],
            data_split_hash=sha256_path(self.split_path),
            manifest_hash=sha256_path(self.manifest_path),
            preprocess_version=str(self.config["data"].get("preprocess_version", "v1")),
        )
        save_checkpoint(path, payload)
        # Read back immediately: a failed schema/reload gate is never deferred
        # to the next site.
        load_checkpoint(path, map_location="cpu")

    def _restore_resume(
        self,
        checkpoint: Path,
        *,
        method: ContinualSegMethod,
        trainer: Trainer,
        site_id: str,
        site_index: int,
    ) -> TrainerState:
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if payload["site_id"] != site_id or int(payload["site_index"]) != site_index:
            raise ValueError("resume checkpoint does not belong to the requested current site")
        if payload["method_name"] != method.method_name:
            raise ValueError("resume checkpoint method differs from requested method")
        method.model.load_state_dict(payload["current_model_state"], strict=True)
        method.load_method_state_dict(payload)
        trainer.load_state_dict(payload)
        restore_rng_state(payload)
        return TrainerState(
            global_step=int(payload["global_step"]),
            site_step=int(payload["site_step"]),
            epoch=int(payload["epoch"]),
        )

    @staticmethod
    def _write_train_log(path: Path, rows: list[dict[str, Any]]) -> None:
        if rows:
            write_csv(path, rows, fieldnames=sorted({key for row in rows for key in row}))

    def _write_matrix(self, run_dir: Path, rows: list[dict[str, Any]], *, metric: str, filename: str) -> None:
        by_train: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_train.setdefault(str(row["trained_site"]), {})[str(row["evaluation_site"])] = row.get(metric)
        table = []
        for train_site in by_train:
            table.append({"trained_site": train_site, **{site: by_train[train_site].get(site, "") for site in self.evaluation_sites}})
        write_csv(run_dir / filename, table, fieldnames=["trained_site", *self.evaluation_sites])

    def _write_per_class_metrics(self, run_dir: Path, rows: list[dict[str, Any]]) -> None:
        if self.config["method"].get("protocol_id") != "lcrseg_v0_3":
            return
        output: list[dict[str, Any]] = []
        for row in rows:
            for class_id in range(1, int(self.spec["classes"])):
                output.append(
                    {
                        "trained_site": row["trained_site"],
                        "trained_site_index": row["trained_site_index"],
                        "evaluation_site": row["evaluation_site"],
                        "class_id": class_id,
                        "patients": row.get("patients", ""),
                        "dice": row.get(f"dice_class_{class_id}", ""),
                        "asd": row.get(f"asd_class_{class_id}", ""),
                        "hd95": row.get(f"hd95_class_{class_id}", ""),
                    }
                )
        write_csv(
            run_dir / "per_class_metrics.csv",
            output,
            fieldnames=["trained_site", "trained_site_index", "evaluation_site", "class_id", "patients", "dice", "asd", "hd95"],
        )

    def _summary(self, matrix_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self.method_name in _JOINT_METHODS:
            values = [
                _safe_float(row["mean_foreground_dice"])
                for row in matrix_rows
                if str(row.get("trained_site")) == "JOINT_ALL"
            ]
            values = [value for value in values if value is not None]
            return {
                "trained_sites": ["JOINT_ALL"],
                "joint_all_sites_dice": float(np.mean(values)) if values else None,
                "bwt": None,
                "incoming_dice": None,
                "final_average_dice": float(np.mean(values)) if values else None,
                "previous_site_dice": None,
                "unseen_site_dice": None,
            }
        lookup = {(str(row["trained_site"]), str(row["evaluation_site"])): _safe_float(row["mean_foreground_dice"]) for row in matrix_rows}
        # Prefix-lineage runs (for example V0.3 P0) may inherit completed rows
        # before their configured site_order begins.  evaluation_sites is the
        # frozen full sequence and therefore provides the canonical ordering.
        trained = [site for site in self.evaluation_sites if any(key[0] == site for key in lookup)]
        diagonal = [lookup.get((site, site)) for site in trained]
        diagonal_values = [value for value in diagonal if value is not None]
        summary: dict[str, Any] = {
            "trained_sites": trained,
            "incoming_dice": float(np.mean(diagonal_values)) if diagonal_values else None,
            "final_average_dice": None,
            "bwt": None,
            "previous_site_dice": None,
            "unseen_site_dice": None,
        }
        if trained:
            final_site = trained[-1]
            final_values = [lookup.get((final_site, site)) for site in self.evaluation_sites]
            final_values = [value for value in final_values if value is not None]
            summary["final_average_dice"] = float(np.mean(final_values)) if final_values else None
            bwt_values = [lookup[(final_site, earlier)] - lookup[(earlier, earlier)] for earlier in trained[:-1] if lookup.get((final_site, earlier)) is not None and lookup.get((earlier, earlier)) is not None]
            summary["bwt"] = float(np.mean(bwt_values)) if bwt_values else None
            previous = [lookup.get((site, earlier)) for index, site in enumerate(trained) for earlier in trained[:index]]
            previous = [value for value in previous if value is not None]
            summary["previous_site_dice"] = float(np.mean(previous)) if previous else None
            unseen = [lookup.get((final_site, site)) for site in self.evaluation_sites if site not in trained]
            unseen = [value for value in unseen if value is not None]
            summary["unseen_site_dice"] = float(np.mean(unseen)) if unseen else None
        return summary

    def run(self, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
        seed_everything(
            self.optimization_seed,
            deterministic=bool(self.config["experiment"].get("deterministic", True)),
        )
        run_dir = self._run_dir()
        resume_payload = load_checkpoint(resume_checkpoint, map_location="cpu") if resume_checkpoint is not None else None
        resume_site_index = int(resume_payload["site_index"]) if resume_payload is not None else None
        configured_stage_indices = {
            self.site_index_offset + index for index in range(len(self.site_order))
        }
        if resume_site_index is not None and resume_site_index not in configured_stage_indices:
            raise ValueError("resume checkpoint has a site index outside this experiment")
        if resume_checkpoint is None:
            self._write_provenance(run_dir)
            self._inherit_initial_artifacts(run_dir)
        elif not run_dir.is_dir():
            raise FileNotFoundError("resume requires the original run directory")
        elif resume_checkpoint is not None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            write_text(run_dir / f"resume_config_{timestamp}.yaml", _json_yaml(self.config))
            write_text(run_dir / f"resume_command_{timestamp}.txt", " ".join(sys.argv) + "\n")
        def existing_csv(path: Path) -> list[dict[str, Any]]:
            return list(csv.DictReader(path.open())) if path.is_file() else []

        train_rows: list[dict[str, Any]] = existing_csv(run_dir / "train_log.csv")
        matrix_rows: list[dict[str, Any]] = existing_csv(run_dir / "site_matrix_long.csv")
        per_case_rows: list[dict[str, Any]] = existing_csv(run_dir / "per_case_metrics.csv")
        analysis = AnalysisAccumulator()
        gradient_rows: list[dict[str, Any]] = existing_csv(run_dir / "analysis" / "gradient_cosine.csv")
        pilot_trajectory_rows: list[dict[str, Any]] = existing_csv(run_dir / "pilot_trajectory.csv")
        previous_final: Path | None = self.initial_previous_checkpoint
        shared_method: ContinualSegMethod | None = None
        completed_global = int(self.config["experiment"].get("initial_global_step", 0))
        interrupted = False
        invocation_steps = 0
        stages = (
            [(self.site_index_offset, "JOINT_ALL")]
            if self.method_name in _JOINT_METHODS
            else [(self.site_index_offset + index, site) for index, site in enumerate(self.site_order)]
        )
        for stage_position, (site_index, site_id) in enumerate(stages):
            if resume_site_index is not None and site_index < resume_site_index:
                previous_candidates = sorted(run_dir.glob(f"checkpoint_final_site{site_index}_*.pt"))
                if len(previous_candidates) != 1:
                    raise FileNotFoundError(f"cannot locate exactly one completed checkpoint for resumed site {site_index}")
                previous_final = previous_candidates[0]
                continue
            static = self.method_name in _STATIC_METHODS or self.method_name in _JOINT_METHODS
            if shared_method is None or static:
                method = self._build_method()
                if not static:
                    shared_method = method
            else:
                method = shared_method
            method.set_site_index(site_index)
            scope = self._site_data_scope(site_index, site_id)
            labeled_dataset, unlabeled_dataset = self._datasets(scope)
            labeled_batcher = DeterministicBatcher(
                labeled_dataset,
                batch_size=int(self.config["training"]["labeled_batch_size"]),
                seed=self.optimization_seed,
                namespace=f"{self.dataset}:{site_id}:labeled:{','.join(scope)}",
                collate=collate_labeled,
            )
            calibration_batcher = None
            if bool(getattr(method, "requires_labeled_calibration", False)):
                calibration_dataset = H5LabeledDataset(
                    self.data_root,
                    seed=self.seed,
                    dataset=self.dataset,
                    sites=scope,
                    transform=LabeledTransform(flip_probability=0.0),
                )
                calibration_batcher = DeterministicBatcher(
                    calibration_dataset,
                    batch_size=int(self.config["training"]["labeled_batch_size"]),
                    seed=self.optimization_seed,
                    namespace=f"{self.dataset}:{site_id}:calibration_labeled:{','.join(scope)}",
                    collate=collate_labeled,
                )
            unlabeled_batcher = (
                DeterministicBatcher(
                    unlabeled_dataset,
                    batch_size=int(self.config["training"]["unlabeled_batch_size"]),
                    seed=self.optimization_seed,
                    namespace=f"{self.dataset}:{site_id}:unlabeled:{','.join(scope)}",
                    collate=collate_unlabeled,
                )
                if unlabeled_dataset is not None
                else None
            )
            total_steps, steps_per_epoch = self._total_steps(labeled_batcher, unlabeled_batcher, scope=scope)
            previous_for_site = None if static else previous_final
            method.begin_site(site_id, previous_for_site, total_steps)
            optimizer = build_optimizer(method, lr=float(self.config["training"]["lr"]), weight_decay=float(self.config["training"]["weight_decay"]))
            scheduler = build_scheduler(optimizer, total_steps=total_steps)
            trainer = Trainer(
                method,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                amp=bool(self.config["training"].get("amp", True)),
                amp_init_scale=float(self.config["training"].get("amp_init_scale", 1024.0)),
                grad_clip_norm=self.config["training"].get("grad_clip_norm"),
            )
            state = TrainerState(global_step=completed_global, site_step=0, epoch=0)
            if resume_checkpoint is not None and site_index == resume_site_index:
                state = self._restore_resume(resume_checkpoint, method=method, trainer=trainer, site_id=site_id, site_index=site_index)
                completed_global = state.global_step
            if hasattr(method, "calibrate_at_site_start"):
                if bool(getattr(method, "requires_labeled_calibration", False)) and calibration_batcher is None:
                    raise RuntimeError("method requested site-start calibration without a labeled-only batcher")
                method.calibrate_at_site_start(  # type: ignore[attr-defined]
                    calibration_batcher=calibration_batcher,
                    device=self.device,
                    run_dir=run_dir,
                )
            checkpoint_interval = int(self.config["training"].get("checkpoint_interval_steps") or 0)
            maximum = self.config["training"].get("max_steps_this_invocation")
            for site_step in range(state.site_step, total_steps):
                state.site_step = site_step
                state.epoch = site_step // max(1, steps_per_epoch)
                labeled_batch = labeled_batcher.batch_at(site_step)
                unlabeled_batch = unlabeled_batcher.batch_at(site_step) if unlabeled_batcher is not None else None
                result: Any = None
                try:
                    if hasattr(method, "set_training_context"):
                        method.set_training_context(epoch=state.epoch, steps_per_epoch=steps_per_epoch)  # type: ignore[attr-defined]
                    interval = int(self.config["training"].get("gradient_cosine_interval") or 0)
                    result = trainer.train_step(
                        labeled_batch,
                        unlabeled_batch,
                        state=state,
                        collect_gradient_cosine=bool(interval and state.global_step % interval == 0),
                    )
                except BaseException as exc:
                    trainer.failure_bundle(run_dir, error=exc, labeled_batch=labeled_batch, unlabeled_batch=unlabeled_batch, state=state, result=result)
                    self._write_train_log(run_dir / "train_log.csv", train_rows)
                    raise
                row = {
                    "site_id": site_id,
                    "site_index": site_index,
                    "epoch": state.epoch,
                    "site_step": site_step + 1,
                    "global_step": state.global_step + 1,
                    "lr": float(result.scalars.get("lr", optimizer.param_groups[0]["lr"])),
                    "loss_total": float(result.total_loss.detach()),
                    **{key: float(value.detach()) for key, value in result.losses.items()},
                    **{key: _log_scalar(value) for key, value in result.scalars.items()},
                }
                train_rows.append(row)
                analysis.add_maps(result.maps)
                pilot_interval = int(self.config["training"].get("pilot_trajectory_interval") or 0)
                if (
                    pilot_interval
                    and self.config["method"].get("protocol_id") in {"srgas_v0_1", "srgas_v0_1a", "srgas_v0_2"}
                    and (site_step + 1) % pilot_interval == 0
                ):
                    trajectory = evaluate_sites(
                        method.model,
                        data_root=self.data_root,
                        seed=self.seed,
                        dataset=self.dataset,
                        sites=("REFUGE", "RIM_ONE_r3"),
                        num_classes=int(self.spec["classes"]),
                        role=str(self.config["data"].get("evaluation_role", "val")),
                        device=self.device,
                        batch_size=int(self.config["training"].get("evaluation_batch_size", 4)),
                    )
                    trajectory_lookup = {str(item["site"]): item for item in trajectory.per_site}
                    pilot_trajectory_rows.append(
                        {
                            "variant": self.config["method"]["srgas_variant"],
                            "site_step": site_step + 1,
                            "global_step": state.global_step + 1,
                            "refuge_mean_foreground_dice": trajectory_lookup["REFUGE"]["mean_foreground_dice"],
                            "rim_one_mean_foreground_dice": trajectory_lookup["RIM_ONE_r3"]["mean_foreground_dice"],
                        }
                    )
                    write_csv(run_dir / "pilot_trajectory.csv", pilot_trajectory_rows)
                if "gradient_cosine_assim_relation" in result.scalars:
                    gradient_rows.append(
                        {
                            "site_id": site_id,
                            "site_step": site_step + 1,
                            "global_step": state.global_step + 1,
                            "gradient_cosine_assim_relation": result.scalars["gradient_cosine_assim_relation"],
                        }
                    )
                state.global_step += 1
                completed_global = state.global_step
                invocation_steps += 1
                completed_epoch = (site_step + 1) % steps_per_epoch == 0 or site_step + 1 == total_steps
                if completed_epoch and hasattr(method, "on_epoch_end"):
                    if calibration_batcher is None:
                        raise RuntimeError("method requested an epoch-end calibrator without a labeled-only batcher")
                    try:
                        method.on_epoch_end(  # type: ignore[attr-defined]
                            epoch=state.epoch,
                            calibration_batcher=calibration_batcher,
                            device=self.device,
                        )
                    except BaseException as exc:
                        trainer.failure_bundle(
                            run_dir,
                            error=exc,
                            labeled_batch=labeled_batch,
                            unlabeled_batch=unlabeled_batch,
                            state=state,
                            result=result,
                        )
                        self._write_train_log(run_dir / "train_log.csv", train_rows)
                        raise
                if checkpoint_interval and state.global_step % checkpoint_interval == 0:
                    self._checkpoint(
                        run_dir / "checkpoint_last.pt",
                        method=method,
                        trainer=trainer,
                        site_id=site_id,
                        site_index=site_index,
                        epoch=state.epoch,
                        completed_site_steps=site_step + 1,
                        completed_global_steps=state.global_step,
                    )
                    if bool(self.config["training"].get("preserve_interval_checkpoints", False)):
                        history = run_dir / f"checkpoint_step_{state.global_step:06d}.pt"
                        if history.exists():
                            raise FileExistsError(history)
                        os.link(run_dir / "checkpoint_last.pt", history)
                    self._write_train_log(run_dir / "train_log.csv", train_rows)
                    if pilot_trajectory_rows:
                        write_csv(run_dir / "pilot_trajectory.csv", pilot_trajectory_rows)
                if maximum is not None and invocation_steps >= int(maximum):
                    self._checkpoint(
                        run_dir / "checkpoint_last.pt",
                        method=method,
                        trainer=trainer,
                        site_id=site_id,
                        site_index=site_index,
                        epoch=state.epoch,
                        completed_site_steps=site_step + 1,
                        completed_global_steps=state.global_step,
                    )
                    self._write_train_log(run_dir / "train_log.csv", train_rows)
                    interrupted = True
                    break
            if interrupted:
                break
            if hasattr(method, "estimate_fisher"):
                fisher_summary = method.estimate_fisher(labeled_batcher, device=self.device)  # type: ignore[attr-defined]
                write_json(run_dir / f"fisher_summary_{site_index}_{site_id}.json", fisher_summary)
            site_summary = method.end_site(site_id)
            if hasattr(method, "write_site_artifacts"):
                method.write_site_artifacts(run_dir=run_dir, site_id=site_id, site_index=site_index)  # type: ignore[attr-defined]
            final_checkpoint = run_dir / f"checkpoint_final_site{site_index}_{site_id}.pt"
            self._checkpoint(
                final_checkpoint,
                method=method,
                trainer=trainer,
                site_id=site_id,
                site_index=site_index,
                epoch=max(0, (total_steps - 1) // max(1, steps_per_epoch)),
                completed_site_steps=total_steps,
                completed_global_steps=completed_global,
            )
            if self.config["method"].get("protocol_id") in {"lcrseg_v0_3", "lcrseg_v0_4a", "srgas_v0_1", "srgas_v0_1a", "srgas_v0_2"}:
                protocol_checkpoint = run_dir / f"checkpoint_site_{site_index}_{site_id}.pt"
                if protocol_checkpoint.exists():
                    raise FileExistsError(protocol_checkpoint)
                os.link(final_checkpoint, protocol_checkpoint)
            # The final stage also receives the canonical name mandated by the
            # result-directory contract.
            if stage_position == len(stages) - 1:
                canonical = run_dir / "checkpoint_final.pt"
                if canonical.exists():
                    raise FileExistsError(canonical)
                os.link(final_checkpoint, canonical)
            previous_final = final_checkpoint
            write_json(run_dir / f"site_summary_{site_index}_{site_id}.json", site_summary)
            evaluation = evaluate_sites(
                method.model,
                data_root=self.data_root,
                seed=self.seed,
                dataset=self.dataset,
                sites=self.evaluation_sites,
                num_classes=int(self.spec["classes"]),
                role=str(self.config["data"].get("evaluation_role", "test")),
                device=self.device,
                batch_size=int(self.config["training"].get("evaluation_batch_size", 4)),
            )
            for row in evaluation.per_case:
                per_case_rows.append({"trained_site": site_id, "trained_site_index": site_index, **row})
            for row in evaluation.per_site:
                matrix_rows.append({"trained_site": site_id, "trained_site_index": site_index, "evaluation_site": row["site"], **row})
            self._write_train_log(run_dir / "train_log.csv", train_rows)
            write_csv(run_dir / "per_case_metrics.csv", per_case_rows)
            write_csv(run_dir / "site_matrix_long.csv", matrix_rows)
            self._write_matrix(run_dir, matrix_rows, metric="mean_foreground_dice", filename="site_matrix_dice.csv")
            self._write_matrix(run_dir, matrix_rows, metric="mean_foreground_asd", filename="site_matrix_asd.csv")
            self._write_matrix(run_dir, matrix_rows, metric="mean_foreground_hd95", filename="site_matrix_hd95.csv")
            self._write_per_class_metrics(run_dir, matrix_rows)
            resume_checkpoint = None
        analysis.write(run_dir)
        # Required analysis placeholders, to be filled by the separate hidden
        # GT analysis CLI rather than the training process.
        analysis_dir = run_dir / "analysis"
        for filename, fields in {
            "learnability_bins.csv": ["bin", "count", "pseudo_accuracy"],
            "compatibility_bins.csv": ["bin", "count", "old_accuracy"],
            "gradient_cosine.csv": ["site_id", "site_step", "global_step", "gradient_cosine_assim_relation"],
        }.items():
            path = analysis_dir / filename
            if not path.exists():
                write_csv(path, [], fieldnames=fields)
        if gradient_rows:
            write_csv(
                analysis_dir / "gradient_cosine.csv",
                gradient_rows,
                fieldnames=["site_id", "site_step", "global_step", "gradient_cosine_assim_relation"],
            )
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "interrupted" if interrupted else "complete",
            "method": self.method_name,
            "dataset": self.dataset,
            "seed": self.seed,
            "optimization_seed": self.optimization_seed,
            "run_dir": str(run_dir),
            "summary": self._summary(matrix_rows),
            "completed_global_steps": completed_global,
            "manifest_hash": sha256_path(self.manifest_path),
            "split_hash": sha256_path(self.split_path),
            "protocol_id": self.config["method"].get("protocol_id"),
            "variant_id": self.config["method"].get("variant_id"),
            "completed_parent_steps": int(self.config["experiment"].get("initial_global_step", 0)),
            "new_optimizer_steps": completed_global - int(self.config["experiment"].get("initial_global_step", 0)),
            "equivalent_full_run_steps": completed_global,
        }
        write_json(run_dir / "run_summary.json", summary)
        return summary
