"""LCR-Seg V0.3: frozen progressive-admission validation candidate.

V0.3 deliberately reuses the V0.2a R1 training path.  It adds no method
component; it only restricts the registered variants to legacy/uniform R0,
progressive/uniform R1, and progressive/no-relation P0.
"""
from __future__ import annotations

from typing import Any, Mapping

from .base import merged_method_config
from .lcrseg_v0_2a import LCRSegV02AMethod, resolve_v02a_method_config


FROZEN_FUNDUS_MANIFEST_HASHES = {
    0: "0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3",
    1: "d5d2913054bc96f13b2baec0f21109a7da92c1a2f5b07f0cde234b35bbfd92a9",
    2: "78379dc43035259f41b0f598e0bda25a31e68b15600bb611758ccc61cd2a0727",
}

FROZEN_FUNDUS_SPLIT_HASHES = {
    0: "f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88",
    1: "87affde62045894a8ce89701137f254ed56ba1f00951041bd2f6282cccbb5727",
    2: "af2f48281d8eb16d299871f12824a729d08cb3854b3753d69d42c0d842e34dd3",
}

FROZEN_V02A_R1_SITE0_SHA256 = "9bdadf34a5a32d936b14cfff3f4c9ffa2ee62c5f24142ca12b4a3b9815c46b32"

V03_REGISTERED_VARIANTS = {
    "R0": ("legacy_continuous_v01", "uniform_relation", 1.0),
    "R1": ("progressive_admission", "uniform_relation", 1.0),
    "P0": ("progressive_admission", "none", 0.0),
}

_FROZEN_PROGRESSIVE_SCHEDULE = {
    "start_fraction": 0.40,
    "end_fraction": 0.80,
    "schedule": "linear",
    "schedule_scope": "per_site",
    "classwise": True,
    "minimum_pixels_for_class_quantile": 32,
    "minimum_admitted_per_present_class": 1,
    "weight_after_admission": 1.0,
}


def validate_parent_checkpoint_lineage(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    expected_sha256: str = FROZEN_V02A_R1_SITE0_SHA256,
) -> dict[str, Any]:
    """Validate the only checkpoint authorized as a V0.3 P0 parent."""

    identity = (payload.get("site_id"), int(payload.get("site_index", -1)), int(payload.get("global_step", -1)))
    if identity != ("REFUGE", 0, 8000):
        raise ValueError("P0 parent must be the R1 REFUGE site-end checkpoint at global step 8000")
    method = dict(dict(payload.get("config_resolved") or {}).get("method") or {})
    semantics = (method.get("assimilation_mode"), method.get("consolidation_mode"))
    if semantics != ("progressive_admission", "uniform_relation"):
        raise ValueError("P0 parent does not have the frozen V0.2a R1 semantics")
    if checkpoint_sha256 != expected_sha256:
        raise ValueError("P0 parent checkpoint SHA-256 differs from the preregistration")
    return {
        "site_id": "REFUGE",
        "site_index": 0,
        "completed_parent_steps": 8000,
        "checkpoint_sha256": checkpoint_sha256,
    }


def resolve_v03_method_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve one of the three preregistered V0.3 variants."""

    provided = dict(config or {})
    if provided.get("protocol_id", "lcrseg_v0_3") != "lcrseg_v0_3":
        raise ValueError("V0.3 requires protocol_id=lcrseg_v0_3")
    variant = str(provided.get("variant_id", "R1")).upper()
    if variant not in V03_REGISTERED_VARIANTS:
        raise ValueError(f"unregistered V0.3 variant: {variant}")
    assimilation, consolidation, lambda_relation = V03_REGISTERED_VARIANTS[variant]
    actual = (
        provided.get("assimilation_mode", assimilation),
        provided.get("consolidation_mode", consolidation),
        float(provided.get("lambda_relation", lambda_relation)),
    )
    if actual != (assimilation, consolidation, lambda_relation):
        raise ValueError(
            f"{variant} differs from the frozen V0.3 semantics: "
            f"expected {(assimilation, consolidation, lambda_relation)}, got {actual}"
        )
    if any(
        bool(provided.get(key, False))
        for key in ("teacher_rejection", "compatibility_calibration", "compatibility_rejection", "multi_agent", "ric")
    ):
        raise ValueError("V0.3 forbids teacher rejection, multi-agent, and RIC")

    # Resolve through the frozen V0.2a R1/R0 path, then change only protocol
    # metadata and (for P0) disable the already-computed relation loss.
    mapped = dict(provided)
    mapped.update(
        {
            "protocol_id": "lcrseg_v0_2a",
            "variant_id": "R0" if variant == "R0" else "R1",
            "assimilation_mode": assimilation,
            "consolidation_mode": "uniform_relation",
            "lambda_relation": 1.0,
        }
    )
    base = resolve_v02a_method_config(mapped)
    base.update(
        {
            "protocol_id": "lcrseg_v0_3",
            "variant_id": variant,
            "assimilation_mode": assimilation,
            "consolidation_mode": consolidation,
            "lambda_relation": lambda_relation,
            "progressive_schedule": dict(_FROZEN_PROGRESSIVE_SCHEDULE),
            "teacher_rejection_enabled": False,
            "multi_agent": False,
            "ric": False,
            "v02a_r1_semantic_reference": (
                "/home/jiangsuiyang/SSL_CL/runs/"
                "fundus_seed0_lcrseg_v0_2a_r1_progressive_uniform_full200e"
            ),
        }
    )
    return merged_method_config(base)


def semantic_method_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return formula-bearing fields used by the V0.2a-R1 semantic diff."""

    excluded = {
        "name",
        "version",
        "protocol_id",
        "variant_id",
        "formal_r0_run",
        "auxiliary_u0_run",
        "u0_registered_as_formal_r0",
        "teacher_rejection_enabled",
        "multi_agent",
        "ric",
        "v02a_r1_semantic_reference",
    }
    return {key: value for key, value in dict(config).items() if key not in excluded}


class LCRSegV03Method(LCRSegV02AMethod):
    method_name = "lcrseg_v0_3"
    method_version = "0.3"

    def __init__(self, model, *, config: Mapping[str, Any] | None = None) -> None:
        resolved = resolve_v03_method_config(config)
        mapped = dict(resolved)
        mapped.update(
            {
                "protocol_id": "lcrseg_v0_2a",
                "variant_id": "R0" if resolved["variant_id"] == "R0" else "R1",
                "consolidation_mode": "uniform_relation",
                "lambda_relation": 1.0,
            }
        )
        super().__init__(model, config=mapped)
        self.config = resolved
        self.requires_labeled_calibration = False

    def protocol_semantics(self) -> dict[str, Any]:
        return {
            "protocol_id": self.config["protocol_id"],
            "variant_id": self.config["variant_id"],
            "assimilation_mode": self.config["assimilation_mode"],
            "consolidation_mode": self.config["consolidation_mode"],
            "lambda_relation": self.config["lambda_relation"],
            "learnability_formula_version": self.config["learnability_formula_version"],
            "progressive_schedule": dict(self.config["progressive_schedule"]),
            "teacher_rejection_enabled": False,
            "multi_agent": False,
            "ric": False,
            "v02a_r1_semantic_reference": self.config["v02a_r1_semantic_reference"],
        }
