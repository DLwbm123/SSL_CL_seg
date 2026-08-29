"""Method factory for all baseline and LCR-Seg V0.1 variants."""
from __future__ import annotations

from typing import Any, Mapping

from ..models import UNet2D
from .base import merged_method_config
from .sequential_ssl import SequentialSSLMethod, UniformKDMethod
from .supervised import SupervisedMethod


def resolve_method_config(name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expand method defaults before provenance or checkpoint serialization.

    ``ContinualSegMethod`` also resolves these values defensively at runtime,
    but the runner must persist the identical mapping so a run can be replayed
    without relying on source-code defaults that may change later.
    """

    normalized = name.strip().lower().replace("-", "_")
    provided = dict(config or {})
    if normalized in {"lcrseg_v0_1", "lcrseg", "lcr"}:
        from .lcrseg_v0_1 import LCR_DEFAULTS

        lcr_config = dict(LCR_DEFAULTS)
        lcr_config.update(provided)
        return merged_method_config(lcr_config)
    if normalized in {"lcrseg_v0_2", "lcr_v0_2"}:
        from .lcrseg_v0_2 import LCR_V02_DEFAULTS

        lcr_config = dict(LCR_V02_DEFAULTS)
        lcr_config.update(provided)
        return merged_method_config(lcr_config)
    if normalized in {"lcrseg_v0_2a", "lcr_v0_2a"}:
        from .lcrseg_v0_2a import resolve_v02a_method_config

        return resolve_v02a_method_config(provided)
    if normalized in {"lcrseg_v0_3", "lcr_v0_3"}:
        from .lcrseg_v0_3 import resolve_v03_method_config

        return resolve_v03_method_config(provided)
    if normalized in {"lcrseg_v0_4a", "lcr_v0_4a"}:
        from .lcrseg_v0_4a import resolve_v04a_method_config

        return resolve_v04a_method_config(provided)
    if normalized in {"srgas_v0_1", "srgas", "sr_gas"}:
        from .srgas_v0_1 import resolve_srgas_method_config

        return resolve_srgas_method_config(provided)
    if normalized in {"srgas_v0_2", "lw_srgas"}:
        from .srgas_v0_2 import resolve_srgas_v02_method_config

        return resolve_srgas_v02_method_config(provided)
    if normalized in {"ss_ewc", "ewc"}:
        from .ewc import EWC_DEFAULTS

        ewc_config = dict(EWC_DEFAULTS)
        ewc_config.update(provided)
        return merged_method_config(ewc_config)
    return merged_method_config(provided)


def build_method(name: str, model: UNet2D, *, config: Mapping[str, Any] | None = None):
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"static_sup", "static_supervised"}:
        return SupervisedMethod(model, config=config, continual=False)
    if normalized in {"finetune_sup", "fine_tune_sup"}:
        return SupervisedMethod(model, config=config, continual=True)
    if normalized in {"joint_sup", "joint_supervised"}:
        method = SupervisedMethod(model, config=config, continual=False)
        method.method_name = "joint_sup"
        return method
    if normalized in {"static_ssl"}:
        return SequentialSSLMethod(model, config=config, static=True)
    if normalized in {"sequential_ssl", "seq_ssl"}:
        return SequentialSSLMethod(model, config=config, static=False)
    if normalized in {"joint_ssl"}:
        method = SequentialSSLMethod(model, config=config, static=False)
        method.method_name = "joint_ssl"
        return method
    if normalized in {"uniform_kd", "lwf", "uniform_kd_lwf"}:
        return UniformKDMethod(model, config=config, static=False)
    if normalized in {"ss_ewc", "ewc"}:
        from .ewc import EWCSegMethod

        return EWCSegMethod(model, config=config, static=False)
    if normalized in {"lcrseg_v0_1", "lcrseg", "lcr"}:
        from .lcrseg_v0_1 import LCRSegV01Method

        return LCRSegV01Method(model, config=config)
    if normalized in {"lcrseg_v0_2", "lcr_v0_2"}:
        from .lcrseg_v0_2 import LCRSegV02Method

        return LCRSegV02Method(model, config=config)
    if normalized in {"lcrseg_v0_2a", "lcr_v0_2a"}:
        from .lcrseg_v0_2a import LCRSegV02AMethod

        return LCRSegV02AMethod(model, config=config)
    if normalized in {"lcrseg_v0_3", "lcr_v0_3"}:
        from .lcrseg_v0_3 import LCRSegV03Method

        return LCRSegV03Method(model, config=config)
    if normalized in {"lcrseg_v0_4a", "lcr_v0_4a"}:
        from .lcrseg_v0_4a import LCRSegV04AMethod

        return LCRSegV04AMethod(model, config=config)
    if normalized in {"srgas_v0_1", "srgas", "sr_gas"}:
        from .srgas_v0_1 import SRGASV01Method

        return SRGASV01Method(model, config=config)
    if normalized in {"srgas_v0_2", "lw_srgas"}:
        from .srgas_v0_2 import SRGASV02Method

        return SRGASV02Method(model, config=config)
    raise ValueError(f"unknown method: {name}")


__all__ = ["SequentialSSLMethod", "SupervisedMethod", "UniformKDMethod", "build_method", "resolve_method_config"]
