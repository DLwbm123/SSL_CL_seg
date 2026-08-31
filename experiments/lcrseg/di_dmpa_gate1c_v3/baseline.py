"""Record v3 evidence around the unchanged repaired Gate0 training engine."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from di_dmpa_jascl import runner as legacy
from di_dmpa_jascl.checkpoint import capture_rng_state, save_checkpoint
from di_dmpa_jascl.config import sha256_file
from . import PROTOCOL
from .durable import write_new, now


def state_digest(value):
    """Typed value hash for nested model/optimizer/RNG state, independent of pickle."""
    digest = hashlib.sha256()

    def visit(item):
        digest.update(type(item).__name__.encode() + b"\0")
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(item.dtype).encode() + repr(tuple(item.shape)).encode() + array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(item.dtype.str.encode() + repr(item.shape).encode() + item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\xff")

    visit(value)
    return digest.hexdigest()


HASHED_FIELDS = ("student", "ema_teacher", "optimizer", "scheduler", "gas_state", "rng_state",
                 "stage_state", "sampler_state", "best_metric", "prototypes", "prototype_counts", "prototype_valid_mask")


def bind_payload(payload, binding, capture, *, stage_best=False):
    """An early best receives the actual live stage bank when the stage is sealed."""
    payload["prototype_counts"] = None if capture is None else capture["counts"]
    payload["prototype_valid_mask"] = None if capture is None else capture["valid_mask"]
    if capture is not None:
        payload["prototypes"] = capture["bank"]
    payload["v3"] = dict(protocol=PROTOCOL, binding=binding, stage_best_sealed=stage_best,
                         pas_state="NOT_YET_CREATED" if capture is None else "HISTORICALLY_CAPTURED",
                         legacy_pas_capture=None if capture is None else capture["metadata"],
                         legacy_pas_reconstructed=False)
    payload["v3"]["field_sha256"] = {key: state_digest(payload[key]) for key in HASHED_FIELDS}
    payload["v3"]["classifier_sha256"] = {
        source: state_digest({key: value for key, value in payload[source].items() if key.startswith("decoder.conv_logit.")})
        for source in ("student", "ema_teacher")}
    return payload


def verify_payload(payload, *, require_stage_best=True):
    record = payload["v3"]
    if record["protocol"] != PROTOCOL or record["legacy_pas_reconstructed"]:
        raise RuntimeError("wrong or reconstructed v3 checkpoint")
    for key in HASHED_FIELDS:
        if state_digest(payload[key]) != record["field_sha256"][key]:
            raise RuntimeError(f"checkpoint internal hash mismatch: {key}")
    if require_stage_best:
        if not record["stage_best_sealed"] or record["pas_state"] != "HISTORICALLY_CAPTURED":
            raise RuntimeError("not a sealed v3 stage-best checkpoint")
        if not math.isfinite(float(payload["best_metric"])):
            raise RuntimeError("stage best metric is not finite")
        bank, counts, mask = (payload[k] for k in ("prototypes", "prototype_counts", "prototype_valid_mask"))
        if bank.ndim != 2 or bank.shape[0] != 3 or counts.shape != (3,) or mask.shape != (3,):
            raise RuntimeError("incomplete direct PAS bank/count/validity fields")
        if not torch.isfinite(bank).all() or not (counts > 0).all() or not mask.all():
            raise RuntimeError("invalid stage PAS bank")
        for item, name in ((bank, "bank"), (counts, "counts"), (mask, "valid_mask")):
            if state_digest(item) != record["legacy_pas_capture"][name + "_sha256"]:
                raise RuntimeError("historical PAS capture does not match checkpoint")
        if record["legacy_pas_capture"]["binding"] != record["binding"]:
            raise RuntimeError("historical PAS capture provenance mismatch")
    for source in ("student", "ema_teacher"):
        if not all(torch.isfinite(value).all() for value in payload[source].values()):
            raise RuntimeError("nonfinite model state")
        classifier = {key: value for key, value in payload[source].items() if key.startswith("decoder.conv_logit.")}
        if not classifier or state_digest(classifier) != record["classifier_sha256"][source]:
            raise RuntimeError("classifier checkpoint hash mismatch")
    return record


class RegeneratedB0Runner(legacy.Gate0RepairedRunner):
    """Only evidence capture and final checkpoint sealing; no training-loop copy."""

    def __init__(self, *, provenance, **kwargs):
        if float(kwargs["config"]["training"]["lambda_u"]) != 0.5:
            raise RuntimeError("C0 is not authorized in this v3 phase")
        self.pas_capture = None
        self.provenance = dict(provenance)
        super().__init__(**kwargs)
        self.provenance.update(code_commit=self.git_commit, config_sha256=self.config_hash,
                               training_manifest_sha256=sha256_file(self.adapter.manifest_path),
                               fundus_split_sha256=sha256_file(self.adapter.split_path),
                               official_jascl_commit=self.config["upstream_commit"], seed=self.seed)

    def _capture_prototypes(self, student, batches, *, num_classes, device, ignore_label):
        """Observe real prototype forwards; do not refit or add stochastic draws."""
        counts = torch.zeros(num_classes, dtype=torch.int64)
        seen = 0
        stage = int(self.stage_state["stage_index"])
        rng_before = state_digest(capture_rng_state())
        student_before = state_digest(student.state_dict())

        def count_labels(module, args, result):
            nonlocal seen
            if seen >= len(batches):
                raise RuntimeError("unexpected extra prototype forward")
            labels = batches[seen]["label"].detach().cpu()
            resized = F.interpolate(labels[:, None].float(), result[1].shape[-2:], mode="nearest").squeeze(1).long()
            for class_id in range(num_classes):
                counts[class_id] += int(((resized == class_id) & (resized != ignore_label)).sum())
            seen += 1

        hook = student.register_forward_hook(count_labels)
        try:
            bank = self._original_prototype_function(student, batches, num_classes=num_classes,
                                                     device=device, ignore_label=ignore_label)
        finally:
            hook.remove()
        if seen != len(batches):
            raise RuntimeError("missing prototype forward observations")
        bank_cpu = bank.detach().cpu().clone()
        valid = (counts > 0) & torch.isfinite(bank_cpu).all(1) & (torch.linalg.vector_norm(bank_cpu.float(), dim=1) > 0)
        metadata = dict(captured_at=now(), seed=self.seed, stage_index=stage, domain=self.domain_order[stage],
                        epoch=int(self.stage_state["epoch"]), global_step=int(self.stage_state["global_step"]),
                        source="unchanged_compute_single_prototypes_live_return", source_role="train_labeled",
                        prototype_forward_batches=seen, additional_model_forwards=0,
                        student_state_before_sha256=student_before, rng_before_sha256=rng_before,
                        rng_after_sha256=state_digest(capture_rng_state()), bank_sha256=state_digest(bank_cpu),
                        counts_sha256=state_digest(counts), valid_mask_sha256=state_digest(valid),
                        binding=dict(self.provenance), reconstruction=False)
        self.pas_capture = dict(bank=bank_cpu, counts=counts, valid_mask=valid, metadata=metadata)
        directory = self.output_dir / "legacy_pas_captures"
        directory.mkdir(exist_ok=True)
        path = directory / f"stage{stage}.pt"
        with path.open("xb") as handle:
            torch.save(self.pas_capture, handle)
            handle.flush()
            os.fsync(handle.fileno())
        write_new(directory / f"stage{stage}.json", dict(metadata, capture_file_sha256=sha256_file(path)))
        return bank

    def _active_capture(self):
        capture = self.pas_capture
        return capture if capture and capture["metadata"]["stage_index"] == int(self.stage_state["stage_index"]) else None

    def _checkpoint_payload(self):
        return bind_payload(super()._checkpoint_payload(), self.provenance, self._active_capture())

    def resume(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        record = verify_payload(payload, require_stage_best=False)
        if record["stage_best_sealed"]:
            raise RuntimeError("sealed stage-best is a diagnostic artifact; resume requires last.pt")
        if record["binding"] != self.provenance:
            raise RuntimeError("v3 resume provenance mismatch")
        if record["pas_state"] == "HISTORICALLY_CAPTURED":
            self.pas_capture = dict(bank=payload["prototypes"], counts=payload["prototype_counts"],
                                    valid_mask=payload["prototype_valid_mask"], metadata=record["legacy_pas_capture"])
        super().resume(checkpoint_path)

    def _stage_end(self, stage_index, domain):
        capture = self._active_capture()
        if capture is None:
            raise RuntimeError("stage completed without a directly captured historical PAS bank")
        super()._stage_end(stage_index, domain)
        path = self.output_dir / f"stage_{stage_index}_{domain}" / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selected_epoch = int(payload["stage_state"]["epoch"])
        selected_state_hash = state_digest({k: payload[k] for k in HASHED_FIELDS if k not in
                                           ("prototypes", "prototype_counts", "prototype_valid_mask")})
        bind_payload(payload, self.provenance, capture, stage_best=True)
        payload["v3"].update(selected_checkpoint_epoch=selected_epoch,
                             bank_captured_after_selected_epoch=selected_epoch <= capture["metadata"]["epoch"],
                             selected_state_sha256=selected_state_hash)
        verify_payload(payload)
        # The mutable training best becomes its sealed v3 stage checkpoint once,
        # with no change to selection, model/optimizer/RNG state or PAS values.
        save_checkpoint(path, payload)
        write_new(path.parent / "V3_STAGE_CHECKPOINT_SEAL.json", dict(
            sealed_at=now(), stage_index=stage_index, domain=domain, checkpoint_sha256=sha256_file(path),
            selected_epoch=selected_epoch, selected_state_sha256=selected_state_hash,
            legacy_pas_capture=capture["metadata"], field_sha256=payload["v3"]["field_sha256"],
            checkpoint_selection_changed=False, additional_model_forwards=0, reconstructed=False))

    def run(self, **kwargs):
        self._original_prototype_function = legacy.compute_single_prototypes
        # This process runs one training engine. The scoped adapter records its
        # genuine return; no old file, equation, draw, or caller is changed.
        with patch.object(legacy, "compute_single_prototypes", self._capture_prototypes):
            return super().run(**kwargs)
