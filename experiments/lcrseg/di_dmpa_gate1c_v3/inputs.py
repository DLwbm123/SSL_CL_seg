"""Read-only loading of directly sealed v3 inputs; old private inputs are ineligible."""
from pathlib import Path
import subprocess

import torch

from di_dmpa_gate1.feature_extraction import state_hash
from di_dmpa_jascl.modeling import build_lcrseg_unet_jascl_model, assert_complete_classifier_load
from di_dmpa_jascl.provenance import assert_upstream_unchanged
from .baseline import verify_payload
from .durable import sha256


def load_models(root, checkpoint, *, device, sources=("student", "ema_teacher")):
    if checkpoint["baseline"] != "B0" or sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise RuntimeError("unbound or changed regenerated B0 input")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    record = verify_payload(payload)
    if payload["schema_version"] != 2 or payload["git_commit"] != record["binding"]["code_commit"] or payload["config_hash"] != record["binding"]["config_sha256"]:
        raise RuntimeError("checkpoint code/config provenance mismatch")
    if record["binding"] != checkpoint["legacy_pas_capture"]["binding"]:
        raise RuntimeError("checkpoint differs from the registered v3 baseline manifest")
    capture = record["legacy_pas_capture"]
    if (payload["stage_state"]["stage_index"] != checkpoint["stage_index"] or record["binding"]["seed"] != checkpoint["seed"]
            or capture["stage_index"] != checkpoint["stage_index"] or capture["domain"] != checkpoint["domain"]):
        raise RuntimeError("checkpoint seed/stage mismatch")
    if payload["prototypes"].shape != (3, 16) or payload["prototypes"].dtype != torch.float32:
        raise RuntimeError("legacy PAS is not the direct FP32 3x16 bank")
    if not torch.equal(payload["student"]["decoder.conv_logit.grad_update"], payload["gas_state"]["grad_update"]):
        raise RuntimeError("GAS/classifier mismatch")
    reference = Path(root) / "third_party/JASCL_REFERENCE"
    upstream = "Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT"
    assert_upstream_unchanged(reference, upstream)
    if subprocess.check_output(["git", "-C", str(reference), "diff", "--name-only", "HEAD"], text=True).strip():
        raise RuntimeError("official tracked reference changed")
    if not sources or not set(sources).issubset({"student", "ema_teacher"}):
        raise RuntimeError("unregistered probability/feature source")
    models = {}
    for source in sources:
        model = build_lcrseg_unet_jascl_model(reference, upstream_path=upstream, input_channels=3, num_classes=3)
        assert_complete_classifier_load(payload[source], model)
        model.load_state_dict(payload[source], strict=True)
        if state_hash(model.state_dict()) != state_hash(payload[source]):
            raise RuntimeError("strict loaded model value mismatch")
        models[source] = model.to(device).eval().requires_grad_(False)
    return models, payload
