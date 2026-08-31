"""Observe fresh native forwards without changing the frozen score builder."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from di_dmpa_jascl.modeling import LCRSegUNet2DJASCL
from di_dmpa_gate1c_v2 import binding as b, execution as e, reliability as r
from di_dmpa_gate1c_v2.cache_reuse import validate_unit, validate_case
from .durable import write_new, read, sha256

PAS_FIELDS = ("valid_mask", "predicted_class", "confidence", "similarity")
EXTRA_SCORES = ("current_margin", "history_margin", "history_similarity")


@contextmanager
def capture(output, maximum_forwards):
    """Persist already-computed tensors and PAS intermediates; add no forwards."""
    output = Path(output)
    original_forward, original_pas = LCRSegUNet2DJASCL.forward, r.compute_pas_validity
    original_build, original_save = e.build, e.save_arrays
    pending, pas = {}, []
    counts = dict(native_forwards=0, cases=0, original_PAS_calls=0)

    def forward(model, *args, **kwargs):
        b.require(counts["native_forwards"] < maximum_forwards and next(model.parameters()).dtype == torch.float32,
                  "fresh validation forward budget/dtype exceeded")
        counts["native_forwards"] += 1
        return original_forward(model, *args, **kwargs)

    def observe_pas(*args, **kwargs):
        result = original_pas(*args, **kwargs)
        pas.append({k: getattr(result, k).detach().cpu().numpy().copy() for k in PAS_FIELDS})
        counts["original_PAS_calls"] += 1
        return result

    def build(sl, sf, tl, tf, legacy, current, history):
        b.require(not pending and not pas, "unsaved previous validation case")
        result = original_build(sl, sf, tl, tf, legacy, current, history)
        b.require(len(pas) == 2, "student/teacher PAS observations incomplete")
        for name, value in (("student_logits", sl), ("student_features", sf), ("teacher_logits", tl), ("teacher_features", tf),
                            ("student_probability", sl.float().softmax(1)), ("teacher_probability", tl.float().softmax(1))):
            pending[name] = value.detach().cpu().numpy().copy()
        for source, values in zip(("student", "teacher"), pas):
            pending.update({source + "_pas_" + k: v for k, v in values.items()})
        pending.update({k: result[k] for k in EXTRA_SCORES})
        pending["null_mask"] = ~result["active_mask"]
        height, width = sl.shape[-2:]
        pending["pixel_yx"] = np.indices((height, width), dtype=np.uint16).reshape(2, -1).T.copy()
        b.require(np.array_equal(result["R1"], (pas[0]["valid_mask"] & pas[1]["valid_mask"]).reshape(-1)), "direct PAS parity failed")
        pas.clear()
        return result

    def save(path, arrays):
        b.require(bool(pending), "raw validation values missing")
        path = Path(path)
        relative = path.relative_to(output / "validation_cache")
        desc = original_save(path, arrays)
        desc["raw_values"] = original_save(output / "validation_raw" / relative, pending)
        desc["uid_order"] = "one case; all pixels in row-major [y,x] order; seed/stage/case identity in the containing unit"
        pending.clear()
        counts["cases"] += 1
        return desc

    with patch.object(LCRSegUNet2DJASCL, "forward", forward), patch.object(r, "compute_pas_validity", observe_pas), \
            patch.object(e, "build", build), patch.object(e, "save_arrays", save):
        yield counts
    b.require(not pending and not pas, "unpublished final validation case")


def validate_raw(case, stage, *, height=384, width=384):
    raw = b.read_arrays(case["arrays"]["raw_values"])
    scores = b.read_arrays(case["arrays"])
    expected = {"student_logits", "student_features", "teacher_logits", "teacher_features", "student_probability", "teacher_probability",
                "null_mask", "pixel_yx", *EXTRA_SCORES, *(s + "_pas_" + k for s in ("student", "teacher") for k in PAS_FIELDS)}
    b.require(set(raw) == expected, "incomplete raw validation cache")
    for source in ("student", "teacher"):
        for kind, channels in (("logits", 3), ("features", 16), ("probability", 3)):
            a = raw[source + "_" + kind]
            b.require(a.shape == (1, channels, height, width) and a.dtype == np.float32, "native raw tensor shape/dtype changed")
            b.finite(a)
            if kind != "probability":
                b.require(b.tensor_hash(torch.from_numpy(a)) == case[source + "_" + kind + "_sha256"], "raw/native tensor identity changed")
        for kind, dtype in zip(PAS_FIELDS, (np.bool_, np.int64, np.float32, np.float32)):
            a = raw[source + "_pas_" + kind]
            b.require(a.shape == (1, height, width) and a.dtype == dtype, "PAS intermediate shape/dtype changed")
            b.finite(a)
    b.require(np.array_equal(raw["pixel_yx"], np.indices((height, width), dtype=np.uint16).reshape(2, -1).T), "pixel UID order changed")
    b.require(np.array_equal(raw["null_mask"], ~scores["active_mask"]), "raw null mask changed")
    b.require(np.array_equal(scores["R1"], (raw["student_pas_valid_mask"] & raw["teacher_pas_valid_mask"]).reshape(-1)), "raw PAS parity changed")
    probability = raw["teacher_probability"].transpose(0, 2, 3, 1).reshape(-1, 3)
    b.require(np.array_equal(probability, scores["teacher_probability"]), "raw/native probability changed")
    for key in EXTRA_SCORES:
        a = raw[key]
        b.require(a.shape == (height * width,) and a.dtype == np.float64 and np.isnan(a[~scores["active_mask"]]).all(), "raw score/null contract changed")
        if key == "current_margin" or stage:
            b.finite(a[scores["active_mask"]])
        else:
            b.require(np.isnan(a).all(), "stage0 history must be unavailable")
    return dict(raw_fields=len(raw), native_values_verified=True, direct_PAS_parity=True, full_UID_order_verified=True)


def audit(output, data_root, p, freeze, meta):
    output = Path(output)
    units = []
    for seed in range(3):
        for stage in range(3):
            name = e.unit_name(seed, stage)
            path = output / "validation_units" / (name + ".json")
            unit = read(path)
            cp = b.checkpoint(p, seed, stage)
            guard_path = output / "validation_models" / name / "immutability" / f"B0_seed{seed}_stage{stage}.json"
            guard = read(guard_path)
            validate_unit(unit, guard, meta, seed, stage, cp, cp["legacy_pas_tensor_sha256"], r.bank_identity(freeze, seed, stage))
            rows = b.records(data_root, p, seed, stage, "val")
            plan = next(u for u in p["validation"]["plans"] if (u["seed"], u["stage_index"]) == (seed, stage))
            b.require([c["case_id"] for c in unit["cases"]] == [c["case_id"] for c in rows], "validation coverage/order mismatch")
            for case, row, registered in zip(unit["cases"], rows, plan["cases"]):
                desc = case["arrays"]
                manifest_row = dict(path=str(Path(desc["path"]).relative_to(output)), sha256=desc["sha256"], bytes=desc["bytes"])
                validate_case(case, row, registered, manifest_row, output, stage)
                raw_path = Path(desc["raw_values"]["path"])
                b.require(raw_path.resolve().is_relative_to(output.resolve()) and raw_path.stat().st_size == desc["raw_values"]["bytes"], "raw cache escaped/changed size")
                validate_raw(case, stage)
            units.append(dict(seed=seed, stage_index=stage, path=str(path), sha256=sha256(path), cases=unit["cases"],
                              checkpoint_sha256=cp["sha256"], model_guard=dict(path=str(guard_path), sha256=sha256(guard_path))))
    b.require(sum(len(u["cases"]) for u in units) == 495 and len(list((output / "validation_models").rglob("immutability/*.json"))) == 9,
              "495 fresh cases and nine complete model guards required")
    b.require(len(list((output / "validation_cache").rglob("*.npz"))) == len(list((output / "validation_raw").rglob("*.npz"))) == 495,
              "missing/extra validation files")
    manifest = dict(metadata=meta, units=units, case_count=495, model_guards=9, fresh_generation=True,
                    old_private_cache_reads=0, UID_schema="[seed,stage,case_id,y,x]; every pixel retained")
    write_new(output / "VALIDATION_CACHE_V3_MANIFEST.json", manifest)
    return dict(status="PASS_FRESH_VALIDATION_CACHE", metadata=meta, cases=495, model_guards=9,
                manifest_sha256=sha256(output / "VALIDATION_CACHE_V3_MANIFEST.json"), raw_native_values_and_all_intermediates_verified=True,
                hidden_gt_training_usage="none", test_gt_usage="none", old_raw_cache_reused=False)
