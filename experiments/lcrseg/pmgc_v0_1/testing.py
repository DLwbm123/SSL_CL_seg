"""Unchanged compatible regressions plus synthetic execution of the real kernels."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b, execution as e
from di_dmpa_gate1c_v2.precision import attach_gradient_student
from di_dmpa_gate1c_v3 import durable as d
from tests.di_dmpa_gate1c_v2.test_core import Tiny
from . import run, evaluator as ev
from .core import require, COUNT_KEYS
from .protocol import authority, ROOT, REPO


def synthetic_models():
    torch.manual_seed(71)
    model = Tiny().eval()
    models = dict(student=model, ema_teacher=copy.deepcopy(model).requires_grad_(False), old=copy.deepcopy(model).requires_grad_(False))
    attach_gradient_student(models, dict(diagnostic_precision="float64_shadow", _precision_contract_verified=True))
    return models, torch.ones(3, 16)


def synthetic_images(rows, data_root):
    return torch.stack([torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(b.S(["synthetic", r["case_id"]]))) for r in rows])


def synthetic_labels(rows, data_root, *, role):
    require(role in ("train_labeled", "val"), "synthetic hidden/test role")
    return [np.tile(np.arange(8) % 3, (8, 1)).astype(np.int64) for _ in rows]


def synthetic_records(data_root, p, seed, stage, role):
    require(role in ("train_labeled", "train_unlabeled", "val"), "forbidden synthetic role")
    unit = next(x for x in p["benchmark"]["case_plans"] if (x["seed"], x["stage_index"]) == (seed, stage))
    return [dict(case_id=case, image_h5_relpath=case, image_sha256="synthetic_only", label_h5_relpath="" if role == "train_unlabeled" else case,
                 label_sha256="" if role == "train_unlabeled" else "synthetic_only", primary_20pct_split=role, dataset="fundus") for case in unit["roles"][role]]


def synthetic_geometry(unit):
    labels = synthetic_labels([{}], None, role="train_labeled")[0]
    cases = []
    for case in unit["guard_case_ids"]:
        classes = []
        for c in range(3):
            coords = np.argwhere(labels == c).tolist()
            classes.append(dict(class_id=c, sampled_pixels=len(coords), coordinates=coords, boundary=[False]*len(coords)))
        cases.append(dict(case_id=case, classes=classes))
    return dict(role="train_labeled", seed=unit["seed"], stage_index=unit["stage_index"], cases=cases)


def compile_graph(code_commit, output):
    reg, p = authority()
    observed = {}
    with contextlib.ExitStack() as stack:
        for module in (run, ev): stack.enter_context(patch.object(module, "_images", synthetic_images))
        stack.enter_context(patch.object(b, "records", synthetic_records))
        stack.enter_context(patch.object(e, "visible_labels", synthetic_labels))
        for stage in (1, 2):
            unit = next(u for u in reg["fixed_units"] if (u["seed"], u["stage_index"]) == (0, stage))
            folder = output/f"synthetic_stage{stage}"; folder.mkdir()
            models, bank = synthetic_models()
            prepared = run.prepare(models, bank, unit, synthetic_geometry(unit), p, "synthetic", folder, "cpu")
            prepared["checkpoint_hashes_unchanged"] = True
            d.write_new(folder/"PREPARATION_UNIT.json", prepared)
            panels = {side: ev.load_panel(prepared["validation"][side], "synthetic", "cpu", role="val") for side in ("previous", "current")}
            panels["train_labeled"] = ev.load_panel(prepared["train_labeled"], "synthetic", "cpu", role="train_labeled")
            pair = unit["formal_pairs"][0]
            images_u = synthetic_images([dict(case_id=c) for c in pair["unlabeled_case_ids"]], None)
            images_l = synthetic_images([dict(case_id=c) for c in pair["labeled_case_ids"]], None)
            labels = torch.from_numpy(np.stack(synthetic_labels([{}, {}], None, role="train_labeled")))
            row, arrays = run.pair_kernel(models, bank, pair, (images_u, images_l, labels), unit, prepared, panels, reg["call_graph"]["per_pair_by_stage"][str(stage)])
            row["before"] = {side: prepared["validation"][side]["before"] for side in ("previous", "current")}
            row["before"]["train_labeled"] = prepared["train_labeled"]["before"]
            row["before_batches"] = {side: prepared["validation"][side]["before_batches"] for side in ("previous", "current")}
            row["before_batches"]["train_labeled"] = prepared["train_labeled"]["before_batches"]
            row["arrays"] = b.save_arrays(folder/"pair_vectors.npz", arrays)
            d.write_new(folder/"PAIR_RESULT.json", row)
            from .report import audit_pair
            audit = audit_pair(row, prepared, reg["call_graph"]["per_pair_by_stage"][str(stage)], require_full_inventory=False)
            d.write_new(folder/"SYNTHETIC_ARTIFACT_AUDIT.json", audit)
            observed[str(stage)] = dict(preparation=prepared["counts"], pair=row["counts"],
                                        preparation_trace=prepared["call_trace"], pair_trace=row["call_trace"])
    prep = {k: 3*sum(observed[str(s)]["preparation"][k] for s in (1, 2)) for k in COUNT_KEYS}
    integration = {k: 3*sum(observed[str(s)]["pair"][k] for s in (1, 2)) for k in COUNT_KEYS}
    formal = {k: 8*integration[k] for k in COUNT_KEYS}
    total = {k: prep[k]+integration[k]+formal[k] for k in COUNT_KEYS}
    total["total_forwards"] = total["native_forwards"]+total["fp64_forwards"]
    require(total == reg["call_graph"]["total"], "compiled graph differs from preregistration", "BLOCKED_CALL_GRAPH_MISMATCH")
    sources = {str(path.relative_to(REPO)): d.sha256(path) for folder in (ROOT/"pmgc_v0_1", ROOT/"tests/pmgc_v0_1") for path in sorted(folder.glob("*.py"))}
    return dict(status="PASS", code_commit=code_commit, data="synthetic_only", real_model_forwards=0,
                representative_units="seed0_stage1 and seed0_stage2, identical real prepare/pair/evaluator kernels, real batch counts and shapes scaled to synthetic8x8",
                model="unchanged Gate1C Tiny synthetic model; real 51/484016 inventory guarded separately in tests and runtime", observed=observed,
                preparation=prep, integration=integration, formal=formal, total=total, exact_source_sha256=sources,
                total_real_forward_budget=6654, total_real_autograd_budget=684, model_optimizer_steps=0, backward_called=False)


def arguments(output):
    reg, _ = authority()
    command = ["-q", "-p", "no:cacheprovider", "tests/di_dmpa_gate1c_v2", "tests/di_dmpa_gate1c_v3", "tests/mmpr_gs_v0_1", "tests/pmgc_v0_1",
               "tests/gate0/test_config_protocol.py", "tests/gate0/test_report_compiler.py", "tests/gate0/test_official_model_contract.py",
               "tests/gate0/test_classifier_stochasticity.py", "tests/gate0/test_pas_probability.py"]
    for name in reg["tests"]["excluded_before_collection"]:
        command.append(("--deselect=" if "::" in name else "--ignore=")+name)
    return command+["--junitxml", str(output/"pytest.xml"), "--basetemp", str(output/"scratch")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    code = b.git(REPO, "rev-parse", "HEAD")
    require(code == args.code_commit, "test/compile checkout mismatch")
    if not args.development:
        require(not b.git(REPO, "status", "--porcelain"), "exact source compiler requires clean checkout")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    command = arguments(args.output)
    with (args.output/"pytest_output.txt").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), b.no_updates():
        result = pytest.main(command)
    totals = {k: sum(int(s.get(k, 0)) for s in ET.parse(args.output/"pytest.xml").iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    report = dict(status="PASS" if result == 0 and totals["failures"] == totals["errors"] == totals["skipped"] == 0 else "FAIL",
                  code_commit=code, tests=totals["tests"], failures=totals["failures"], errors=totals["errors"], skips=totals["skipped"],
                  junit_path=str(args.output/"pytest.xml"), junit_sha256=d.sha256(args.output/"pytest.xml"), exit_code=int(result),
                  pytest_arguments=command, data="synthetic_only", real_model_forwards=0, development=args.development,
                  historical_test_files_modified=False, optimizer_constructed=False, backward_called=False)
    d.write_new(args.output/"PMGC_TEST_REPORT.json", report)
    aliases = []
    scratch = args.output/"scratch"
    for path in scratch.rglob("*"):
        if path.is_symlink():
            target = path.resolve()
            require(path.name.endswith("current") and target.is_relative_to(scratch.resolve()), "unexpected test symlink")
            aliases.append(dict(path=str(path.relative_to(scratch)), target=str(target.relative_to(scratch))))
            path.unlink()
    d.write_new(args.output/"PYTEST_TEMPORARY_ALIASES.json", dict(aliases=aliases, payloads_removed=0))
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS": raise SystemExit(1)
    with b.no_updates():
        compiler = compile_graph(code, args.output)
    d.write_new(args.output/"PMGC_CALL_GRAPH.json", compiler)
    print(json.dumps(dict(status=compiler["status"], total=compiler["total"])), flush=True)


if __name__ == "__main__":
    main()
