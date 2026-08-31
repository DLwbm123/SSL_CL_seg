"""Run unchanged permitted regressions and compile the real kernel synthetically."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v2.precision import attach_gradient_student
from .diagnostic import compute_pair, LIMITS, ROW_COUNTS
from .core import require
from .run import authority, ROOT, REPO


def synthetic_pair():
    from tests.di_dmpa_gate1c_v2.test_core import Tiny
    reg, p = authority()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = Tiny().eval()
    models = dict(student=model, ema_teacher=copy.deepcopy(model).requires_grad_(False))
    attach_gradient_student(models, dict(diagnostic_precision="float64_shadow", _precision_contract_verified=True))
    generator = torch.Generator().manual_seed(19)
    inputs = (torch.randn(2, 3, 8, 8, generator=generator), torch.randn(2, 3, 8, 8, generator=generator),
              torch.arange(128).reshape(2, 8, 8) % 3)
    current = np.repeat(np.eye(16, dtype=np.float64)[:3, None], 2, axis=1)
    history = np.empty((3, 0, 16), np.float64)
    return models, torch.ones(3, 16), current, history, p["gradient_diagnostic"]["batch_pairs"][0], inputs


def compile_call_graph(code_commit):
    result, _ = compute_pair(*synthetic_pair())
    reg, _ = authority()
    per_pair = dict(result["counts"], total_forwards=result["counts"]["native_forwards"]+result["counts"]["fp64_forwards"], **result["output_rows"])
    require(per_pair == reg["call_graph"]["per_pair"], "synthetic compiler mismatch", "BLOCKED_CALL_GRAPH_MISMATCH")
    source = {str(path.relative_to(REPO)): d.sha256(path) for folder in (ROOT/"mmpr_gs_v0_1", ROOT/"tests/mmpr_gs_v0_1")
              for path in sorted(folder.glob("*.py"))}
    return dict(status="PASS", code_commit=code_commit, per_pair=per_pair, native_and_fp64_call_trace=result["call_trace"],
                exact_source_sha256=source, synthetic_model="existing tests.di_dmpa_gate1c_v2.test_core.Tiny",
                synthetic_shapes=[2, 3, 8, 8], data="synthetic_only", real_model_forwards=0,
                integration_forwards=15, formal_forwards=360, total_real_forward_budget=375,
                integration_autograd=dict(native=9, fp64=18), formal_autograd=dict(native=216, fp64=432),
                model_bank_rng_immutability=result["isolation"], complete_parameter_inventory=result["parameter_inventory"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    code = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(code == args.code_commit, "test exact source mismatch")
    authority()
    xml = args.output/"pytest.xml"
    command = ["-q", "-p", "no:cacheprovider", "tests/di_dmpa_gate1c_v2", "tests/di_dmpa_gate1c_v3", "tests/mmpr_gs_v0_1",
               "--ignore=tests/di_dmpa_gate1c_v2/test_real.py", "--ignore=tests/di_dmpa_gate1c_v3/test_baseline.py",
               "--junitxml", str(xml), "--basetemp", str(args.output/"scratch")]
    with (args.output/"pytest_output.txt").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), b.no_updates():
        result = pytest.main(command)
    suites = list(ET.parse(xml).iter("testsuite"))
    totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    status = "PASS" if result == 0 and totals["failures"] == totals["errors"] == totals["skipped"] == 0 else "FAIL"
    report = dict(status=status, code_commit=code, tests=totals["tests"], failures=totals["failures"], errors=totals["errors"],
                  skips=totals["skipped"], junit_path=str(xml), junit_sha256=d.sha256(xml), exit_code=int(result),
                  pytest_arguments=command, data="synthetic_only", real_model_forwards=0,
                  optimizer_construction=False, backward_called=False, historical_test_files_modified=False)
    d.write_new(args.output/"MMPR_GS_TEST_REPORT.json", report)
    aliases = []
    scratch = args.output/"scratch"
    for path in scratch.rglob("*"):
        if path.is_symlink():
            target = path.resolve()
            require(path.name.endswith("current") and target.is_relative_to(scratch.resolve()), "unexpected pytest alias")
            aliases.append(dict(path=str(path.relative_to(scratch)), target=str(target.relative_to(scratch))))
            path.unlink()
    d.write_new(args.output/"PYTEST_TEMPORARY_ALIASES.json", dict(aliases=aliases, payload_files_removed=0))
    print(json.dumps(report), flush=True)
    if status != "PASS":
        raise SystemExit(1)
    with b.no_updates():
        compiler = compile_call_graph(code)
    d.write_new(args.output/"MMPR_GS_CALL_GRAPH.json", compiler)


if __name__ == "__main__":
    main()
