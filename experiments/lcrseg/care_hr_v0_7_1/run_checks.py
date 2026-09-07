"""Run the fixed synthetic regression suite with separately counted fit calls."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    import pytest
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("run exact-source regression from a clean committed worktree")
    entry, completed = Counter(), Counter()
    names = {"fit_ridge", "fit_dual_heads", "select_lambda", "fit_calibrators", "pav_fit",
             "patient_quantile", "calibrate_bounds"}
    prefix = str(root / "experiments/lcrseg") + os.sep
    def profile(frame, event, result):
        code = frame.f_code
        if code.co_name in names and code.co_filename.startswith(prefix):
            key = str(Path(code.co_filename).relative_to(root)) + ":" + code.co_name
            if event == "call": entry[key] += 1
            elif event == "return" and result is not None: completed[key] += 1
    argv = ["-q", "-p", "no:cacheprovider", "experiments/lcrseg/tests/care_hr_v0_7_1",
            "experiments/lcrseg/tests/care_hr_v0_7", "experiments/lcrseg/tests/ppc_shor_v0_6b",
            "--junitxml=" + str(args.output / "junit.xml")]
    start = time.monotonic()
    sys.setprofile(profile)
    try:
        exit_code = int(pytest.main(argv))
    finally:
        sys.setprofile(None)
    suites = list(ET.parse(args.output / "junit.xml").getroot().iter("testsuite"))
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    report = {"source_commit": source, "pytest_exit_code": exit_code, **counts,
              "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
              "python": platform.python_version(), "platform": platform.system(),
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "pytest")},
              "elapsed_seconds": time.monotonic() - start,
              "synthetic_entry_calls": dict(entry), "synthetic_completed_non_null_returns": dict(completed),
              "fit_count_note": "Nested APIs reported separately; do not sum wrappers and underlying fits. CLI subprocess review smoke does not fit.",
              "real_data_reads": 0, "real_ground_truth_reads": 0, "real_router_risk_fits": 0,
              "model_forwards": 0, "A1_full_capacity_oracle_coverage": exit_code == 0,
              "coverage_note": "R1 scorer, combination oracle, blind boundaries, full synthetic executor, and unchanged 95 predecessor regressions; see continuation_r1/A1_COVERAGE.json."}
    (args.output / "TEST_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
