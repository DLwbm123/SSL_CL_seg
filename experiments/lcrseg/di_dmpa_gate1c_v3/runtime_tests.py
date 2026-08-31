"""Run the unchanged Gate0 tests plus v3 checks in the audited runtime."""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from .durable import sha256, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    # The three original read-only data tests use a module-level old-host path.
    # Resolve only that operational path; do not modify any old test or data file.
    from tests.gate0 import test_manifest_adapter
    test_manifest_adapter.DATA_ROOT = args.data_root
    os.environ.update(GATE0_RESUME_ACTUAL_MODEL="1", GATE0_RESUME_DEVICE="cuda",
                      GATE0_RESUME_REPORT_DIR=str(args.output / "resume_trajectories"),
                      V3_TEST_ACTUAL_MODEL="1", V3_TEST_DEVICE="cuda",
                      V3_TEST_REPORT_DIR=str(args.output / "v3_adapter_report"))
    xml_path = args.output / "pytest.xml"
    command = ["-q", "-p", "no:cacheprovider", "tests/gate0", "tests/di_dmpa_gate1c_v3",
               "--junitxml", str(xml_path), "--basetemp", str(args.output / "pytest_artifacts")]
    exit_code = pytest.main(command)
    cases = []
    for case in ET.parse(xml_path).iter("testcase"):
        status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else (
            "SKIP" if case.find("skipped") is not None else "PASS")
        cases.append(dict(name=case.get("name"), status=status))
    skipped = [row["name"] for row in cases if row["status"] == "SKIP"]
    expected_skip = ["test_interrupted_resume_matches_uninterrupted_six_step_trajectory"]
    passed = exit_code == 0 and skipped == expected_skip
    report = dict(status="PASS_WITH_DISCLOSED_LEGACY_PATH_SKIP" if passed else "FAIL", exit_code=int(exit_code),
                  passed=sum(row["status"] == "PASS" for row in cases), failed=sum(row["status"] == "FAIL" for row in cases),
                  skipped=skipped, skip_reason="old /root/LCRSeg TinySegNet six-step test; four production-model resume trajectories and v3 three-stage resume/independent audit run instead",
                  original_test_path_override=str(args.data_root), cases=cases, pytest_arguments=command,
                  junit_sha256=sha256(xml_path), formal_B0_training=False, diagnostic_budget_consumed=0)
    write_new(args.output / "V3_RUNTIME_TEST_REPORT.json", report)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
