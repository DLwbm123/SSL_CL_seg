#!/usr/bin/env python3
"""Run the real pytest suite and compile its actual JUnit results, including resume fixtures."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from di_dmpa_jascl.config import sha256_file
from di_dmpa_jascl.metrics import write_json
from di_dmpa_jascl.provenance import git_revision
from scripts.compile_gate0_reports import config_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-device", default="cuda")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "UNIT_INTEGRATION_TEST_REPORT.json").exists():
        raise RuntimeError("refusing to overwrite prior test evidence")
    _, hashes = config_contract()
    common = {"git_commit": git_revision(ROOT), "config_hashes": hashes}
    env = dict(os.environ, PYTHONPATH=str(ROOT), GATE0_RESUME_REPORT_DIR=str(out / "resume_trajectories"),
               GATE0_RESUME_ACTUAL_MODEL="1", GATE0_RESUME_DEVICE=args.resume_device)
    command = [sys.executable, "-m", "pytest", "-q", "tests/gate0", "--junitxml", str(out / "pytest.xml"),
               "--basetemp", str(out / "pytest_artifacts")]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    transcript = out / "pytest_output.txt"
    transcript.write_text(completed.stdout)
    print(completed.stdout)
    cases = []
    for case in ET.parse(out / "pytest.xml").iter("testcase"):
        status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else (
            "SKIP" if case.find("skipped") is not None else "PASS")
        cases.append({"name": case.get("name"), "class": case.get("classname"), "status": status})
    counts = {key: sum(row["status"] == key for row in cases) for key in ("PASS", "FAIL", "SKIP")}
    report = {**common, "status": "PASS" if completed.returncode == 0 and not counts["SKIP"] else "FAIL",
              "exit_code": completed.returncode, "passed": counts["PASS"], "failed": counts["FAIL"],
              "skipped": counts["SKIP"], "test_cases": cases, "command": command,
              "transcript_sha256": sha256_file(transcript), "junit_sha256": sha256_file(out / "pytest.xml"),
              "interpreter": sys.executable}
    write_json(out / "UNIT_INTEGRATION_TEST_REPORT.json", report)
    trajectories = {}
    for path in (out / "resume_trajectories").glob("*.json"):
        trajectories[path.stem] = json.loads(path.read_text())
    resume = {**common, "status": "PASS" if len(trajectories) == 4 and all(
        row["status"] == "PASS" for row in trajectories.values()) else "FAIL",
        "atol": 1e-6, "rtol": 1e-6, "trajectories": trajectories,
        "scope": "production UNet/JASCL classifier and runner on hashed synthetic HDF5 fixtures; not a formal Fundus result"}
    write_json(out / "RESUME_EQUIVALENCE_REPORT.json", resume)
    if report["status"] != "PASS" or resume["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
