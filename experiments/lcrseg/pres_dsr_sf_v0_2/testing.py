"""Create the exact-source zero-skip PRES-DSR-SF pytest admission receipt."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from di_dmpa_gate1c_v3 import durable as d

from .core import require
from .protocol import source_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--pytest-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--exact-command", required=True)
    args = parser.parse_args()
    publication = source_gate(args.code_commit)
    root = ET.parse(args.junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    cases = [case for case in root.iter("testcase") if "pres_dsr_sf_v0_2" in case.attrib.get("classname", "")]
    require(len(cases) >= 82 and totals["failures"] == totals["errors"] == totals["skipped"] == 0,
            "PRES-DSR-SF/PRES/Gate0 test admission failed", "BLOCKED_INCOMPLETE_EVIDENCE")
    result = dict(status="PASS", code_commit=args.code_commit, publication=publication,
                  pres_dsr_sf_test_cases=len(cases), required_pres_dsr_sf_categories=82, **totals,
                  junit_path=str(args.junit.resolve()), junit_sha256=d.sha256(args.junit),
                  pytest_output_path=str(args.pytest_output.resolve()), pytest_output_sha256=d.sha256(args.pytest_output),
                  exact_test_command=args.exact_command, created_at=d.now(), skips=0)
    d.write_new(args.output, result)


if __name__ == "__main__":
    main()
