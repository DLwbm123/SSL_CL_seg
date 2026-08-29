#!/usr/bin/env python3
"""Remove only stale M&Ms failure bundles after an accepted zero-failure rerun."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcrseg.common import DATA_ROOT, PROJECT_ROOT, read_csv, utc_now, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Required to unlink stale generated JSON failure bundles.")
    args = parser.parse_args()
    h5_rows = read_csv(DATA_ROOT / "manifests" / "diagnostics" / "mnms_h5.csv")
    failed = [row for row in h5_rows if row.get("status") not in {"written", "skipped"}]
    if failed:
        raise SystemExit("M&Ms still has active HDF5 failures; refusing stale-failure cleanup")
    failure_root = PROJECT_ROOT / "reports" / "preprocessing" / "failures" / "mnms"
    targets = sorted(path for path in failure_root.glob("*.json") if path.is_file() and path.parent == failure_root)
    error_counts = Counter()
    for path in targets:
        try:
            error = json.loads(path.read_text(encoding="utf-8")).get("error", "unknown")
            category = (
                "recovered_idempotent_hdf5_validation_bug"
                if error.startswith("Existing target is invalid and will not be overwritten:")
                else "recovered_auxiliary_metadata_phase_empty_label"
                if error == "empty cardiac foreground after preprocessing"
                else "other_recovered_failure"
            )
            error_counts[category] += 1
        except Exception as exc:
            raise SystemExit(f"refusing to clean unreadable failure bundle {path}: {exc}") from exc
    summary = {
        "generated_at": utc_now(),
        "active_mnms_h5_failures": 0,
        "stale_failure_bundle_count": len(targets),
        "error_counts": dict(error_counts),
        "action": "removed" if args.execute else "dry_run",
        "scope": str(failure_root),
        "reason": "The final idempotent M&Ms rerun accepted 690/690 HDF5 pairs after fixing a local validate_h5_file(None) bug.",
    }
    if args.execute:
        for path in targets:
            path.unlink()
    write_json(PROJECT_ROOT / "reports" / "preprocessing" / "retry_history" / "mnms_recovered_failure_cleanup.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
