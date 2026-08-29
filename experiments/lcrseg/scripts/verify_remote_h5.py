#!/usr/bin/env python3
"""Read-only remote HDF5 readability and metadata check after checksum verification."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap


REMOTE_VALIDATOR = textwrap.dedent(
    r'''
    import json
    import sys
    from pathlib import Path
    import h5py
    import numpy as np

    root = Path(sys.argv[1])
    errors = []
    files = sorted(path for path in (root / "h5").rglob("*.h5") if path.is_file()) if (root / "h5").is_dir() else []
    if not files:
        errors.append("no HDF5 files found")
    for path in files:
        try:
            parts = path.relative_to(root / "h5").parts
            kind = parts[1] if len(parts) >= 5 else ""
            expected = "image" if kind == "images" else "label" if kind == "labels" else ""
            if not expected:
                raise ValueError("unexpected HDF5 layout")
            with h5py.File(path, "r") as handle:
                if list(handle.keys()) != [expected]:
                    raise ValueError("unexpected dataset list: %s" % list(handle.keys()))
                for attr in ("case_id", "dataset", "preprocess_version", "h5_schema_version", "preprocess_config_sha256"):
                    if attr not in handle.attrs:
                        raise ValueError("missing attr: %s" % attr)
                array = handle[expected][...]
                if expected == "image" and not np.isfinite(array).all():
                    raise ValueError("NaN/Inf image values")
                if expected == "label" and "allowed_labels" not in handle.attrs:
                    raise ValueError("label lacks allowed_labels")
        except Exception as exc:
            errors.append("%s: %s: %s" % (path, type(exc).__name__, exc))
    print(json.dumps({"root": str(root), "h5_files": len(files), "valid": not errors, "errors": errors}, sort_keys=True))
    raise SystemExit(0 if not errors else 1)
    '''
).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a read-only HDF5 check in an explicit remote Python runtime.")
    parser.add_argument("--remote-host", default="jiangsuiyang")
    parser.add_argument("--ssh-user", default="")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--bind-address", default="")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", required=True, help="Absolute remote Python interpreter with h5py and numpy.")
    args = parser.parse_args()
    if not args.remote_root.startswith("/") or args.remote_root == "/":
        raise SystemExit("--remote-root must be a non-root absolute path")
    if not (1 <= args.ssh_port <= 65535):
        raise SystemExit("--ssh-port must be between 1 and 65535")
    command = " ".join(
        (
            shlex.quote(args.remote_python),
            "-c",
            shlex.quote(REMOTE_VALIDATOR),
            shlex.quote(args.remote_root),
        )
    )
    target = f"{args.ssh_user}@{args.remote_host}" if args.ssh_user else args.remote_host
    ssh_args = ["ssh", "-p", str(args.ssh_port)]
    if args.bind_address:
        ssh_args.extend(("-4", "-b", args.bind_address))
    result = subprocess.run([*ssh_args, target, command], text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
