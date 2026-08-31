"""Standard-library detached parent, create-only receipts, and byte manifests."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import traceback


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Publish complete bytes without replacing any existing file or symlink."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def file_entries(root, *, exclude=()):
    root = Path(root).resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in artifact bundle: {path}")
        if not path.is_file() or path.relative_to(root).as_posix() in exclude:
            continue
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"artifact changed while hashing: {path}")
        entries.append(dict(path=path.relative_to(root).as_posix(), bytes=after.st_size, sha256=digest))
    return entries


def seal(root, filename="PRIVATE_BUNDLE_MANIFEST.json"):
    root = Path(root).resolve()
    if Path(filename).name != filename:
        raise ValueError("manifest must be directly inside its bundle")
    entries = file_entries(root, exclude=(filename,))
    result = dict(schema_version=1, created_at=now(), entries=entries,
                  content_sha256=hashlib.sha256(canonical(entries)).hexdigest(),
                  files=len(entries), bytes=sum(e["bytes"] for e in entries))
    write_new(root / filename, result)
    return result


def verify(root, filename="PRIVATE_BUNDLE_MANIFEST.json", *, exact=True):
    root = Path(root).resolve()
    if Path(filename).name != filename or (root / filename).is_symlink():
        raise ValueError("unsafe manifest path")
    manifest = read(root / filename)
    entries = manifest["entries"]
    if hashlib.sha256(canonical(entries)).hexdigest() != manifest["content_sha256"]:
        raise RuntimeError("manifest content hash mismatch")
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in seen:
            raise RuntimeError("unsafe or duplicate manifest entry")
        seen.add(entry["path"])
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError("manifest entry escapes bundle")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"artifact byte/hash mismatch: {relative}")
    if exact and {e["path"] for e in file_entries(root, exclude=(filename,))} != seen:
        raise RuntimeError("manifest does not cover the entire bundle")
    if len(entries) != manifest["files"] or sum(e["bytes"] for e in entries) != manifest["bytes"]:
        raise RuntimeError("manifest totals mismatch")
    return manifest


def process_identity(pid):
    stat = Path(f"/proc/{pid}/stat")
    text = stat.read_text() if stat.exists() else None
    # /proc comm can contain spaces; starttime is field 22, after the final ')'.
    return dict(pid=pid, start_ticks=text.rsplit(")", 1)[1].split()[19] if text else None)


def launch(output, phase, command, *, cwd, env=None):
    output = Path(output)
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"create-only execution already exists: {output}")
    if not re.fullmatch(r"[A-Za-z0-9_]+", phase) or not command:
        raise ValueError("invalid phase or empty command")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    output = output.resolve()
    overrides = dict(env or {})
    for key in overrides:
        if not re.fullmatch(r"(CUDA_VISIBLE_DEVICES|CUBLAS_WORKSPACE_CONFIG|PYTHONPATH|PYTHONHASHSEED|PYTHONDONTWRITEBYTECODE|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|GATE0_[A-Z_]+|V3_[A-Z_]+)", key):
            raise ValueError(f"unregistered environment override: {key}")
    request = dict(schema_version=1, requested_at=now(), command=list(command), cwd=str(Path(cwd).resolve()),
                   phase=phase, env=overrides, supervisor_file_sha256=sha256(__file__))
    write_new(output / "LAUNCH_REQUEST.json", request)
    with (output / "supervisor.log").open("xb") as log:
        parent = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "_supervise", "--output", str(output)],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, close_fds=True)
    receipt = dict(launched_at=now(), supervisor=process_identity(parent.pid), output=str(output),
                   detached=True, ssh_exit_is_child_exit_evidence=False)
    write_new(output / "LAUNCH_RECEIPT.json", receipt)
    return receipt


def supervise(output):
    output = Path(output).resolve()
    request = read(output / "LAUNCH_REQUEST.json")
    if sha256(__file__) != request["supervisor_file_sha256"]:
        raise RuntimeError("supervisor source changed after launch")
    identity = dict(hostname=socket.gethostname(), uid=os.getuid(), supervisor=process_identity(os.getpid()))
    write_new(output / "PROCESS_START.json", dict(started_at=now(), request=request, **identity))
    child = None
    error = None
    returncode = None
    with (output / "controller.log").open("xb") as log:
        try:
            env = dict(os.environ)
            env.update(PYTHONDONTWRITEBYTECODE="1")
            env.update(request["env"])
            child = subprocess.Popen(request["command"], cwd=request["cwd"], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     close_fds=True)
            write_new(output / "PROCESS_PID.json", dict(child=process_identity(child.pid), **identity))
            returncode = child.wait()
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            if child is not None:
                returncode = child.wait()
            else:
                write_new(output / "PROCESS_PID.json", dict(child=None, **identity))
        log.flush()
        os.fsync(log.fileno())
    exit_receipt = dict(exited_at=now(), actual_child_exit_code=returncode,
                        child_pid=child.pid if child else None, error=error, written_by_server_local_parent=True, **identity)
    write_new(output / "PROCESS_EXIT.json", exit_receipt)
    status = "COMMAND_COMPLETED" if returncode == 0 and error is None else "COMMAND_FAILED"
    done = dict(completed_at=now(), status=status, actual_child_exit_code=returncode,
                process_exit_sha256=sha256(output / "PROCESS_EXIT.json"), scientific_admission=None,
                ssh_exit_is_child_exit_evidence=False)
    write_new(output / "EXECUTION_COMPLETION.json", done)
    phase = request["phase"]
    write_new(output / f"PHASE_{phase}.json", dict(phase=phase, **done))
    # Launcher publishes its PID receipt immediately; include it in the closed bundle.
    import time
    for _ in range(100):
        if (output / "LAUNCH_RECEIPT.json").is_file():
            break
        time.sleep(0.01)
    if not (output / "LAUNCH_RECEIPT.json").is_file():
        raise RuntimeError("launcher receipt was not published")
    seal(output, f"PHASE_{phase}_MANIFEST.json")
    return 0 if status == "COMMAND_COMPLETED" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "_supervise", "seal", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--manifest", default="PRIVATE_BUNDLE_MANIFEST.json")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if args.mode == "_supervise":
        raise SystemExit(supervise(args.output))
    if args.mode == "launch":
        result = launch(args.output, args.phase, command, cwd=args.cwd, env=dict(x.split("=", 1) for x in args.env))
    elif args.mode == "seal":
        result = seal(args.output, args.manifest)
    else:
        result = verify(args.output, args.manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
