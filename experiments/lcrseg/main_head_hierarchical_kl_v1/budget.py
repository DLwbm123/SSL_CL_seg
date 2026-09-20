"""Study-global CPU quota ledger; missing or damaged ledgers are hard stops."""
import json
import os
import tempfile
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - the qualified server is POSIX
    fcntl = None

STUDY = "MAIN_HEAD_HIERARCHICAL_KL_V1"
MAX_ATTEMPTS = 2
MAX_TOTAL = 32
MAX_PER_ATTEMPT = 16
SCHEMA = 1


def _locked(path):
    if fcntl is None:
        raise RuntimeError("persistent quota requires a POSIX file lock")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = path.with_suffix(path.suffix + ".lock").open("a+")
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    return fd


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("CPU quota ledger missing; refusing to reset budget")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError("CPU quota ledger is unreadable") from exc
    if (value.get("schema") != SCHEMA or value.get("study_id") != STUDY
            or type(value.get("attempts")) is not list
            or type(value.get("optimizer_calls")) is not int
            or not 0 <= value["optimizer_calls"] <= MAX_TOTAL
            or len(value["attempts"]) > MAX_ATTEMPTS):
        raise RuntimeError("CPU quota ledger binding or counters are invalid")
    for attempt in value["attempts"]:
        if (type(attempt) is not dict or type(attempt.get("optimizer_calls")) is not int
                or not 0 <= attempt["optimizer_calls"] <= MAX_PER_ATTEMPT):
            raise RuntimeError("CPU quota ledger attempt is invalid")
    if sum(a["optimizer_calls"] for a in value["attempts"]) != value["optimizer_calls"]:
        raise RuntimeError("CPU quota ledger total does not match attempts")
    return value


def _write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temp = stream.name
    os.replace(temp, path)


def reserve_attempt(path, label):
    """Atomically reserve one study attempt; output directories do not matter."""
    path = Path(path)
    lock = _locked(path)
    try:
        value = _read(path)
        if len(value["attempts"]) >= MAX_ATTEMPTS:
            raise RuntimeError("CPU study attempts exhausted")
        if value["optimizer_calls"] >= MAX_TOTAL:
            raise RuntimeError("CPU study optimizer quota exhausted")
        attempt = {"label": str(label), "optimizer_calls": 0, "status": "STARTED"}
        value["attempts"].append(attempt)
        _write(path, value)
        return len(value["attempts"]) - 1
    finally:
        lock.close()


def charge(path, attempt_index):
    """Charge before invoking an optimizer; partial failures remain durable."""
    path = Path(path)
    lock = _locked(path)
    try:
        value = _read(path)
        if not 0 <= attempt_index < len(value["attempts"]):
            raise RuntimeError("CPU attempt is not reserved")
        attempt = value["attempts"][attempt_index]
        if attempt["optimizer_calls"] >= MAX_PER_ATTEMPT or value["optimizer_calls"] >= MAX_TOTAL:
            raise RuntimeError("CPU optimizer quota exhausted")
        attempt["optimizer_calls"] += 1
        value["optimizer_calls"] += 1
        _write(path, value)
        return value["optimizer_calls"]
    finally:
        lock.close()


def close_attempt(path, attempt_index, status):
    path = Path(path)
    lock = _locked(path)
    try:
        value = _read(path)
        if not 0 <= attempt_index < len(value["attempts"]):
            raise RuntimeError("CPU attempt is not reserved")
        if status not in ("PASS", "FAIL"):
            raise ValueError("invalid CPU attempt status")
        value["attempts"][attempt_index]["status"] = status
        _write(path, value)
    finally:
        lock.close()
