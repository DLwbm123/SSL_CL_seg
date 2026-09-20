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
REGISTRY_SCHEMA = 1
REGISTRY_ID = "MAIN_HEAD_HKL_LEDGER_REGISTRY_R1"
LEDGER_ID = "MAIN_HEAD_HIERARCHICAL_KL_V1_CPU_LEDGER_R1"
AUTHORIZATION_ID = "MAIN_HEAD_HKL_PREPARATION_AUTH_R1"
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


def _registry(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("CPU quota ledger registry missing; refusing to choose a ledger")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError("CPU quota ledger registry is unreadable") from exc
    required = {
        "schema": REGISTRY_SCHEMA,
        "registry_id": REGISTRY_ID,
        "study_id": STUDY,
        "ledger_id": LEDGER_ID,
        "authorization_id": AUTHORIZATION_ID,
    }
    if any(value.get(k) != v for k, v in required.items()):
        raise RuntimeError("CPU quota ledger registry identity is invalid")
    canonical = value.get("canonical_ledger_path")
    evidence = value.get("historical_evidence_bindings")
    if not isinstance(canonical, str) or not Path(canonical).is_absolute():
        raise RuntimeError("CPU quota ledger registry path is invalid")
    if type(value.get("historical_optimizer_calls")) is not int or value["historical_optimizer_calls"] != 15:
        raise RuntimeError("CPU quota ledger historical binding is invalid")
    if type(value.get("historical_attempts")) is not int or value["historical_attempts"] != 1:
        raise RuntimeError("CPU quota ledger historical attempt binding is invalid")
    if type(evidence) is not list or not evidence:
        raise RuntimeError("CPU quota ledger historical evidence binding is missing")
    return value


def resolve_bound_ledger(registry_path, requested_path):
    registry = _registry(registry_path)
    canonical = Path(registry["canonical_ledger_path"]).resolve(strict=False)
    requested = Path(requested_path).resolve(strict=False)
    if requested != canonical:
        raise RuntimeError("CPU quota ledger path is not the registered canonical ledger")
    _read(canonical, registry)
    return canonical


def _read(path, registry=None):
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
    if registry is not None:
        identity = {
            "registry_id": registry["registry_id"],
            "ledger_id": registry["ledger_id"],
            "authorization_id": registry["authorization_id"],
            "source_commit": registry.get("source_commit"),
            "historical_evidence": registry.get("historical_evidence"),
            "historical_evidence_bindings": registry["historical_evidence_bindings"],
            "historical_optimizer_calls": registry["historical_optimizer_calls"],
            "historical_attempts": registry["historical_attempts"],
        }
        if any(value.get(k) != expected for k, expected in identity.items()):
            raise RuntimeError("CPU quota ledger identity does not match registered ledger")
        if value["optimizer_calls"] < registry["historical_optimizer_calls"]:
            raise RuntimeError("CPU quota ledger historical consumption was reduced")
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


def reserve_attempt(path, label, *, registry_path):
    """Atomically reserve one study attempt; output directories do not matter."""
    path = resolve_bound_ledger(registry_path, path)
    registry = _registry(registry_path)
    lock = _locked(path)
    try:
        value = _read(path, registry)
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


def charge(path, attempt_index, *, registry_path):
    """Charge before invoking an optimizer; partial failures remain durable."""
    path = resolve_bound_ledger(registry_path, path)
    registry = _registry(registry_path)
    lock = _locked(path)
    try:
        value = _read(path, registry)
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


def close_attempt(path, attempt_index, status, *, registry_path):
    path = resolve_bound_ledger(registry_path, path)
    registry = _registry(registry_path)
    lock = _locked(path)
    try:
        value = _read(path, registry)
        if not 0 <= attempt_index < len(value["attempts"]):
            raise RuntimeError("CPU attempt is not reserved")
        if status not in ("PASS", "FAIL"):
            raise ValueError("invalid CPU attempt status")
        value["attempts"][attempt_index]["status"] = status
        _write(path, value)
    finally:
        lock.close()
