import builtins
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

import care_hr_v0_7
from care_hr_v0_7.contracts import REVIEW_STATUS, ReviewBlocked
from care_hr_v0_7.io_guards import reject_dangerous_arguments, static_audit
from care_hr_v0_7 import executor


ROOT = Path(__file__).resolve().parents[4]
CLI = ROOT / "experiments/lcrseg/care_hr_v0_7_review.py"


@pytest.mark.parametrize("flag", ("--train", "--fit", "--evaluate", "--formal", "--data-root",
                                  "--nas-root", "--checkpoint", "--v0-6b-root"))
def test_every_danger_flag_is_locked(flag):
    with pytest.raises(ReviewBlocked, match=REVIEW_STATUS):
        reject_dangerous_arguments([flag])


def test_cli_rejects_data_flag_before_path_is_used(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(CLI), "--data-root", str(sentinel)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and REVIEW_STATUS in result.stderr
    assert not sentinel.exists()


def test_review_lock_file_contains_all_false_authorizations():
    value = json.loads((ROOT / "experiments/lcrseg/docs/care_hr_v0_7_review/CARE_HR_V0_7_REVIEW_LOCK.json").read_text())
    assert value["status"] == REVIEW_STATUS
    assert value["draft_state"] == "DRAFT_NOT_REGISTERED"
    assert value["training_authority"] == "NO_TRAINING_AUTHORITY"
    assert value["evaluation_authority"] == "NO_EVALUATION_AUTHORITY"
    assert value["training_authorized"] is False
    assert value["real_data_access_authorized"] is False
    assert value["evaluation_authorized"] is False
    assert value["formal_output_authorized"] is False


@pytest.mark.parametrize("mode", ("synthetic-tests", "print-contract", "static-audit"))
def test_only_review_modes_execute(mode):
    result = subprocess.run([sys.executable, str(CLI), "--mode", mode], cwd=ROOT,
                            env={**dict(__import__("os").environ), "PYTHONPATH": "experiments/lcrseg"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), dict)


def test_package_reload_performs_no_python_file_open(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("import attempted application-level file I/O")
    monkeypatch.setattr(builtins, "open", blocked)
    importlib.reload(care_hr_v0_7)


def test_static_audit_accepts_only_review_work_package():
    result = static_audit(ROOT)
    assert result["ok"], result["findings"]
    assert all(result["checks"].values())


@pytest.mark.parametrize("entry", (executor.train, executor.fit, executor.evaluate))
def test_real_executor_entries_always_call_review_lock(entry):
    with pytest.raises(ReviewBlocked, match=REVIEW_STATUS):
        entry()


def test_inference_modules_do_not_import_hdf5_dependency():
    for name in ("features", "policy", "proposals"):
        module = __import__(f"care_hr_v0_7.{name}", fromlist=[name])
        assert "h5py" not in inspect.getsource(module)
