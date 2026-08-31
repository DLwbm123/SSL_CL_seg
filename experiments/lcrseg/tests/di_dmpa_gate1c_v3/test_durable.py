"""One end-to-end check of exit provenance, immutability and byte verification."""
import sys
import time
import json
import subprocess

import pytest

from di_dmpa_gate1c_v3 import durable as d


def test_detached_parent_and_archive_guards(tmp_path):
    output = tmp_path / "process"
    command = [sys.executable, "-c", "import time; time.sleep(0.2); print('owned child'); raise SystemExit(7)"]
    invoked = subprocess.run([sys.executable, "-B", d.__file__, "launch", "--output", str(output),
                              "--phase", "owned_test", "--cwd", str(tmp_path), "--", *command],
                             capture_output=True, text=True, check=True)
    launch = json.loads(invoked.stdout)
    manifest = output / "PHASE_owned_test_MANIFEST.json"
    deadline = time.monotonic() + 15
    while not manifest.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert manifest.exists(), (output / "supervisor.log").read_text()
    exit_receipt = d.read(output / "PROCESS_EXIT.json")
    assert exit_receipt["actual_child_exit_code"] == 7
    assert exit_receipt["written_by_server_local_parent"] is True
    assert exit_receipt["supervisor"]["pid"] == launch["supervisor"]["pid"]
    assert d.read(output / "PROCESS_PID.json")["child"]["pid"] == exit_receipt["child_pid"]
    assert d.read(output / "EXECUTION_COMPLETION.json")["status"] == "COMMAND_FAILED"
    assert "owned child" in (output / "controller.log").read_text()
    d.verify(output, manifest.name)
    with pytest.raises(FileExistsError):
        d.launch(output, "owned_test", command, cwd=tmp_path)
    with pytest.raises(FileExistsError):
        d.write_new(output / "PROCESS_EXIT.json", {"actual_child_exit_code": 0})
    assert d.read(output / "PROCESS_EXIT.json")["actual_child_exit_code"] == 7
    bundle = d.seal(output)
    assert d.verify(output)["content_sha256"] == bundle["content_sha256"]
    log = output / "controller.log"
    original = log.read_bytes()
    log.write_bytes(original + b"tampered")
    with pytest.raises(RuntimeError, match="byte/hash mismatch"):
        d.verify(output)
    log.write_bytes(original)
    (output / "escaped").symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="symlink"):
        d.verify(output)
