"""Default-deny mock checks; no torch, payload or model import."""
import json
import tempfile
from pathlib import Path

from .p0_b_runner import GateError, _validate_budget, validate_preflight


def main():
    valid = {
        "total_forward_count": 84,
        "forward_roles": {"student_full_forwards": 42, "teacher_full_forwards": 42,
                           "prototype_initialization_forwards": 0, "readout_only_forwards": 0,
                           "vjp": 0},
        "per_state": [
            {"state_id": "O1_STAGE2_START", "student_full_forwards": 9, "teacher_full_forwards": 9,
             "prototype_initialization_forwards": 0, "readout_only_forwards": 0, "vjp": 0, "total": 18},
            {"state_id": "O1_STAGE2_ENDPOINT", "student_full_forwards": 9, "teacher_full_forwards": 9,
             "prototype_initialization_forwards": 0, "readout_only_forwards": 0, "vjp": 0, "total": 18},
            {"state_id": "O2_STAGE2_START", "student_full_forwards": 12, "teacher_full_forwards": 12,
             "prototype_initialization_forwards": 0, "readout_only_forwards": 0, "vjp": 0, "total": 24},
            {"state_id": "O2_STAGE2_ENDPOINT", "student_full_forwards": 12, "teacher_full_forwards": 12,
             "prototype_initialization_forwards": 0, "readout_only_forwards": 0, "vjp": 0, "total": 24},
        ],
    }
    assert _validate_budget(valid) == 84
    for bad in ({**valid, "total_forward_count": 42},
                {**valid, "forward_roles": {**valid["forward_roles"], "student_full_forwards": 41}},
                {**valid, "per_state": [{**valid["per_state"][0], "total": 17}] + valid["per_state"][1:]}):
        try:
            _validate_budget(bad)
        except GateError:
            pass
        else:
            raise AssertionError("invalid P0 budget was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binding = root / "binding.json"
        budget = root / "budget.json"
        contract = root / "contract.json"
        binding.write_text(json.dumps({"status": "METADATA_ONLY_UNBOUND", "execution_approval": False,
                                       "payload_reads": 0, "model_forwards": 0, "torch_load": False}))
        budget.write_text(json.dumps({"status": "UNBOUND", "total_forward_count": None}))
        contract.write_text(json.dumps({"status": "FROZEN_PREPARATION_CONTRACT"}))
        try:
            validate_preflight(binding, budget, contract,
                               runner_sha="x", plan_sha="x", budget_sha="x")
        except GateError as exc:
            assert "not BOUND" in str(exc)
        else:
            raise AssertionError("unbound P0-B was admitted")
    print("P0-B default-deny mock PASS: no payload/model/optimizer activity")


if __name__ == "__main__":
    main()
