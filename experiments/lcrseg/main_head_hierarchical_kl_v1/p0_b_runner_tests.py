"""Default-deny mock checks; no torch, payload or model import."""
import json
import tempfile
from pathlib import Path

from .p0_b_runner import GateError, validate_preflight


def main():
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
