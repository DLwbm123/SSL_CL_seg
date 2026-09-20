import json
import tempfile
from pathlib import Path
from .execution import MockBackend, NativeHooks, run_p0
from .p0_b_runner import GateError
from .p1_controller import P1ExecutionError, run_p1


def _p0_fixture(root):
    states = []
    rows = []
    for i, (sid, count) in enumerate((('O1_STAGE2_START', 18), ('O1_STAGE2_ENDPOINT', 18),
                                       ('O2_STAGE2_START', 24), ('O2_STAGE2_ENDPOINT', 24))):
        states.append({"state_id": sid, "student": {"status": "BOUND"},
                       "dense_ema": {"status": "BOUND"}, "prototypes": {"status": "BOUND"}})
        rows.append({"state_id": sid, "student_full_forwards": count // 2,
                     "teacher_full_forwards": count // 2,
                     "prototype_initialization_forwards": 0, "readout_only_forwards": 0,
                     "vjp": 0, "total": count})
    binding = {"status": "BOUND", "execution_approval": "P0_B_EXECUTION_APPROVED",
               "states": states, "batch_binding": {"status": "BOUND_PRIVATE"},
               "payload_reads": 0, "model_forwards": 0, "torch_load": False}
    roles = {"student_full_forwards": 42, "teacher_full_forwards": 42,
             "prototype_initialization_forwards": 0, "readout_only_forwards": 0, "vjp": 0}
    budget = {"status": "BOUND", "metadata_plan_sha256": "", "forward_roles": roles,
              "total_forward_count": 84, "per_state": rows}
    contract = {"status": "FROZEN_PREPARATION_CONTRACT"}
    plan = {}
    for name, value in (("binding.json", binding), ("budget.json", budget),
                        ("contract.json", contract), ("plan.json", plan)):
        (root / name).write_text(json.dumps(value))
    import hashlib
    plan_sha = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    budget["metadata_plan_sha256"] = plan_sha
    binding["approved_plan_sha256"] = plan_sha
    (root / "budget.json").write_text(json.dumps(budget))
    (root / "binding.json").write_text(json.dumps(binding))
    return [root / x for x in ("binding.json", "budget.json", "contract.json", "plan.json")]


def test_p0_mock_accounting():
    with tempfile.TemporaryDirectory() as d:
        paths = _p0_fixture(Path(d))
        hooks = NativeHooks(lambda s: s, lambda s, b: b, lambda r, s, b: (r, s, b),
                            lambda s, b, o: None, lambda value, path: path.write_text(json.dumps(value)))
        backend = MockBackend(hooks)
        from .p0_b_runner import _sha
        result = run_p0(*paths[:3], paths[3], backend=backend,
                        runner_sha=_sha(Path(__file__).with_name("p0_b_runner.py")),
                        plan_sha=_sha(paths[3]), budget_sha=_sha(paths[1]))
        assert result["forward_count"] == 84 and result["synthetic"]


if __name__ == "__main__":
    test_p0_mock_accounting()
    print("execution_tests: PASS")
