"""Bound P0/P1 dispatch with injectable native hooks; no implicit training."""
from dataclasses import dataclass
from pathlib import Path
import json

from .p0_b_runner import GateError, _json, validate_content, validate_preflight


class ExecutionError(RuntimeError):
    pass


@dataclass
class NativeHooks:
    """Native runtime callbacks supplied by the reviewed B2 execution environment."""
    load_state: object
    load_batches: object
    forward: object
    diagnose: object
    write_report: object


class NativeBackend:
    """Concrete callback adapter; it never substitutes a synthetic model for native hooks."""
    def __init__(self, hooks):
        if not isinstance(hooks, NativeHooks):
            raise TypeError("NativeBackend requires bound NativeHooks")
        self.hooks = hooks
        self.events = []
        self.synthetic = False

    def state(self, state):
        value = self.hooks.load_state(state)
        self.events.append(("state", state["state_id"]))
        return value

    def batches(self, handle, state, binding):
        value = tuple(self.hooks.load_batches(handle, state, binding))
        self.events.append(("batches", state["state_id"]))
        return value

    def forward(self, role, handle, state, batch):
        value = self.hooks.forward(role, handle, state, batch)
        self.events.append(("forward", role, state["state_id"]))
        return value

    def diagnose(self, state, handle, batches, outputs):
        return self.hooks.diagnose(state, handle, batches, outputs)

    def report(self, result, path):
        return self.hooks.write_report(result, path)


class MockBackend(NativeBackend):
    """Test backend; its outputs are explicit synthetic events, never scientific evidence."""
    def __init__(self, hooks):
        super().__init__(hooks)
        self.synthetic = True


def run_p0(binding_path, budget_path, contract_path, plan_path, *, backend, runner_sha, plan_sha, budget_sha, report_path=None):
    validate_content(binding_path, budget_path, contract_path, plan_path)
    approved = validate_preflight(binding_path, budget_path, contract_path,
                                  runner_sha=runner_sha, plan_sha=plan_sha, budget_sha=budget_sha,
                                  plan_path=plan_path)
    binding, budget = _json(binding_path), _json(budget_path)
    outputs = []
    outputs_diagnostics = []
    for state, row in zip(binding["states"], budget["per_state"]):
        if state["state_id"] != row["state_id"]:
            raise GateError("P0-B state/budget identity mismatch")
        handle = backend.state(state)
        batches = backend.batches(handle, state, binding["batch_binding"])
        if not batches:
            raise ExecutionError("P0-B batch binding produced no batches")
        state_outputs = []
        for role in ("student_full_forwards", "teacher_full_forwards"):
            if row[role] != len(batches):
                raise ExecutionError(f"{role} count does not match bound batches")
            for batch in batches:
                state_outputs.append(backend.forward(role, handle, state, batch))
        outputs.extend(state_outputs)
        diagnosis = backend.diagnose(state, handle, batches, state_outputs)
        if not isinstance(diagnosis, dict):
            raise ExecutionError("P0-B diagnosis must be a structured result")
        outputs_diagnostics.append(diagnosis)
    if len(outputs) != approved["forward_count"]:
        raise ExecutionError("P0-B forward accounting mismatch")
    endpoint_gates = {d.get("state_id"): d.get("gate") for d in outputs_diagnostics}
    required = {"O1_STAGE2_ENDPOINT", "O2_STAGE2_ENDPOINT"}
    if not required <= set(endpoint_gates):
        raise ExecutionError("P0-B diagnosis lacks both endpoint gates")
    science_status = "PASS" if all(endpoint_gates[s] == "PASS" for s in required) else "FAIL"
    result = {"status": "P0_RUNTIME_COMPLETE", "science_status": science_status,
              "synthetic": backend.synthetic, "forward_count": len(outputs),
              "optimizer_calls": 0, "diagnostics": outputs_diagnostics,
              "endpoint_gates": endpoint_gates}
    if report_path is not None:
        backend.report(result, Path(report_path))
    return result
