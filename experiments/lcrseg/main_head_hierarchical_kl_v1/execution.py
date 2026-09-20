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

    def state(self, state):
        value = self.hooks.load_state(state)
        self.events.append(("state", state["state_id"]))
        return value

    def batches(self, state, binding):
        value = self.hooks.load_batches(state, binding)
        self.events.append(("batches", state["state_id"]))
        return value

    def forward(self, role, state, batch):
        value = self.hooks.forward(role, state, batch)
        self.events.append(("forward", role, state["state_id"]))
        return value

    def diagnose(self, state, batch, outputs):
        return self.hooks.diagnose(state, batch, outputs)

    def report(self, result, path):
        return self.hooks.write_report(result, path)


class MockBackend(NativeBackend):
    """Test backend; its outputs are explicit synthetic events, never scientific evidence."""
    pass


def run_p0(binding_path, budget_path, contract_path, plan_path, *, backend, runner_sha, plan_sha, budget_sha, report_path=None):
    validate_content(binding_path, budget_path, contract_path, plan_path)
    approved = validate_preflight(binding_path, budget_path, contract_path,
                                  runner_sha=runner_sha, plan_sha=plan_sha, budget_sha=budget_sha,
                                  plan_path=plan_path)
    binding, budget = _json(binding_path), _json(budget_path)
    outputs = []
    for state, row in zip(binding["states"], budget["per_state"]):
        if state["state_id"] != row["state_id"]:
            raise GateError("P0-B state/budget identity mismatch")
        payload = backend.state(state)
        batches = backend.batches(state, binding["batch_binding"])
        for role in ("student_full_forwards", "teacher_full_forwards"):
            for _ in range(row[role]):
                outputs.append(backend.forward(role, state, batches))
        backend.diagnose(state, batches, outputs[-row["total"]:])
    if len(outputs) != approved["forward_count"]:
        raise ExecutionError("P0-B forward accounting mismatch")
    result = {"status": "P0_RUNTIME_COMPLETE", "synthetic": isinstance(backend, MockBackend),
              "forward_count": len(outputs), "optimizer_calls": 0}
    if report_path is not None:
        backend.report(result, Path(report_path))
    return result
