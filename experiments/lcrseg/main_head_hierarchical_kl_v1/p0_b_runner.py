"""Default-deny, metadata-only P0-B gate; never loads payloads or models."""
import argparse
import hashlib
import json
from pathlib import Path


class GateError(RuntimeError):
    pass


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        raise GateError("metadata JSON is unreadable") from exc


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_content(binding_path, budget_path, contract_path, plan_path):
    """Bind the bytes being reviewed; caller must still supply external approval."""
    budget = _json(budget_path)
    actual_plan = _sha(plan_path)
    if budget.get("metadata_plan_sha256") != actual_plan:
        raise GateError("P0-B metadata plan digest does not match plan bytes")
    return {"plan_sha256": actual_plan, "budget_sha256": _sha(budget_path),
            "binding_sha256": _sha(binding_path), "contract_sha256": _sha(contract_path)}


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{name} must be a nonnegative integer")
    return value


def _validate_bound_states(binding):
    states = binding.get("states")
    if not isinstance(states, list) or len(states) != 4:
        raise GateError("P0-B requires exactly four bound states")
    ids = [s.get("state_id") for s in states]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != 4:
        raise GateError("P0-B state identities are not unique")
    for state in states:
        for role in ("student", "dense_ema", "prototypes"):
            if state.get(role, {}).get("status") not in ("BOUND", "FOUND_METADATA"):
                raise GateError(f"P0-B role {role} is not bound")


def _validate_budget(budget):
    total = _positive_int(budget.get("total_forward_count"), "total_forward_count")
    roles = budget.get("forward_roles")
    if not isinstance(roles, dict):
        raise GateError("P0-B forward roles are missing")
    values = [_nonnegative_int(v, f"forward_roles.{k}") for k, v in roles.items()]
    if sum(values) != total:
        raise GateError("P0-B forward role counts do not sum to total")
    per_state = budget.get("per_state")
    if not isinstance(per_state, list) or len(per_state) != 4:
        raise GateError("P0-B per-state budget is missing")
    seen = set()
    sums = {key: 0 for key in roles}
    for state in per_state:
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or state_id in seen:
            raise GateError("P0-B per-state identities are not unique")
        seen.add(state_id)
        state_total = _positive_int(state.get("total"), f"per_state.{state_id}.total")
        state_sum = 0
        for key in roles:
            value = _nonnegative_int(state.get(key), f"per_state.{state_id}.{key}")
            sums[key] += value
            state_sum += value
        if state_sum != state_total:
            raise GateError(f"P0-B per-state counts do not sum for {state_id}")
    if sums != {key: value for key, value in zip(roles, values)}:
        raise GateError("P0-B per-state role counts do not match role totals")
    if sum(state.get("total", 0) for state in per_state) != total:
        raise GateError("P0-B per-state totals do not match total")
    return total


def validate_preflight(binding_path, budget_path, contract_path, *, runner_sha, plan_sha, budget_sha,
                       plan_path=None):
    binding = _json(binding_path)
    budget = _json(budget_path)
    contract = _json(contract_path)
    if binding.get("status") != "BOUND":
        raise GateError("P0-B metadata binding is not BOUND")
    if binding.get("execution_approval") != "P0_B_EXECUTION_APPROVED":
        raise GateError("P0-B execution approval is missing")
    if budget.get("status") != "BOUND" or budget.get("total_forward_count") is None:
        raise GateError("P0-B forward budget is not bound")
    if contract.get("status") != "FROZEN_PREPARATION_CONTRACT":
        raise GateError("P0-B diagnostic contract is not frozen")
    _validate_bound_states(binding)
    total = _validate_budget(budget)
    if any(binding.get(k) for k in ("payload_reads", "model_forwards", "torch_load")):
        raise GateError("metadata binding reports forbidden payload activity")
    if runner_sha != _sha(Path(__file__)):
        raise GateError("runner SHA does not match approved runner")
    if plan_path is None:
        if plan_sha != binding.get("approved_plan_sha256"):
            raise GateError("plan SHA is not bound to metadata approval")
        if budget_sha != budget.get("approved_budget_sha256"):
            raise GateError("budget SHA is not bound to metadata approval")
    elif plan_sha != _sha(plan_path) or budget_sha != _sha(budget_path):
        raise GateError("approved content digest does not match reviewed bytes")
    return {"status": "P0_B_EXECUTION_APPROVED", "forward_count": total}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--runner-sha", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--budget-sha", required=True)
    args = parser.parse_args(argv)
    result = validate_preflight(args.binding, args.budget, args.contract,
                                runner_sha=args.runner_sha, plan_sha=args.plan_sha,
                                budget_sha=args.budget_sha)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
