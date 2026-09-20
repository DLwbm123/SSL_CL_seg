"""Plan-only P1 gate. It never imports torch or launches a process."""
import json
from pathlib import Path


class GateError(RuntimeError):
    pass


class P1ExecutionError(RuntimeError):
    pass


def validate_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("status") != "P1_EXECUTION_APPROVED":
        raise GateError("P1 plan is not execution-approved")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise GateError("P1 requires exactly four nodes")
    expected = {
        ("C1", 163, 1, "O1_STAGE2_START"): 2100,
        ("C1", 163, 2, "O2_STAGE2_START"): 3200,
        ("C2", 163, 1, "O1_STAGE2_START"): 2100,
        ("C2", 163, 2, "O2_STAGE2_START"): 3200,
    }
    actual = {}
    for node in nodes:
        key = (node.get("arm"), node.get("seed"), node.get("order"), node.get("state_id"))
        if key in actual or not isinstance(node.get("formal_updates"), int) or node["formal_updates"] <= 0:
            raise GateError("P1 node identity or budget is invalid")
        actual[key] = node["formal_updates"]
    if actual != expected:
        raise GateError("P1 nodes must be the four reviewed START identities")
    if sum(n.get("formal_updates", 0) for n in nodes) != plan.get("formal_updates"):
        raise GateError("P1 node budgets do not sum to formal budget")
    if plan.get("formal_updates") != 10600 or plan.get("l_only_smoke_calls") != 16:
        raise GateError("P1 finite budget is not the reviewed budget")
    if plan.get("generated_cuda_optimizer_calls") != 15:
        raise GateError("P1 CUDA call budget is not the reviewed budget")
    if plan.get("new_seed") or plan.get("new_source_or_stage1"):
        raise GateError("P1 scope unexpectedly expands source or seed")
    if plan.get("monitoring"):
        raise GateError("monitoring is forbidden during preparation")
    return {"status": "P1_EXECUTION_APPROVED", "formal_updates": 10600}


def run_p1(path, *, p0_result, backend):
    """Run only after a real P0 result and qualified native backend are supplied."""
    plan = json.loads(Path(path).read_text())
    validate_plan(path)
    if p0_result.get("status") != "P0_RUNTIME_COMPLETE" or p0_result.get("synthetic"):
        raise P1ExecutionError("P1 requires a successful non-synthetic P0 result")
    if backend.qualify() is not True or backend.smoke() is not True:
        raise P1ExecutionError("P1 qualification/smoke gate failed")
    receipts = []
    for node in plan["nodes"]:
        receipts.append(backend.formal_node(node))
    if sum(int(r.get("formal_updates", -1)) for r in receipts) != 10600:
        raise P1ExecutionError("P1 backend accounting mismatch")
    result = {"status": "P1_RUNTIME_COMPLETE", "formal_updates": 10600,
              "smoke_calls": 16, "cuda_calls": 15, "receipts": receipts}
    backend.report(result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    args = parser.parse_args(argv)
    print(json.dumps(validate_plan(args.plan), sort_keys=True))


if __name__ == "__main__":
    main()
