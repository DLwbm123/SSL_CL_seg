"""Plan-only P1 gate. It never imports torch or launches a process."""
import json
from pathlib import Path


class GateError(RuntimeError):
    pass


def validate_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("status") != "P1_EXECUTION_APPROVED":
        raise GateError("P1 plan is not execution-approved")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise GateError("P1 requires exactly four nodes")
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


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    args = parser.parse_args(argv)
    print(json.dumps(validate_plan(args.plan), sort_keys=True))


if __name__ == "__main__":
    main()
