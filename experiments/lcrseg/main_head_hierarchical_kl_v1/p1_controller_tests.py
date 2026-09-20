from pathlib import Path
import json
import tempfile
from .p1_controller import GateError, validate_plan


def test_current_plan_is_denied():
    path = Path(__file__).parents[3] / "experiments/lcrseg/docs/main_head_hierarchical_kl_v1/P1_EXECUTION_PLAN.json"
    try:
        validate_plan(path)
    except GateError:
        return
    raise AssertionError("unapproved P1 plan was accepted")


def test_exact_start_nodes():
    source = json.loads(Path(__file__).parents[3].joinpath(
        "experiments/lcrseg/docs/main_head_hierarchical_kl_v1/P1_EXECUTION_PLAN.json").read_text())
    source["status"] = "P1_EXECUTION_APPROVED"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.json"
        path.write_text(json.dumps(source))
        assert validate_plan(path)["formal_updates"] == 10600
        source["nodes"][0]["state_id"] = "O1_STAGE2_ENDPOINT"
        path.write_text(json.dumps(source))
        try:
            validate_plan(path)
        except GateError:
            pass
        else:
            raise AssertionError("endpoint node was accepted")


if __name__ == "__main__":
    test_current_plan_is_denied()
    test_exact_start_nodes()
    print("p1_controller_tests: PASS")
