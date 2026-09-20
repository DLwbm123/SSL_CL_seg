from pathlib import Path
import json
from .p1_controller import GateError, validate_plan


def test_current_plan_is_denied():
    path = Path(__file__).parents[3] / "experiments/lcrseg/docs/main_head_hierarchical_kl_v1/P1_EXECUTION_PLAN.json"
    try:
        validate_plan(path)
    except GateError:
        return
    raise AssertionError("unapproved P1 plan was accepted")


if __name__ == "__main__":
    test_current_plan_is_denied()
    print("p1_controller_tests: PASS")
