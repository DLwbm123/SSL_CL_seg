import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = "943ab307cc5f1fded0eeb46392a71abe232523c3"
FROZEN = (
    "experiments/lcrseg/docs/shor_jascl_v0_3",
    "experiments/lcrseg/docs/shor_jascl_v0_3_1",
    "experiments/lcrseg/docs/shor_v0_4_fixed_policy_test",
    "experiments/lcrseg/docs/rc_shor_v0_5",
    "experiments/lcrseg/docs/rc_shor_v0_5_erratum",
    "experiments/lcrseg/docs/ppc_shor_v0_6a",
    "experiments/lcrseg/docs/ppc_shor_v0_6b",
    "experiments/lcrseg/rc_shor_v0_5.py",
    "experiments/lcrseg/ppc_shor_v0_6a.py",
    "experiments/lcrseg/ppc_shor_v0_6b.py",
    "experiments/lcrseg/tests/rc_shor_v0_5",
    "experiments/lcrseg/tests/ppc_shor_v0_6a",
    "experiments/lcrseg/tests/ppc_shor_v0_6b",
)


def test_all_frozen_predecessors_are_unchanged_from_base():
    for path in FROZEN:
        assert subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", BASE, "--", path]).returncode == 0
