"""Execute synthetic witnesses against exact historical function ASTs only.

No executor imports, data-root arguments, model loads or runtime file reads.
The returned BLOCKED status is a protocol preflight result, not a research FAIL.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLOCKED = "BLOCKED_EVALUATOR_SEMANTICS_MISMATCH"


def frozen_functions(relative_path, names):
    """Extract literal pure scoring bodies; do not execute module-level imports."""
    source = (ROOT / relative_path).read_bytes()
    tree = ast.parse(source, filename=relative_path)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError("missing historical scoring definition")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace, {"path": relative_path, "file_sha256": hashlib.sha256(source).hexdigest(),
                       "functions": {n.name: {"line": n.lineno, "end_line": n.end_lineno} for n in nodes}}


def one_hot(hard):
    return np.eye(3, dtype=np.float32)[np.asarray(hard)].transpose(2, 0, 1)


def audit():
    v7, source7 = frozen_functions("care_hr_v0_7/targets.py", ("_dice", "proposal_targets"))
    v6, source6 = frozen_functions("shor_v0_4_test.py", ("case_metrics",))
    witnesses = []
    cases = (
        ("required_macro_counterexample", [[1, 1, 2, 2]], [[1, 2, 2, 2]], [[1, 2, 2, 2]]),
        ("ignore_pixel", [[1, 0]], [[1, 1]], [[1, 255]]),
        ("all_pixels_ignored", [[1, 1]], [[0, 0]], [[255, 255]]),
        ("empty_cup_noop", [[1, 1]], [[1, 1]], [[1, 1]]),
    )
    for name, current, revised, truth in cases:
        current, revised, truth = (np.asarray(x, dtype=np.int64) for x in (current, revised, truth))
        before = v6["case_metrics"](current, truth)
        after = v6["case_metrics"](revised, truth)
        witnesses.append({"name": name, "synthetic_only": True,
                          "current": current.tolist(), "revised": revised.tolist(), "truth": truth.tolist(),
                          "v0_7_targets": v7["proposal_targets"](one_hot(current), one_hot(revised), truth),
                          "v0_6b_before": before, "v0_6b_after": after,
                          "v0_6b_gain_macro_fg": after["foreground_dice"] - before["foreground_dice"],
                          "v0_6b_gain_rim": after["rim_dice"] - before["rim_dice"]})
    return {"status": BLOCKED, "sources": [source7, source6], "witnesses": witnesses,
            "call_path": "ppc_shor_v0_6b -> ppc_shor_v0_6a.v4.case_metrics",
            "differences": ["union foreground versus macro foreground target",
                            "GT=255 ignored by frozen V0.6B scorer but not V0.7 target helper",
                            "all-invalid image can produce spurious gain in V0.7 target helper"],
            "matching_rule": "both return Dice=1 for a class with zero prediction and truth support",
            "new_scoring_rule_selected": False, "real_GT_reads": 0, "model_forwards": 0}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(2)
