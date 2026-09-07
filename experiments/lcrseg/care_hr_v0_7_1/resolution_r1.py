"""Independent golden parity for the user-adopted frozen V0.6B scorer."""
from itertools import product
import json

import numpy as np

from .scoring_r1 import case_metrics, gains
from .semantic_audit import frozen_functions


REFERENCE_SHA = "7b62332879985d831efdb90fa50a3f303a6e86bd1d97a04c59b451ca1a6fd997"


def resolution_audit():
    reference, identity = frozen_functions("shor_v0_4_test.py", ("case_metrics",))
    if identity["file_sha256"] != REFERENCE_SHA:
        raise RuntimeError("frozen golden evaluator source changed")
    checked = 0
    for c, r, t in product(product(range(3), repeat=2), product(range(3), repeat=2), product((0, 1, 2, 255), repeat=2)):
        c, r, t = (np.asarray([x], dtype=np.int64) for x in (c, r, t))
        old_c, old_r = reference["case_metrics"](c, t), reference["case_metrics"](r, t)
        new_c, new_r = case_metrics(c, t), case_metrics(r, t)
        for new, old in ((new_c, old_c), (new_r, old_r)):
            if any(new[k] != old[k] for k in old):
                raise AssertionError("exact score parity failure")
        delta = gains(new_c, new_r)
        expected = [old_r[key] - old_c[key] for key in ("foreground_dice", "rim_dice", "cup_dice")]
        if [delta[k] for k in ("gain_macro_fg", "gain_rim", "gain_cup")] != expected or delta["harm"] != max(0.0, *(-v for v in expected)):
            raise AssertionError("exact gain/harm parity failure")
        checked += 1
    return {"status": "PASS_SCORING_RESOLUTION_AND_EXACT_PARITY", "historical_difference_detected": True,
            "resolution": "ADOPT_FROZEN_V0_6B_SCORER", "unresolved_semantics_count": 0,
            "two_pixel_combinations": checked, "pytest_case_count_claim": False,
            "score_tolerance": 0, "reference": identity, "real_GT_reads": 0}


if __name__ == "__main__":
    print(json.dumps(resolution_audit(), indent=2, sort_keys=True))
