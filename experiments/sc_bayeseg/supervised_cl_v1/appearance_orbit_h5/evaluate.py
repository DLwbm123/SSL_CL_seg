"""Frozen H1 read-only evaluator with H5's new seed."""

import json
import os
from pathlib import Path

from ..hierarchical_bayes_h1 import evaluate as base


def evaluate(settings):
    base.SEED = 20261031
    return base.evaluate(settings)


if __name__ == "__main__":
    evaluate(json.loads(Path(os.environ["TASK_EVAL_SETTINGS"]).read_text()))
