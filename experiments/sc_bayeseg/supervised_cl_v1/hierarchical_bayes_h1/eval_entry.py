import json
import os
from pathlib import Path

from h1.evaluate import evaluate


evaluate(json.loads(Path(os.environ["TASK_EVAL_SETTINGS"]).read_text()))
