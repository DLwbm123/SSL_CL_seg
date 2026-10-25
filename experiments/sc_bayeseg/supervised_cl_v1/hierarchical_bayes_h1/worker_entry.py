import json
import os
from pathlib import Path

from h1.run import run


run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
