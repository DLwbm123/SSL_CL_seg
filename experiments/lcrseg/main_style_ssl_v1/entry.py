import os,json
from pathlib import Path
from .runtime import worker
from .checks import qualification
from .controller import controller
config=json.loads(Path(os.environ['TASK_CONFIG']).read_text())
{'worker':worker,'qualification':qualification,'controller':controller}[os.environ['TASK_ACTION']](config)
