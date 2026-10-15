"""Only explicit future authority can reach a worker; no import-time dispatch."""
import json
import os
from pathlib import Path
from .runtime import worker


def main():
    if os.environ.get('TASK_ACTION') != 'worker':
        raise PermissionError('V2 entry supports authorized targets only; no qualification/training auto-launch')
    config=json.loads(Path(os.environ['TASK_CONFIG']).read_text())
    return worker(config)


if __name__ == '__main__':
    main()
