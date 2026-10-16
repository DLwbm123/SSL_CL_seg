"""Environment-based neutral entry; authority checked before output creation."""
import json,os
from pathlib import Path


def main():
    config=json.loads(Path(os.environ['TASK_CONFIG']).read_text());action=os.environ['TASK_ACTION']
    if action=='worker':
        from .runtime import worker
        return worker(config)
    if action=='controller':
        from .controller import controller
        return controller(config)
    raise PermissionError('unknown action')


if __name__=='__main__':main()
