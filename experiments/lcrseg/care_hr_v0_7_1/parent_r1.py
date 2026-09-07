"""Durable parent receipt records actual child return code, independent of SSH."""
import argparse
from pathlib import Path
import subprocess
import time
from .io_r1 import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    if a.receipt.exists():raise FileExistsError(a.receipt)
    start=time.time()
    with a.log.open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        result=child.wait()
    write_json(a.receipt,dict(command=command,child_pid=child.pid,actual_child_exit_code=result,start_unix=start,end_unix=time.time()))
    return result


if __name__=='__main__':raise SystemExit(main())
