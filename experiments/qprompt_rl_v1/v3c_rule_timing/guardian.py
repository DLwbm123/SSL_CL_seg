"""Neutral argv, one worker, bounded lifetime, no automatic retry."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def main():
    root=Path(os.environ['EXEC_RUN'])
    config=json.loads((root/'CONFIG.private.json').read_text())
    session=json.loads((root/'SESSION.json').read_text())
    command=['bash','./with_nas_storage.sh',sys.executable,'-c',
             'import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")']
    env=dict(os.environ, EXEC_MODULE='experiments.qprompt_rl_v1.v3c_rule_timing.runner',
             CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
    with (root/'worker.log').open('a') as log:
        p=subprocess.Popen(command,cwd=Path(config['code'])/'experiments/lcrseg/scripts',
                           env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'LAUNCH.json',dict(run_id=root.name,pid=p.pid,guardian_pid=os.getpid(),
            gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),time=time.time(),command=command,
            commit=config['commit'],process_start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]))
        while p.poll() is None and time.time()<session['hard_deadline']:
            time.sleep(5)
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            write(root/'HARD_DEADLINE.json',dict(time=time.time(),pid=p.pid,reason='fixed hard deadline'))
        write(root/'PROCESS_EXIT.json',dict(time=time.time(),exit_code=p.returncode,pid=p.pid))


if __name__=='__main__':
    main()
