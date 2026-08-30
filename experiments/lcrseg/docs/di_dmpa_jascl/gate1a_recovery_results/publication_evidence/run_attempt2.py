"""Launch only the authorized exact-code Gate1A attempt2, once."""
import json
import os
from pathlib import Path
import subprocess
import sys

CODE='a89716ddbd2eccbe76c574e97e520d424aa923ab'
ROOT=Path('/root/SSL_CL_gate1_recovery_code/experiments/lcrseg')
PARENT=Path('/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20')
OUTPUT=PARENT/f'gate1a_formal_{CODE}_attempt2'
PREFIX=PARENT/(OUTPUT.name+'.launch')
COMMAND=[sys.executable,'-m','di_dmpa_gate1.gate1a_runner','run','--code-commit',CODE,'--output',str(OUTPUT),
    '--tests',str(PARENT/'gate1a_recovery_tests_a89716d_attempt1'),
    '--localization',str(PARENT/'gate1a_known_failure_a89716d_attempt1/GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json'),
    '--gpus','0,1','--workers','16']
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'LD_LIBRARY_PATH':'/lib/x86_64-linux-gnu','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':'.'}

if len(sys.argv)>1:
    result=subprocess.run(COMMAND,cwd=ROOT,env=ENV)
    with Path(str(PREFIX)+'.completion.json').open('x') as handle:
        json.dump(dict(exit_code=result.returncode),handle)
else:
    assert not OUTPUT.exists()
    audit=json.loads(Path(COMMAND[COMMAND.index('--localization')+1]).read_text())
    assert audit['attempt2_authorized'] and audit['localization_status']=='PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED'
    with Path(str(PREFIX)+'.txt').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with Path(str(PREFIX)+'.json').open('x') as handle:
        json.dump(dict(pid=child.pid,command=COMMAND,cwd=str(ROOT),environment={k:ENV[k] for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH')}),handle,indent=2)
    print(json.dumps(dict(pid=child.pid,output=str(OUTPUT))))
