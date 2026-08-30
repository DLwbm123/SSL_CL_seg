"""Run the sole authorized attempt and retain its exit receipt outside artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CODE='f2a3ed7476323119b1a4fa22481b44038bc4148c'
ROOT=Path('/root/SSL_CL_gate1b_v2/experiments/lcrseg')
OUTPUT=Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2/b20f186deff287843f3c9f18bf4ab5633908f441')/f'gate1b_v2_{CODE}_attempt1'
TESTS=Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2_validation')/CODE/'attempt1'
sys.path.insert(0,str(ROOT))
from di_dmpa_gate1b_v2.binding import require, read_json, write_json, check_hash
test=read_json(TESTS/'GATE1B_V2_UNIT_INTEGRATION_TEST_REPORT.json')
require(test['status']=='PASS' and test['passed']==77 and test['diagnostic_code_commit']==CODE,'exact-code tests must pass')
for rel,digest in test['source_sha256'].items():check_hash(ROOT/rel,digest)
require(not OUTPUT.exists(),'no overwrite / retry')
environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    LD_LIBRARY_PATH='/lib/x86_64-linux-gnu',PYTHONPATH=str(ROOT),CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0,1')
command=[sys.executable,'-m','di_dmpa_gate1b_v2.runner','run','--code-commit',CODE,'--output',str(OUTPUT),'--tests',str(TESTS)]
started=datetime.now(timezone.utc).isoformat()
write_json('/root/gate1b_v2_formal_f2a3ed7_launch.json',dict(started_at_UTC=started,diagnostic_code_commit=CODE,command=command,working_directory=str(ROOT),output=str(OUTPUT)))
with Path('/root/gate1b_v2_formal_f2a3ed7_stdout.txt').open('x') as stream:
    result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
receipt=dict(diagnostic_code_commit=CODE,exit_code=result.returncode,started_at_UTC=started,finished_at_UTC=datetime.now(timezone.utc).isoformat(),output=str(OUTPUT))
write_json('/root/gate1b_v2_formal_f2a3ed7_exit.json',receipt)
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
