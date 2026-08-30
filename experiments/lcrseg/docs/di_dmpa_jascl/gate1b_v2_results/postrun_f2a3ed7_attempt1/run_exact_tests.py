"""One exact-code validation attempt; never launches a formal diagnostic."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

CODE='f2a3ed7476323119b1a4fa22481b44038bc4148c'
ROOT=Path('/root/SSL_CL_gate1b_v2/experiments/lcrseg')
OUTPUT=Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2_validation')/CODE/'attempt1'
OUTPUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT))
from di_dmpa_gate1b_v2.binding import sha256, read_json, write_json

environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    LD_LIBRARY_PATH='/lib/x86_64-linux-gnu',PYTHONPATH=str(ROOT),CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0,1',
    GATE1B_V2_CODE_COMMIT=CODE,GATE1B_V2_INTEGRATION_OUTPUT=str(OUTPUT/'integration'))
command=[sys.executable,'-m','pytest','-q','tests/di_dmpa_gate1b_v2','--junitxml='+str(OUTPUT/'pytest.xml')]
with (OUTPUT/'pytest_output.txt').open('x') as stream:
    result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
cases=list(ET.parse(OUTPUT/'pytest.xml').getroot().iter('testcase'))
failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
skipped=sum(c.find('skipped') is not None for c in cases)
integration=OUTPUT/'integration/GATE1B_V2_REAL_INTEGRATION.json'
receipt=read_json(integration) if integration.exists() else {}
if integration.exists():shutil.copy2(integration,OUTPUT/integration.name)
names=['pytest.xml','pytest_output.txt']+([integration.name] if integration.exists() else [])
sources=sorted((ROOT/'di_dmpa_gate1b_v2').glob('*.py'))+sorted((ROOT/'tests/di_dmpa_gate1b_v2').glob('*.py'))
report=dict(status='PASS' if result.returncode==0 and failures==skipped==0 and len(cases)==77 and receipt.get('status')=='PASS' else 'FAIL',
    diagnostic_code_commit=CODE,metadata=receipt.get('metadata'),tests=len(cases),passed=len(cases)-failures-skipped,failures=failures,skipped=skipped,
    pytest_exit_code=result.returncode,real_integration_status=receipt.get('status','MISSING'),
    model_optimizer_steps=0,real_transport_optimizer_steps=0,synthetic_optimizer_tests_are_not_experiment_steps=True,
    coverage='all44 preregistered categories plus synthetic end-to-end and exact-code read-only RIM/Drishti integration',
    command=command,working_directory=str(ROOT),existing_python=sys.executable,
    artifact_sha256={name:sha256(OUTPUT/name) for name in names},source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources})
write_json(OUTPUT/'GATE1B_V2_UNIT_INTEGRATION_TEST_REPORT.json',report)
print(json.dumps(report,ensure_ascii=False),flush=True)
raise SystemExit(0 if report['status']=='PASS' else 2)
