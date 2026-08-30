"""Exact published v2 tests/formal launcher; never auto-advances tests->formal."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

CODE='8ae5d7532f90aee5d53c0d966706ef64c18a19ac'
PREREG='eaae37bbaa7546679d9e6893023afbeeef0ab5c6'
ROOT=Path('/root/SSL_CL_gate1_v2/experiments/lcrseg')
PARENT=Path('/root/LCRSeg/runs/di_dmpa_gate1_v2')/PREREG
TESTS=PARENT/'tests_8ae5d75_attempt1'
INTEGRATION=PARENT/'integration_8ae5d75_attempt1'
FORMAL=PARENT/f'gate1a_v2_{CODE}_attempt1'
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'LD_LIBRARY_PATH':'/lib/x86_64-linux-gnu','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':'.'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def command(mode):
    if mode=='tests':return [sys.executable,'-m','pytest','-q','tests/di_dmpa_gate1_v2',
        'tests/di_dmpa_gate1/test_gate1a_core.py','tests/di_dmpa_gate1/test_gate1a_recovery.py',
        '--junitxml='+str(TESTS/'pytest.xml')]
    return [sys.executable,'-m','di_dmpa_gate1_v2.runner','run','--code-commit',CODE,
        '--output',str(FORMAL),'--tests',str(TESTS),'--gpus','0,1','--workers','16']

def worker(mode):
    env=dict(ENV)
    if mode=='tests':env.update(CUDA_VISIBLE_DEVICES='0',GATE1A_CODE_COMMIT=CODE,GATE1A_V2_INTEGRATION_OUTPUT=str(INTEGRATION))
    if mode=='tests':
        with (TESTS/'pytest_output.txt').open('x') as log:
            result=subprocess.run(command(mode),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        cases=list(ET.parse(TESTS/'pytest.xml').getroot().iter('testcase'))
        failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
        skipped=sum(c.find('skipped') is not None for c in cases)
        integration_path=INTEGRATION/'GATE1A_V2_REAL_INTEGRATION.json'
        integration=json.loads(integration_path.read_text()) if integration_path.exists() else None
        if integration:
            with integration_path.open('rb') as source,(TESTS/'GATE1A_V2_REAL_INTEGRATION.json').open('xb') as target:target.write(source.read())
        write(TESTS/'GATE1A_V2_UNIT_INTEGRATION_TEST_REPORT.json',dict(
            status='PASS' if result.returncode==0 and len(cases)==98 and failures==skipped==0 and integration and integration['status']=='PASS' else 'FAIL',
            diagnostic_code_git_commit=CODE,tests=len(cases),passed=len(cases)-failures-skipped,failures=failures,skipped=skipped,
            real_integration_status=None if integration is None else integration['status'],model_optimizer_steps=0,transport_optimizer_steps=0,
            integration_sha256=digest(integration_path) if integration else None,
            pytest_xml_sha256=digest(TESTS/'pytest.xml'),pytest_output_sha256=digest(TESTS/'pytest_output.txt')))
    else:result=subprocess.run(command(mode),cwd=ROOT,env=env)
    write(PARENT/f'{mode}_8ae5d75_attempt1.completion.json',dict(exit_code=result.returncode))

if __name__=='__main__':
    mode=sys.argv[1]
    assert mode in ('tests','formal')
    if len(sys.argv)>2:worker(mode)
    else:
        PARENT.mkdir(parents=True,exist_ok=True)
        if mode=='tests':
            TESTS.mkdir(exist_ok=False);assert not INTEGRATION.exists()
        else:
            assert not FORMAL.exists()
            assert json.loads((TESTS/'GATE1A_V2_UNIT_INTEGRATION_TEST_REPORT.json').read_text())['status']=='PASS'
        with (PARENT/f'{mode}_8ae5d75_attempt1.launch.txt').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),mode,'worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(PARENT/f'{mode}_8ae5d75_attempt1.launch.json',dict(pid=child.pid,command=command(mode),cwd=str(ROOT),source=CODE,launcher_sha256=digest(__file__),
            environment={k:ENV[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH')}))
        print(json.dumps(dict(pid=child.pid,mode=mode,output=str(TESTS if mode=='tests' else FORMAL))))
