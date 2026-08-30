"""One immutable exact-source test/localization invocation; no formal attempt2."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

CODE='a89716ddbd2eccbe76c574e97e520d424aa923ab'
ROOT=Path('/root/SSL_CL_gate1_recovery_code/experiments/lcrseg')
PARENT=Path('/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20')
OUTPUT=PARENT/'gate1a_recovery_tests_a89716d_attempt1'
LOCALIZATION=PARENT/'gate1a_known_failure_a89716d_attempt1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def worker():
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'LD_LIBRARY_PATH':'/lib/x86_64-linux-gnu','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':'.',
        'CUDA_VISIBLE_DEVICES':'0','GATE1A_CODE_COMMIT':CODE,'GATE1A_RECOVERY_LOCALIZATION_OUTPUT':str(LOCALIZATION)}
    command=[sys.executable,'-m','pytest','-q','tests/di_dmpa_gate1/test_gate1a_core.py',
        'tests/di_dmpa_gate1/test_gate1a_recovery.py','tests/di_dmpa_gate1/test_gate1a_recovery_integration.py',
        '--junitxml='+str(OUTPUT/'GATE1A_RECOVERY_PYTEST.xml')]
    write(OUTPUT/'EXACT_TEST_COMMAND.json',dict(command=command,cwd=str(ROOT),environment={k:env[k] for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG',
         'PYTHONPATH','CUDA_VISIBLE_DEVICES','GATE1A_CODE_COMMIT','GATE1A_RECOVERY_LOCALIZATION_OUTPUT')}))
    with (OUTPUT/'GATE1A_RECOVERY_PYTEST_OUTPUT.txt').open('x') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    suites=ET.parse(OUTPUT/'GATE1A_RECOVERY_PYTEST.xml').getroot()
    cases=list(suites.iter('testcase'))
    failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped=sum(c.find('skipped') is not None for c in cases)
    locpath=LOCALIZATION/'GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json'
    localization=json.loads(locpath.read_text()) if locpath.exists() else None
    report=dict(status='PASS' if result.returncode==0 and len(cases)==65 and failures==skipped==0 and localization else 'FAIL',
        diagnostic_code_git_commit=CODE,recovery_diagnostic_code_git_commit=CODE,tests=len(cases),
        passed=len(cases)-failures-skipped,failures=failures,skipped=skipped,pytest_exit_code=result.returncode,
        localization_status=None if localization is None else localization['localization_status'],
        localization_path=str(locpath),localization_sha256=digest(locpath) if locpath.exists() else None,
        test_pass_is_not_localization_or_admission_pass=True,formal_attempt2_launched=False,
        model_optimizer_steps=0,transport_optimizer_steps=0,clustering_jobs=0,
        test_files_sha256={p.name:digest(p) for p in (ROOT/'tests/di_dmpa_gate1').glob('test_gate1a*.py')},
        evidence_sha256={n:digest(OUTPUT/n) for n in ('GATE1A_RECOVERY_PYTEST.xml','GATE1A_RECOVERY_PYTEST_OUTPUT.txt','EXACT_TEST_COMMAND.json')})
    write(OUTPUT/'GATE1A_RECOVERY_UNIT_INTEGRATION_TEST_REPORT.json',report)
    write(OUTPUT/'completion.json',dict(exit_code=result.returncode,test_status=report['status'],localization_status=report['localization_status']))

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:
        OUTPUT.mkdir(parents=True,exist_ok=False)
        assert not LOCALIZATION.exists()
        with (PARENT/'gate1a_recovery_tests_a89716d_attempt1.launch.txt').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUTPUT/'launch.json',dict(pid=child.pid,source_commit=CODE,launcher_sha256=digest(__file__)))
        print(json.dumps(dict(pid=child.pid,output=str(OUTPUT),localization=str(LOCALIZATION))))
