"""One bounded continuation after the identified sealed-step metadata assertion.

Never retries training or evaluation and never handles other failures.
"""
import json,math,os,shutil,subprocess,time
from pathlib import Path


def main():
    root=Path(os.environ['RUN_ROOT']);ops=Path(os.environ['NEUTRAL_OP_ROOT']);launch=json.loads((root/'LAUNCH.json').read_text());deadline=launch['deadline']
    while time.time()<deadline-30:
        if (root/'EXIT.json').exists():
            process=Path('/proc')/str(launch['pid'])
            if process.exists():
                args=(process/'cmdline').read_bytes().decode().replace('\0',' ')
                if args:
                    assert str(ops) in args
                    time.sleep(5);continue
            break
        time.sleep(5)
    else:raise RuntimeError('original deadline reached; no continuation')
    exit_record=json.loads((root/'EXIT.json').read_text())
    if exit_record['exit_code']==0:return
    status=json.loads((root/'STATUS.json').read_text());ledger=json.loads((root/'results/RESOURCE_LEDGER.json').read_text())
    assert status['stage']=='FAILED' and status['error_type']=='AssertionError'
    assert 'source_digest' in (root/'run.log').read_text()[-3000:]
    assert ledger['updates_reserved']==1600 and set(ledger['training_sessions'])=={'W_source'}
    assert ledger['training_sessions']['W_source']['status']=='COMPLETE'
    assert ledger['training_sessions']['W_source']['completed_steps']==1600
    assert {v['name'] for v in ledger['attempts']}=={'preflight','W_source_attempt001','eval_W_source_RUNMC','eval_W_source_BMC'}
    assert all(v['status']=='PASS' for v in ledger['attempts'])
    for domain in ('RUNMC','BMC'):
        assert json.loads((root/'results'/f'eval_W_source_{domain}/validation/SUMMARY.json').read_text())['complete']
    failure=root/'failure_entry001';failure.mkdir(exist_ok=False)
    for name in ('EXIT.json','STATUS.json','run.log'):(root/name).replace(failure/name)
    shutil.copy2(root/'source/sc_bayeseg/hbsi_v1/multiscale_recognition_v46/run.py',failure/'run.py')
    record=dict(reason='inactive_load_resets_steps_only',first_exit=exit_record,original_start=launch['started'],original_deadline=deadline,retained_formal_updates=1600,extra_formal_updates=0,source_retrained=False,source_rescored=False,preflight_repeated=False,corrected_module='resume',scientific_protocol_changed=False)
    (root/'REPAIR_RECORD.json').write_text(json.dumps(record,indent=2))
    (root/'STATUS.json').write_text(json.dumps(dict(stage='CONTINUING_SEALED_SOURCE',time=time.time())))
    entry=(ops/'run.py').read_text().replace('multiscale_recognition_v46.run','multiscale_recognition_v46.resume')
    (ops/'r.py').write_text(entry)
    env=dict(os.environ,RESUME_SEALED_SOURCE='1');remaining=math.floor(deadline-time.time())
    assert remaining>30
    with (root/'run.log').open('x') as log:
        proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=15',str(remaining),'bash',str(ops/'a.sh'),str(ops/'python'),str(ops/'r.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    args=subprocess.check_output(['ps','-p',str(proc.pid),'-o','args='],text=True).strip()
    assert str(ops) in args and args==' '.join(proc.args)
    (root/'REPAIR_LAUNCH_002.json').write_text(json.dumps(dict(pid=proc.pid,started=time.time(),deadline=deadline,remaining_timeout_seconds=remaining,neutral_parent_argv_verified=True),indent=2))

if __name__=='__main__':main()
