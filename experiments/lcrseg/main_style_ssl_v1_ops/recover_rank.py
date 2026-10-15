"""One authorized zero-update Prostate164 recovery; adopt other live workers."""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from experiments.lcrseg.main_style_ssl_v1.protocol import nodes
from experiments.lcrseg.main_style_ssl_v1_ops.resource import process, alive, write


def recover(config):
    root=Path(config['root']);repair=Path(config['repair']);code=Path(config['code'])
    qualified=json.loads((repair/'engineering/RESULT.json').read_text())
    assert qualified['status']=='PASS' and qualified['source_optimizer_updates']==qualified['formal_optimizer_updates']==0
    plan=json.loads((root/'PLAN.json').read_text())
    assert plan['nodes']==nodes() and plan['total_optimizer_updates']==511600
    keys=[n['id'] for n in nodes() if n['dataset']=='prostate' and n['seed']==164 and n['stage']>0]
    assert len(keys)==5 and sum(n['steps'] for n in nodes() if n['id'] in keys)==75100
    failed=root/'jobs'/keys[0]
    assert json.loads((failed/'STATUS.json').read_text())['status']=='FAILED'
    assert json.loads((failed/'EXIT.json').read_text())['exit_code']==1
    assert json.loads((failed/'COSTS.json').read_text())['physical_optimizer_updates']==0
    assert not (failed/'physical_updates.jsonl').exists() or not (failed/'physical_updates.jsonl').read_text().strip()
    source=root/'jobs/prostate_s164_source'
    assert json.loads((source/'STATUS.json').read_text())['step']==13600
    assert json.loads((source/'EXIT.json').read_text())['exit_code']==0
    assert sum(1 for _ in (source/'physical_updates.jsonl').open())==13600
    source_stat=(source/'sealed.pt').stat()
    prior=repair/'prior_attempt';prior.mkdir();(prior/'jobs').mkdir()
    receipt=json.loads((root/'CONTROLLER_PID.json').read_text())
    old=dict(pid=receipt['pid'],birth=process(receipt['pid'])['birth'])
    assert alive(old) and b'./c.py' in Path(f"/proc/{old['pid']}/cmdline").read_bytes()
    copied=[];dispatched=False;started=time.time()
    os.kill(old['pid'],signal.SIGSTOP)
    try:
        for _ in range(100):
            if process(old['pid'])['state']=='T':break
            time.sleep(.01)
        assert process(old['pid'])['state']=='T'
        status_path=root/'STATUS.json';status=json.loads(status_path.read_text())
        assert status['completed']['prostate_s164_source']=='COMPLETE'
        for key in keys:
            expected='FAILED' if key==keys[0] else 'BLOCKED'
            assert status['completed'][key]==expected and key not in status['running']
            assert json.loads((root/'jobs'/key/'STATUS.json').read_text())['status']==expected
        for entry in status['running'].values():
            info=process(entry['pid'])
            assert info is None or info['birth']==entry['birth']
        for name in ('STATUS.json','CONTROLLER_PID.json'):
            shutil.copy2(root/name,prior/name)
        shutil.copy2(root/'configs'/(keys[0]+'.json'),prior/'failed_worker_config.json')
        old_config=json.loads((root/'resource_change_20261014T021117Z/CONFIG.json').read_text())
        shutil.copy2(Path(old_config['change'])/(keys[0]+'.new.log'),prior/'failed_worker.log')
        for key in keys:
            os.rename(root/'jobs'/key,prior/'jobs'/key);copied.append(key)
        change=repair/'scheduler';change.mkdir()
        cfg={**old_config,'change':str(change),'script':str(code/'experiments/lcrseg/main_style_ssl_v1_ops/resource.py'),
             'inherited':status['running'],'states':{k:v for k,v in status['completed'].items() if k not in keys},
             'old_controller':old,'previous_elapsed':status['elapsed_seconds']+time.time()-status_path.stat().st_mtime,
             'worker_code_roots':{key:str(code) for key in keys},
             'worker_code_commits':{key:config['code_commit'] for key in keys},
             'attempt_history':str(root/'ATTEMPT_HISTORY.json')}
        assert cfg['gpus']==[4,5,6,7] and cfg['capacity']==2
        write(change/'CONFIG.json',cfg)
        history_path=root/'ATTEMPT_HISTORY.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else {'attempts':[]}
        history['attempts'].append(dict(kind='AUTHORIZED_ENGINEERING_RANK_RECOVERY',prior_attempt=str(prior),
             failed=keys[0],blocked=keys[1:],original_physical_updates=0,source_updates=13600,
             source_rerun_updates=0,remaining_branch_budget=75100,code_commit=config['code_commit'],
             engineering_result=str(repair/'engineering/RESULT.json'),engineering_updates=100))
        write(history_path,history)
        env=dict(os.environ,PYTHONPATH=str(root/'code_v4'),TASK_SCRIPT=cfg['script'],RESOURCE_ACTION='run',
                 RESOURCE_CONFIG=str(change/'CONFIG.json'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (change/'controller.log').open('x') as log:
            child=subprocess.Popen(['bash','./s',cfg['python'],'./c.py'],cwd=cfg['ops'],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dispatched=True
        new=dict(pid=child.pid,birth=process(child.pid)['birth'])
        write(root/'CONTROLLER_PID.json',dict(**new,code_commit=receipt['code_commit'],scheduling_commit=config['code_commit'],
             log=str(change/'controller.log'),previous_pid=old['pid'],worker_code_commits=cfg['worker_code_commits'],
             attempt_history=str(history_path)))
        now=(source/'sealed.pt').stat()
        assert (source_stat.st_ino,source_stat.st_size,source_stat.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns)
        write(repair/'HANDOFF.json',dict(old_controller=old,new_controller=new,adopted_workers=status['running'],
             recovered_nodes=keys,formal_budget=511600,source_rerun_updates=0,repeated_formal_updates=0,
             source_checkpoint=str(source/'sealed.pt'),source_checkpoint_size=source_stat.st_size,
             code_commit=config['code_commit'],seconds=time.time()-started))
        print(json.dumps(dict(controller=new,recovery=keys,source_rerun_updates=0)),flush=True)
    finally:
        if not dispatched:
            for key in reversed(copied):os.rename(prior/'jobs'/key,root/'jobs'/key)
            if alive(old):os.kill(old['pid'],signal.SIGCONT)


if __name__=='__main__':
    recover(json.loads(Path(os.environ['RESOURCE_CONFIG']).read_text()))
