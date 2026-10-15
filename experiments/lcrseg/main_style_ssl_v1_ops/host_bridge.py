"""One authorized final handoff, run on the SSH client; never retry training."""
import json
import subprocess
import time
import traceback

OLD = '/mnt/source-nas/account/LCR-Seg/SSL_CL_seg/runs/main_style_ssl_v1_20261014T013808Z'
NEW = '/srv/intermediate/account/SSL_CL_seg/runs/main_style_ssl_v1_20261014T013808Z'
CHANGE = 'host_migration_20261014T044549Z'
KEY = 'mnms_s164_source'


def remote(host, code):
    value = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'python3 -'],
                           input=code, text=True, capture_output=True, timeout=90)
    if value.returncode:
        raise RuntimeError(host+': '+value.stderr[-1500:]+value.stdout[-1500:])
    return json.loads(value.stdout)


def transfer(names):
    source = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'source-gpu',
                               'tar -cf - -C '+OLD+' '+' '.join(names)], stdout=subprocess.PIPE)
    target = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'bridge-gpu', 'tar -xf - -C '+NEW], stdin=source.stdout)
    source.stdout.close()
    target_code, source_code = target.wait(), source.wait()
    if (source_code, target_code) != (0, 0):
        raise RuntimeError('handoff transfer failed: '+str((source_code, target_code)))


def status(value):
    remote('bridge-gpu', "from pathlib import Path\nimport json\np=Path("+repr(NEW+'/'+CHANGE+'/BRIDGE_STATUS.json')+")\np.write_text("+repr(json.dumps(value, indent=2))+")\nprint('{}')\n")


def run():
    status(dict(status='WAITING_FINAL_CHECKPOINT',node=KEY,training_retries=0))
    while True:
        ready = remote('source-gpu', "from pathlib import Path\nimport json\np=Path("+repr(OLD+'/'+CHANGE+'/'+KEY+'.ready.json')+")\nprint(p.read_text() if p.exists() else '{}')\n")
        if ready:
            break
        time.sleep(30)
    assert ready['step'] >= 34000 and ready['step'] <= 42000 and ready['step'] % 2000 == 0
    assert ready['physical_updates'] == ready['step']
    status(dict(status='TRANSFERRING_FROZEN_PREFIX',node=KEY,step=ready['step'],training_retries=0))
    transfer(['jobs/'+KEY, 'configs/'+KEY+'.json', CHANGE])
    admission = remote('bridge-gpu', f'''from pathlib import Path
import json,os,subprocess
r=Path({NEW!r});c=r/{CHANGE!r};key={KEY!r}
h=json.loads((c/(key+'.ready.json')).read_text())
rows=[json.loads(x) for x in (r/'jobs'/key/'physical_updates.jsonl').read_text().splitlines()]
assert len(rows)==h['step'] and all(v['scientific_step']==i+1 for i,v in enumerate(rows))
worker=json.loads((r/'configs'/(key+'.json')).read_text())
worker.update(output=str(r/'jobs'/key),data='/srv/intermediate/account/SSL_CL_seg/data',reference='/srv/intermediate/account/SSL_CL_seg/reference/JASCL_REFERENCE',predecessor=None)
checkpoint=str(c/key/'latest.pt')
cfg=dict(worker=worker,checkpoint=checkpoint,compatibility=True,result=str(c/(key+'.preflight.json')))
p=c/(key+'.preflight.config.json');p.write_text(json.dumps(cfg))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',PYTHONPATH=str(r/'code_v4'),HOST_SCRIPT='/srv/intermediate/account/SSL_CL_seg/ops/host_preflight.py',HOST_CONFIG=str(p),CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (c/(key+'.preflight.log')).open('x') as log:
 result=subprocess.run(['bash','./s','/srv/intermediate/account/envs/x20261014/bin/python','./n.py'],cwd='/srv/intermediate/account/.ops/x20261014',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
assert result.returncode==0
check=json.loads(Path(cfg['result']).read_text());assert check['status']=='PASS' and check['step']==h['step']
h.update(checkpoint=checkpoint,origin_host='source-gpu',preflight_status='PASS')
temp=c/(key+'.ADMITTED.tmp');temp.write_text(json.dumps(h,indent=2));os.replace(temp,c/(key+'.ADMITTED.json'))
print(json.dumps(dict(step=h['step'],preflight='PASS')))
''')
    status(dict(status='AWAITING_TARGET_PROGRESS',node=KEY,step=admission['step'],training_retries=0))
    for _ in range(24):
        target = remote('bridge-gpu', f'''from pathlib import Path
import json
r=Path({NEW!r});key={KEY!r};s=json.loads((r/'STATUS.json').read_text());entry=s['running'].get(key)
v=json.loads((r/'jobs'/key/'STATUS.json').read_text())
if v.get('status')=='FAILED':raise RuntimeError(v.get('error'))
result={{}}
if entry and v.get('status')=='RUNNING' and v.get('step',0)>{admission['step']}:
 f=Path('/proc/%s/stat'%entry['pid']).read_text().split(') ',1)[1].split();assert int(f[19])==entry['birth'] and f[0]!='Z'
 assert v['last']['active_U'] and v['physical_updates']==v['step']
 result=dict(entry=entry,step=v['step'],physical_updates=sum(1 for _ in (r/'jobs'/key/'physical_updates.jsonl').open()))
print(json.dumps(result))
''')
        if target:
            break
        time.sleep(10)
    else:
        raise RuntimeError('target did not confirm progress; origin remains suspended')
    retired = remote('source-gpu', f'''from pathlib import Path
import json,os,signal,time
r=Path({OLD!r});c=r/{CHANGE!r};key={KEY!r};h=json.loads((c/(key+'.ready.json')).read_text());cfg=json.loads((c/'WATCH_CONFIG.json').read_text())
assert sum(1 for _ in (r/'jobs'/key/'physical_updates.jsonl').open())==h['step']
for entry in (h['old'],cfg['origin_controller']):
 f=Path('/proc/%s/stat'%entry['pid']).read_text().split(') ',1)[1].split();assert int(f[19])==entry['birth'] and f[0]=='T'
 os.kill(entry['pid'],signal.SIGTERM);os.kill(entry['pid'],signal.SIGCONT)
receipt=dict(status='ORIGIN_RETIRED_AFTER_TARGET_PROGRESS',old=h['old'],target={target!r},origin_checkpoint_step=h['step'],repeated_optimizer_updates=0,time=time.time())
(c/(key+'.migrated.json')).write_text(json.dumps(receipt,indent=2))
(r/'STATUS.json').write_text(json.dumps(dict(status='ARCHIVED_MIGRATED',active_host='bridge-gpu',active_root={NEW!r},running={{}},history=str(c/'ORIGIN_STATUS_FROZEN.json')),indent=2))
(r/'HOST_MIGRATION_PENDING.json').write_text(json.dumps(dict(status='COMPLETE',change=str(c),target_host='bridge-gpu',target_root={NEW!r},training_retries=0),indent=2))
print(json.dumps(receipt))
''')
    transfer([CHANGE+'/'+KEY+'.migrated.json'])
    remote('bridge-gpu', f'''from pathlib import Path
import json,time
r=Path({NEW!r});c=r/{CHANGE!r};p=r/'HOST_MIGRATION.json';v=json.loads(p.read_text());v.update(status='COMPLETE',pending=[],completed_unix=time.time());v['handoff_steps'][{KEY!r}]={admission['step']};v['last_handoff_verified']={target!r}
v['checkpoint_holds'][{KEY!r}]=dict(stopped_unix={ready['stopped']},target_launch_unix=(c/({KEY!r}+'.config.json')).stat().st_mtime,retired_unix={retired['time']},origin_GPU_hold_seconds={retired['time']-ready['stopped']})
p.write_text(json.dumps(v,indent=2));(r/'HOST_MIGRATION_PENDING.json').write_text(json.dumps(dict(status='COMPLETE',target_host='bridge-gpu',origin_preserved=True),indent=2))
print('{{}}')
''')
    status(dict(status='COMPLETE',node=KEY,checkpoint_step=admission['step'],verified_step=target['step'],training_retries=0))
    print('Final handoff COMPLETE', admission['step'], target['step'], flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        try:
            status(dict(status='FAILED_HANDOFF_NO_RETRY',error=str(exc),training_retries=0))
        except Exception:
            pass
        traceback.print_exc()
        raise
