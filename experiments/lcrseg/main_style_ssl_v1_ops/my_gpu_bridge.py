"""Continue admitted chains across hosts without replay or training retries."""
import importlib.util
import json
import subprocess
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location('prior_bridge', Path(__file__).with_name('host_bridge.py'))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
remote = prior.remote
OLD = '/srv/intermediate/account/SSL_CL_seg/runs/main_style_ssl_v1_20261014T013808Z'
NEW = '/mnt/target-nas/account/SSL_CL_seg/runs/main_style_ssl_v1_20261014T013808Z'
CHANGE = 'host_migration_my_gpu_20261014T125300Z'
KEYS = ['mnms_s163_o1_t1', 'mnms_s164_o1_t1', 'prostate_s163_o1_t5', 'prostate_s164_o1_t4']


def transfer(names):
    source = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'bridge-gpu',
                               'tar -cf - -C '+OLD+' '+' '.join(names)], stdout=subprocess.PIPE)
    target = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'target-gpu', 'tar -xf - -C '+NEW], stdin=source.stdout)
    source.stdout.close()
    codes = target.wait(), source.wait()
    if codes != (0, 0):
        raise RuntimeError('frozen prefix transfer failed: '+str(codes))


def progress(key):
    return remote('target-gpu', f'''import json,math,time
from pathlib import Path
r=Path({NEW!r});key={key!r};s=json.loads((r/'STATUS.json').read_text());e=s['running'].get(key)
p=r/'jobs'/key;v=json.loads((p/'STATUS.json').read_text())
if v.get('status')=='FAILED':raise RuntimeError(v.get('error'))
result={{}}
if e and v.get('status')=='RUNNING':
 f=Path('/proc/%s/stat'%e['pid']).read_text().split(') ',1)[1].split()
 assert int(f[19])==e['birth'] and f[0] not in ('Z','T')
 assert all(math.isfinite(v['last'][k]) for k in ('labeled_loss','unlabeled_loss') if v['last'].get(k) is not None)
 rows=[json.loads(x) for x in (p/'physical_updates.jsonl').read_text().splitlines()]
 assert all(x['scientific_step']==i+1 and x['optimizer_executed'] for i,x in enumerate(rows))
 result=dict(entry=e,reported_step=v['step'],updates=len(rows),active_U=v['last']['active_U'],time=time.time())
print(json.dumps(result))
''')


def handoff(key, ready):
    assert ready['physical_updates'] == ready['step'] and ready['step'] % 2000 == 0
    remote('bridge-gpu', f'''import json,os,subprocess
from pathlib import Path
r=Path({OLD!r});c=r/{CHANGE!r};key={key!r};h=json.loads((c/(key+'.ready.json')).read_text())
w=json.loads((r/'host_migration_20261014T044549Z'/(key+'.config.json')).read_text())
cfg=dict(worker=w,checkpoint=h['checkpoint'],entry_state=str(c/(key+'.entry.pt')),compatibility=True,result=str(c/(key+'.origin_preflight.json')),batch_trace=str(c/(key+'.batches.pt')))
p=c/(key+'.origin_preflight.config.json');p.write_text(json.dumps(cfg))
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(h['old']['gpu']),PYTHONPATH=str(r/('rank_repair_20261014T024957Z/code' if key=='prostate_s164_o1_t4' else 'code_v4')),HOST_SCRIPT=str(c/'code/host_preflight.py'),HOST_CONFIG=str(p),CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (c/(key+'.origin_preflight.log')).open('x') as log:
 result=subprocess.run(['bash','./s','/srv/intermediate/account/envs/x20261014/bin/python','./n.py'],cwd='/srv/intermediate/account/.ops/x20261014',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=80)
assert result.returncode==0,'origin native check failed; preserved with zero optimizer calls'
print(Path(cfg['result']).read_text())
''')
    transfer(['jobs/'+key, CHANGE+'/'+key, CHANGE+'/'+key+'.ready.json',
              CHANGE+'/'+key+'.entry.pt', CHANGE+'/'+key+'.entry.json', CHANGE+'/'+key+'.batches.pt',
              CHANGE+'/'+key+'.origin_preflight.json', CHANGE+'/'+key+'.origin_preflight.log'])
    admitted = remote('target-gpu', f'''import json,os,subprocess
from pathlib import Path
r=Path({NEW!r});c=r/{CHANGE!r};key={key!r};h=json.loads((c/(key+'.ready.json')).read_text())
rows=[json.loads(x) for x in (r/'jobs'/key/'physical_updates.jsonl').read_text().splitlines()]
assert len(rows)==h['step'] and all(x['scientific_step']==i+1 for i,x in enumerate(rows))
base=json.loads((c/'CONTROLLER_CONFIG.json').read_text())
worker=json.loads((r/'host_migration_20261014T044549Z'/(key+'.config.json')).read_text())
worker.update(output=str(r/'jobs'/key),data=base['data'],reference=base['reference'],predecessor=str(r/'jobs'/worker['node']['predecessor']/'sealed.pt'))
h.update(checkpoint=str(c/key/'latest.pt'),entry_state=str(c/(key+'.entry.pt')),origin_host='bridge-gpu')
cfg=dict(worker=worker,checkpoint=h['checkpoint'],entry_state=h['entry_state'],compatibility=True,result=str(c/(key+'.preflight.json')),batch_reference=str(c/(key+'.batches.pt')))
p=c/(key+'.preflight.config.json');p.write_text(json.dumps(cfg))
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(base['placements'][key]),PYTHONPATH=base['worker_code_roots'].get(key,base['default_code_root']),HOST_SCRIPT=str(c/'code/host_preflight.py'),HOST_CONFIG=str(p),CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (c/(key+'.preflight.log')).open('x') as log:
 result=subprocess.run(['bash','./s',base['python'],'./n.py'],cwd=base['ops'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
assert result.returncode==0,'native preflight failed; origin remains stopped'
check=json.loads(Path(cfg['result']).read_text());assert check['status']=='PASS' and check['step']==h['step']
h['preflight_status']='PASS'
temp=c/(key+'.ADMITTED.tmp');temp.write_text(json.dumps(h,indent=2));os.replace(temp,c/(key+'.ADMITTED.json'))
print(json.dumps(h))
''')
    for _ in range(60):
        first = progress(key)
        if first and first['reported_step'] > admitted['step']:
            time.sleep(25)
            second = progress(key)
            if second and second['updates'] > first['updates']:
                break
        time.sleep(5)
    else:
        raise RuntimeError('target growth not confirmed; origin remains stopped: '+key)
    receipt = remote('bridge-gpu', f'''import json,os,signal,time
from pathlib import Path
r=Path({OLD!r});c=r/{CHANGE!r};key={key!r};h=json.loads((c/(key+'.ready.json')).read_text());e=h['old']
assert sum(1 for _ in (r/'jobs'/key/'physical_updates.jsonl').open())==h['step']
f=Path('/proc/%s/stat'%e['pid']).read_text().split(') ',1)[1].split();assert int(f[19])==e['birth'] and f[0]=='T'
os.kill(e['pid'],signal.SIGTERM);os.kill(e['pid'],signal.SIGCONT)
for _ in range(100):
 p=Path('/proc/%s/stat'%e['pid'])
 if not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z':break
 time.sleep(.05)
else:raise RuntimeError('origin retirement unconfirmed')
assert sum(1 for _ in (r/'jobs'/key/'physical_updates.jsonl').open())==h['step']
value=dict(status='ORIGIN_RETIRED_AFTER_VERIFIED_TARGET_GROWTH',old=e,target_first={first!r},target_second={second!r},checkpoint_step=h['step'],replayed_optimizer_updates=0,retired=time.time())
(c/(key+'.migrated.json')).write_text(json.dumps(value,indent=2));print(json.dumps(value))
''')
    transfer([CHANGE+'/'+key+'.migrated.json'])
    return receipt


def run():
    done = {}
    while len(done) < len(KEYS):
        available = remote('bridge-gpu', f'''import json
from pathlib import Path
c=Path({OLD!r})/{CHANGE!r}
print(json.dumps({{k:json.loads((c/(k+'.ready.json')).read_text()) for k in {KEYS!r} if (c/(k+'.ready.json')).exists()}}))
''')
        for key, ready in available.items():
            if key not in done:
                done[key] = handoff(key, ready)
                print('migrated', key, ready['step'], flush=True)
        if len(done) < len(KEYS):
            time.sleep(30)
    retired = remote('bridge-gpu', f'''import json,os,signal,time
from pathlib import Path
r=Path({OLD!r});c=r/{CHANGE!r};pause=json.loads((r/'PAUSED_BY_USER.json').read_text())
result={{}}
for name,e in [('controller',pause['controller']),*pause['watchers'].items()]:
 p=Path('/proc/%s/stat'%e['pid'])
 if p.exists():
  f=p.read_text().split(') ',1)[1].split();assert int(f[19])==e['birth']
  if f[0]!='Z':
   assert f[0]=='T';os.kill(e['pid'],signal.SIGTERM);os.kill(e['pid'],signal.SIGCONT)
 result[name]=e
value=dict(status='COMPLETE',origin_host='bridge-gpu',target_host='target-gpu',target_root={NEW!r},nodes={done!r},retired_cpu=result,formal_budget=511600,engineering_optimizer_updates=0,optimizer_replay=0,independent_test_access=False,other_projects_untouched=True,finished=time.time())
(c/'BRIDGE_STATUS.json').write_text(json.dumps(value,indent=2));(r/'HOST_MIGRATION_MY_GPU.json').write_text(json.dumps(value,indent=2))
(r/'STATUS.json').write_text(json.dumps(dict(status='ARCHIVED_MIGRATED',active_host='target-gpu',active_root={NEW!r},running={{}},migration_receipt=str(r/'HOST_MIGRATION_MY_GPU.json')),indent=2))
print(json.dumps(value))
''')
    transfer(['HOST_MIGRATION_MY_GPU.json', CHANGE+'/BRIDGE_STATUS.json'])
    print('All four exact handoffs COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        remote('target-gpu', f'''import json,time
from pathlib import Path
p=Path({NEW!r})/{CHANGE!r}/'BRIDGE_FAILURE.json'
p.write_text(json.dumps(dict(status='FAILED_HANDOFF_NO_AUTOMATIC_RETRY',error={str(exc)!r},time=time.time(),original_checkpoints_preserved=True),indent=2));print('{{}}')
''')
        raise
