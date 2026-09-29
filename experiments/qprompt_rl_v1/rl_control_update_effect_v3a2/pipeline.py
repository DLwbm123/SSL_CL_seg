"""Phase ordering: fit-only features -> development -> locks -> transfer feedback."""
import copy,csv,json,os,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from .runtime import *
from . import policy
from rl_control_fine_intervention_v3a.control import folds

def prepare():
    if (ROOT/'CONTEXT_MANIFEST.private.json').exists():return
    bindings=read(PREVIOUS/'PREFIX_BINDINGS.private.json');data=read(PREVIOUS/'DATA_BINDING.json');manifest={};pools={}
    for domain in PLAN['domains']:
        doc=read(PREVIOUS/'schedules'/f'261__{domain}.private.json');unique={}
        for i,row in enumerate(doc['u_rows']):unique.setdefault(row['sha256'],i)
        assert len(unique)>=32,'insufficient distinct U payloads';anchor=[doc['u_rows'][doc['steps'][20*(60*j//16)]['u']]['sha256'] for j in range(16)];dev=list(dict.fromkeys(anchor));remaining=set(unique)-set(dev);dev+=sorted(remaining,key=lambda x:stable('dev',domain,x))[:16-len(dev)];transfer=sorted(set(unique)-set(dev),key=lambda x:stable('transfer',domain,x))[:16];assert len(dev)==len(transfer)==16 and not set(dev)&set(transfer);pools[domain]=dict(development=dev,transfer=transfer)
        for seed in (261,262,263):
            p=PREVIOUS/'schedules'/f'{seed}__{domain}.private.json';assert digest(p)==data['schedules'][f'{seed}__{domain}'];doc=read(p);scenes=[];role='development' if seed==261 else 'transfer';pool=pools[domain][role]
            for j in range(16):
                origin=20*(60*j//16);items=copy.deepcopy(doc['steps'][origin:origin+5]);first=anchor[j] if seed==261 else transfer[j]
                for h,item in enumerate(items):
                    selected=first if h==0 else pool[stable(seed,domain,j,h,'continuation')%16];item['u']=unique[selected];assert all(item[k]==items[0][k] for k in ('fit','online','audit'));assert not set(item['fit'])&(set(item['online'])|set(item['audit'])) and not set(item['online'])&set(item['audit'])
                scenes.append(dict(j=j,origin_t=origin,U_identity=first,items=items))
            for backbone in PLAN['backbones']:
                cell=f'S{seed}__{backbone}__{domain}';prefix=bindings.get(f'FROZEN_PREFIX__S{seed}__{backbone}__{domain}',{'valid':False});eligible=prefix['valid'] and Path(prefix['path']).is_file()
                if eligible:assert digest(prefix['path'])==prefix['sha256']
                manifest[cell]=dict(seed=seed,backbone=backbone,domain=domain,eligible=eligible,prefix=prefix,schedule_sha=digest(p),scenes=scenes,l_rows=doc['l_rows'],u_rows=doc['u_rows'],folds=folds([x['U_identity'] for x in scenes]))
    write(ROOT/'CONTEXT_MANIFEST.private.json',manifest);write(ROOT/'U_POOLS.private.json',pools);scales={}
    for b in PLAN['backbones']:
        for d in PLAN['domains']:
            p=PREVIOUS/'tasks'/f'R3A__{b}__{d}'/'R3A_DONE.json';scales[b+'__'+d]=dict(scale=read(p)['scale'],sha=digest(p))
    write(ROOT/'SCALES.json',scales);save(ROOT/'reports/DATA_BINDING.json',prefix_cells=len(manifest),eligible=sum(x['eligible'] for x in manifest.values()),distinct_U_by_domain={k:dict(development=len(v['development']),transfer=len(v['transfer']),disjoint=True) for k,v in pools.items()},manifest_sha=digest(ROOT/'CONTEXT_MANIFEST.private.json'),pool_sha=digest(ROOT/'U_POOLS.private.json'),feature_schema_sha=digest(SOURCE/'FEATURE_SCHEMA.json'),data_manifest_sha=data['manifest_sha'],split_sha=data['split_sha'])

def feature_table(cell):
    return np.array([np.mean([receipt(ROOT/'features'/cell/f'{j}_{r}.private.json')['features'] for r in range(3)],0) for j in range(16)])
def values(cell,role='online',h=5):
    out=np.empty((3,16,3))
    for r in range(3):
        for j in range(16):
            v=[receipt(ROOT/'panels'/cell/f'{j}_{r}_{a}.private.json')['horizons'][str(h)][role] for a in range(3)];assert all(x is not None and np.isfinite(x) for x in v),'missing feedback';out[r,j]=np.array(v)-v[2]
    return out

def train(cell,ledger):
    spec=read(ROOT/'CONTEXT_MANIFEST.private.json')[cell];x=feature_table(cell);online=np.array(receipt(ROOT/'training_targets'/f'{cell}.private.json')['online_h5']).mean(0);scale=read(ROOT/'SCALES.json')[spec['backbone']+'__'+spec['domain']]['scale'];fold=np.array(spec['folds']);methods=PLAN['learners']['methods']
    for split in list(range(4))+['all']:
        tr=np.arange(16) if split=='all' else np.flatnonzero(fold!=split);te=np.arange(16) if split=='all' else np.flatnonzero(fold==split);seed=stable(cell,split,'init')
        for method in methods:
            key=f'{cell}/{split}/{method}';path=ROOT/'fits'/cell/f'{split}_{method}.private.json'
            if path.exists():receipt(path);continue
            fit=dict(method=method) if method in ('FINE','EFFECT_RULE') else policy.fit(method,x[tr],online[tr]/scale,seed,ledger,key)
            pred=policy.predict(fit,x[te]);save(path,cell=cell,split=split,method=method,train_indices=tr.tolist(),test_indices=te.tolist(),fit=fit,prediction=pred,feature_digest=[digest(ROOT/'features'/cell/f'{j}_{r}.private.json') for j in tr for r in range(3)],scale=scale,shuffle_hash=__import__('hashlib').sha256(json.dumps(fit.get('diagnostics',{}).get('permutation',[])).encode()).hexdigest())

def freeze():
    paths=sorted((ROOT/'fits').glob('S261*/*.private.json'));assert len(paths)==160
    save(ROOT/'FINAL_POLICY_LOCK.json',files={str(p.relative_to(ROOT)):digest(p) for p in paths},feature_schema_sha=digest(SOURCE/'FEATURE_SCHEMA.json'),scale_sha=digest(ROOT/'SCALES.json'),methods=PLAN['learners']['methods'])

def predict_transfer():
    lock=receipt(ROOT/'FINAL_POLICY_LOCK.json');assert all(digest(ROOT/p)==v for p,v in lock['files'].items());manifest=read(ROOT/'CONTEXT_MANIFEST.private.json');files={}
    for cell,spec in manifest.items():
        if spec['seed']==261:continue
        x=feature_table(cell);dev='S261__'+spec['backbone']+'__'+spec['domain']
        for method in PLAN['learners']['methods']:
            fit=receipt(ROOT/'fits'/dev/f'all_{method}.private.json')['fit'];prediction=policy.predict(fit,x);path=ROOT/'predictions'/cell/f'{method}.private.json';save(path,cell=cell,method=method,prediction=prediction,policy_sha=digest(ROOT/'fits'/dev/f'all_{method}.private.json'),feature_shas=[digest(ROOT/'features'/cell/f'{j}_{r}.private.json') for j in range(16) for r in range(3)]);files[str(path.relative_to(ROOT))]=digest(path)
    assert len(files)==64;save(ROOT/'TRANSFER_PREDICTION_LOCK.json',files=files,final_policy_lock_sha=digest(ROOT/'FINAL_POLICY_LOCK.json'),sealed_at=time.time(),horizon=5,deterministic_primary=True)

def job(role,cell=None):
    old.provenance();old.setup();torch.set_num_threads(2);assert read(ROOT/'AUTHORIZATION.private.json')['run_authorized'];ledger=Ledger()
    try:
        if role=='qualify':
            from .qualification import qualify
            qualify(ledger)
        elif role=='prepare':prepare()
        elif role in ('features','panels'):
            from .worker import Student,Guard
            if role=='features' and not cell.startswith('S261'):receipt(ROOT/'FINAL_POLICY_LOCK.json')
            guard=Guard() if role=='features' else None;w=Student(cell,ledger,role=='features');spec=read(ROOT/'CONTEXT_MANIFEST.private.json')[cell]
            for scene in spec['scenes']:
                for repeat in range(3):
                    if guard:w.extract(scene,repeat,guard)
                    else:
                        for action in ([0,1,2],[1,2,0],[2,0,1])[repeat]:w.branch(scene,action,repeat)
            if role=='panels':save(ROOT/'training_targets'/f'{cell}.private.json',online_h5=values(cell).tolist())
        elif role=='train':train(cell,ledger)
        elif role=='freeze':freeze()
        elif role=='predict':predict_transfer()
        elif role=='report':
            from .report import report
            report()
        else:raise ValueError(role)
        save(ROOT/'jobs'/f'{role}_{cell or "all"}.json',role=role,cell=cell,status='COMPLETE',finished=time.time())
    finally:write(ROOT/'BUDGET.json',dict(ledger.count))

def supervise():
    old.provenance();assert read(ROOT/'AUTHORIZATION.private.json')['run_authorized'];lock=(ROOT/'OWNER.lock').open('a');__import__('fcntl').flock(lock,__import__('fcntl').LOCK_EX|__import__('fcntl').LOCK_NB)
    if not (ROOT/'SESSION.json').exists():save(ROOT/'SESSION.json',start=time.time(),deadline=time.time()+43200)
    else:receipt(ROOT/'SESSION.json')
    from r1_6_readout_v1.runtime import process_identity
    save(ROOT/'PROCESS.json',pid=os.getpid(),identity=process_identity(os.getpid()))
    cells=[f'S{s}__{b}__{d}' for s in (261,262,263) for b in PLAN['backbones'] for d in PLAN['domains']];steps=[('qualify',None),('prepare',None)]+[(role,cell) for cell in cells[:4] for role in ('features','panels','train')]+[('freeze',None)]+[('features',cell) for cell in cells[4:]]+[('predict',None)]+[('panels',cell) for cell in cells[4:]]+[('report',None)]
    for role,cell in steps:
        path=ROOT/'jobs'/f'{role}_{cell or "all"}.json'
        if path.exists():receipt(path);continue
        deadline();size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert size<32*2**30,'32GiB storage budget'
        save(ROOT/'SUPERVISOR.json',role=role,cell=cell,status='RUNNING')
        with (ROOT/'worker.private.log').open('ab') as log:
            env=os.environ.copy();env.update(EXEC_ROLE=role,EXEC_CELL=cell or '')
            child=subprocess.Popen(['python3.12','-'],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            c.append(ROOT/'PROCESS_LEDGER.jsonl',dict(pid=child.pid,identity=process_identity(child.pid),role=role,cell=cell,code_commit=CODE));child.stdin.write(b'from rl_control_update_effect_v3a2.pipeline import job\nimport os\njob(os.environ["EXEC_ROLE"],os.environ["EXEC_CELL"] or None)\n');child.stdin.close()
            try:code=child.wait(timeout=max(1,read(ROOT/'SESSION.json')['deadline']-time.time()))
            except subprocess.TimeoutExpired:
                assert process_identity(child.pid) is not None
                child.terminate();code=child.wait(timeout=30)
        if code:save(ROOT/'FAILURE.private.json',role=role,cell=cell,returncode=code,time=time.time());return
    save(ROOT/'FINAL.json',status='COMPLETE',finished=time.time(),retained_student_updates=0,medical_endpoints=0)
