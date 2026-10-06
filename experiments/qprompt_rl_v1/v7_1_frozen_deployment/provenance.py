"""Verify recorded lineage and patient-role identity without reading images."""
import hashlib,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.qprompt_rl_v1.v3b_lrref_endpoint import engine as e
from experiments.lcrseg.single_teacher_scd_v0_1 import data as primitive
from . import protocol as p

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def run(root,c):
    old=Path(c['old_campaign']);jobs=old/'scopes/pilot/jobs';doc=Path(c['code'])/'experiments/qprompt_rl_v1/v7_1_frozen_deployment'
    bindings=e.read(doc/'CODE_BINDINGS.json')
    for path,expected in bindings.items():
        assert sha(old/'code'/path)==expected and sha(Path(c['code'])/path)==expected,('code mismatch',path)
    from experiments.lcrseg.single_teacher_scd_v0_1.engine import UPSTREAM
    assert subprocess.check_output(['git','-C',c['reference'],'rev-parse','HEAD'],text=True).strip()==UPSTREAM
    subprocess.run(['git','-C',c['reference'],'diff','--quiet','HEAD'],check=True)
    src=Path(c['source'])/'SOURCE_S168';receipt=e.read(src/'receipt.json');source=load(src/'student.pt')
    assert receipt['status']=='SEALED' and receipt['step']==source['step']==8000 and source['identity']==receipt['identity'] and source['identity']['seed']==168
    assert e.tensor_fingerprint(source['student'])==receipt['student_hash']
    rows=primitive.metadata(c['data']);roles={};configs={};policies={};baselines=[]
    oldrows=e.read(old/'RESULTS.json')
    for d,h in p.old.DOMAINS.items():
        pts=sorted({r['patient_id'] for r in rows if r['site_or_vendor']==d and r['primary_20pct_split']=='train_labeled'})
        order=np.random.RandomState(e.stable('V7/roles',7101,d)).permutation(len(pts));held={pts[i] for i in order[:p.old.REWARD_LABELS[d]]}
        role=dict(fit=sorted(set(pts)-held),online=sorted(held),audit=[]);assert len(pts)==p.old.LABELS[d]
        ref=load(jobs/f'endpoint_{d}_ALL_U/ENTRY.private.pt')['state'];options=ref['state']['extra']['options'];configs[d]=options
        roles[d]=dict(sha256=digest(role),fit=len(role['fit']),reward=len(role['online']),U=p.old.UNLABELED[d])
        for r in [a for a in oldrows if a['domain']==d]:
            name=f'endpoint_{d}_{r["method"]}'+(f'_{r["controller"]}' if r['controller'] is not None else '')
            path=jobs/name;cfg=e.read(path/'CONFIG.private.json');entry=load(path/'ENTRY.private.pt');final=load(path/'student_latest.private.pt')
            assert cfg['commit']==entry['commit']==final['commit']==p.TRAINING
            assert cfg['domain']==d and cfg['seed']==168 and cfg['source']==c['source'] and cfg['data']==c['data']
            for state in (entry['state'],final['state']):
                extra=state['state']['extra'];identity=extra['provider_identity']
                assert extra['roles']==identity['roles']==role==e.read(path/'ROLES.private.json')
                assert identity['stage_source']['student_hash']==receipt['student_hash'] and identity['manifest']==primitive.MANIFEST_SHA and identity['split']==primitive.SPLIT_SHA and identity['role_seed']==7101
            assert e.same(entry['state']['state']['native']['student'],ref['state']['native']['student'])
            assert entry['state']['state']['extra']['options']==options
            actual=final['state']['state']['extra']['options'];expect=dict(options,lambda_U=0 if r['method']=='NO_U' else .5);assert actual==expect
            assert final['state']['state']['native']['step']==h and len(final['state']['exposure'])==p.old.UNLABELED[d]
            assert e.read(path/'FINAL.json')['status']=='COMPLETE' and e.read(path/'COMPLETION_AUDIT.json')['physical_calls']=={'endpoint':h}
            scores=e.read(path/'endpoint.scores.json');assert scores['step']==h
            for key in ('macro_Dice','rim','cup','disc_union'):assert scores['scores'][d][key]==r[key]
            assert scores['scores']['REFUGE']['macro_Dice']==r['old_REFUGE']
            source_old=e.read(path/'entry.scores.json')['scores']['REFUGE']['macro_Dice'];assert abs(source_old-(r['old_REFUGE']-r['old_change']))<1e-12
            baselines.append(dict(domain=d,method=r['method'],controller=r['controller'],source_old=source_old,role_sha256=digest(role),entry_state_hash=e.tensor_fingerprint(entry['state']['state']['native']['student']),status='PASS',training_commit=p.TRAINING))
        for s in p.old.SEEDS:
            path=jobs/f'learn_{d}_{s}';raw=load(path/'FROZEN_POLICY.private.pt');latest=load(path/'student_latest.private.pt')
            assert raw['commit']==latest['commit']==p.TRAINING and raw['domain']==d and raw['controller']==s
            assert e.same(raw['state'],latest['policy']) and e.read(path/'ROLES.private.json')==role
            assert e.read(path/'FINAL.json')['status']=='COMPLETE' and e.read(path/'CONFIG.private.json')['commit']==p.TRAINING
            assert all(torch.isfinite(t).all() for t in raw['state'].values())
            policies[f'{d}/{s}']=dict(sha256=sha(path/'FROZEN_POLICY.private.pt'),tensor_hash=e.tensor_fingerprint(raw['state']),training_commit=p.TRAINING,status='PASS')
    assert len(baselines)==18 and len(policies)==4
    audit=dict(status='PASS',baseline_commit=p.BASELINE,actual_training_commit=p.TRAINING,execution_commit=c['commit'],source_student_hash=receipt['student_hash'],source_training_commit=receipt['commit'],source_steps=8000,source_seed=168,policies=policies,roles=roles,manifest_sha256=primitive.MANIFEST_SHA,split_sha256=primitive.SPLIT_SHA,native_config=configs,native_code_bindings=bindings,reference_upstream=UPSTREAM,baselines=baselines,read_scope='metadata and saved states only; no image reads')
    e.write(root/'PROVENANCE_AUDIT.json',audit)
    env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],text=True).splitlines())
    e.write(root/'FROZEN_SCOPE.json',p.scope());(root/'EXPERIMENT_PLAN.md').write_text((doc/'EXPERIMENT_PLAN.md').read_text())
    e.write(root/'RUN_MANIFEST.json',dict(frozen_at_epoch=time.time(),execution_commit=c['commit'],provenance_sha256=sha(root/'PROVENANCE_AUDIT.json'),scope=p.scope(),environment=env,native_config=configs,source_hash=receipt['student_hash'],policies=policies,data_roles=roles,evaluation_order=[j['name'] for j in p.matrix() if j['job']=='endpoint']))
    return audit
