"""Read-only postrun audit, no forward/fit; export the public report subset."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

CODE='f2a3ed7476323119b1a4fa22481b44038bc4148c'
ROOT=Path('/root/SSL_CL_gate1b_v2/experiments/lcrseg')
RUN=Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2/b20f186deff287843f3c9f18bf4ab5633908f441')/f'gate1b_v2_{CODE}_attempt1'
POST=Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2_postrun')/CODE/'attempt1'
sys.path.insert(0,str(ROOT))
import numpy as np
from di_dmpa_gate1b_v2.binding import require, read_json, write_json, check_hash, sha256, verify
from di_dmpa_gate1b_v2.pairs import load_pair
from di_dmpa_gate1b_v2.transport import feature_error
from di_dmpa_gate1b_v2.runner import disk_audit

exit_receipt=read_json('/root/gate1b_v2_formal_f2a3ed7_exit.json')
require(exit_receipt['exit_code']==0,'formal process did not complete successfully')
p,old,frozen,verified=verify(ROOT,CODE)
status=read_json(RUN/'GATE1B_V2_STATUS.json')
require(status['transport_status'] in ('PASS_TRANSPORT_SUPPORTED','FAIL_TRANSPORT_NOT_SUPPORTED'),'not a complete scientific outcome')
require(status['transport_optimizer_steps']==6000 and status['model_optimizer_steps']==0,'wrong updates')
for k in ('method_registered','di_dmpa_training_launched','Gate1C'):require(status[k] is False,'forbidden downstream action')
require(status['hidden_gt_training_usage']==status['test_gt_usage']=='none','forbidden GT usage')
require(status['next_action']=='STOP_FOR_INDEPENDENT_REVIEW','wrong stopping point')
manifest=read_json(RUN/'GATE1B_V2_ARTIFACT_MANIFEST.json')
require(manifest['file_count']==len(manifest['artifacts']),'manifest file count')
required={r['path'] for r in manifest['artifacts']}
actual={str(f.relative_to(RUN)) for f in RUN.rglob('*') if f.is_file() and f.name not in ('GATE1B_V2_ARTIFACT_MANIFEST.json','GATE1B_V2_ARTIFACT_MANIFEST.sha256')}
require(actual==required,'manifest coverage mismatch')
for row in manifest['artifacts']:
    require((RUN/row['path']).stat().st_size==row['bytes'],'artifact byte count')
    check_hash(RUN/row['path'],row['sha256'])
require(sum(r['bytes'] for r in manifest['artifacts'])==manifest['total_bytes'],'manifest total bytes')
plan=read_json(RUN/'SHARED_TRANSPORT_COORDINATE_PLAN.json')
transports=[read_json(q) for q in sorted((RUN/'transport_models').glob('seed*_stage*.json'))]
oracles=[read_json(q) for q in sorted((RUN/'oracle_units').glob('*.json'))]
require(len(transports)==6 and len(oracles)==9,'incomplete map/evaluator coverage')
max_error=0.
for t in transports:
    for partition in ('fit','holdout'):
        unit=next(u for u in plan['units'] if (u['seed'],u['stage_index'],u['partition'])==(t['seed'],t['stage_index'],partition))
        entry=read_json(RUN/'paired_units'/f'seed{t["seed"]}_stage{t["stage_index"]}_{partition}.json')
        data=load_pair(RUN,entry,unit)
        for method,model in t['models'].items():
            recomputed=feature_error(data,model);reported=t['feature_errors'][partition][method]
            for field in ('full_null_aware_support_error','AA_conditional_error','support_constant_term'):
                error=abs(recomputed[field]-reported[field]);max_error=max(max_error,error)
                require(error<=1e-12,'recomputed feature error differs')
    with (RUN/t['trace_path']).open(newline='') as handle:trace=list(csv.DictReader(handle))
    require([int(r['step']) for r in trace]==list(range(1001)),'incomplete update trajectory')
    require(all(r['finite']==r['gradient_finite']=='True' and float(r['minimum_raw_output_norm'])>1e-12 for r in trace),'invalid step output')
# Independent, direct numerical B1--B6 inequalities, using no rounded table values.
gates={}
def change(ref,new,improve):
    if ref==0:return 0. if new==0 else None
    return (ref-new)/ref if improve else (new-ref)/ref
def holds(ref,new,threshold,improve):
    v=change(ref,new,improve)
    return v is not None and (v>=threshold if improve else v<=threshold)
b1=[];b2=[]
for stage in (1,2):
    selected=[t for t in transports if t['stage_index']==stage]
    errors={m:[t['feature_errors']['holdout'][m]['full_null_aware_support_error'] for t in selected] for m in ('T0','T2')}
    b1.append(holds(float(np.mean(errors['T0'])),float(np.mean(errors['T2'])),.15,True))
    b2.append(sum(b<a for a,b in zip(errors['T0'],errors['T2']))>=2)
immediate=[o for o in oracles if o['kind']=='immediate'];chains=[o for o in oracles if o['kind']=='chain']
angles={m:[o['metrics'][m]['class_angles'][c]['mean_angular_error'] for o in immediate for c in (1,2)] for m in ('T0','T2')}
gates['B1']=all(b1);gates['B2']=all(b2)
gates['B3']=holds(float(np.mean(angles['T0'])),float(np.mean(angles['T2'])),.10,True)
gates['B4']=all(holds(a,b,.05,False) for a,b in zip(angles['T0'],angles['T2']))
gates['B5']=all(o['metrics']['T0']['accuracy']['macro_accuracy']-o['metrics']['T2']['accuracy']['macro_accuracy']<=.005 for o in oracles)
gates['B6']=all(holds(o['metrics']['T0']['class_angles'][c]['mean_angular_error'],o['metrics']['T2']['class_angles'][c]['mean_angular_error'],.05,False) for o in chains for c in (1,2))
def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value,float) else True
gates['B7']=all(finite(read_json(f)) for f in RUN.rglob('*.json'))
require(gates=={k:v['pass_'] for k,v in status['B1_B7'].items()},'independent gate result differs')
model_audit=read_json(RUN/'GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json')
require(model_audit['model_load_guards']==21 and model_audit['all_model_states_unchanged'],'missing immutability evidence')
checkpoints=disk_audit(p,frozen)
POST.mkdir(parents=True,exist_ok=False)
audit=dict(status='PASS',diagnostic_code_commit=CODE,formal_exit_receipt=exit_receipt,formal_artifact_manifest_sha256=sha256(RUN/'GATE1B_V2_ARTIFACT_MANIFEST.json'),
    formal_files_verified=manifest['file_count'],formal_bytes_verified=manifest['total_bytes'],registered_pairs=638976,paired_units=12,
    transport_fits=6,oracle_units=9,model_load_guards=21,transport_updates=6000,model_updates=0,
    cached_feature_metric_max_abs_error=max_error,independent_B1_B7=gates,all_JSON_numbers_finite=gates['B7'],
    B0_checkpoint_sha256_after_postrun=checkpoints,no_additional_forward=True,no_additional_fit=True,method_registered=False,Gate1C=False)
write_json(POST/'GATE1B_V2_POSTRUN_AUDIT.json',audit)
included=[RUN/r['path'] for r in manifest['artifacts'] if not r['path'].endswith('.npy') and r['path']!='FROZEN_GEOMETRY_SAMPLING_PLAN.json']
included += [RUN/'GATE1B_V2_ARTIFACT_MANIFEST.json',RUN/'GATE1B_V2_ARTIFACT_MANIFEST.sha256']
copy=dict(diagnostic_code_commit=CODE,formal_attempt=str(RUN),included=[dict(path=str(f.relative_to(RUN)),bytes=f.stat().st_size,sha256=sha256(f)) for f in included],
    excluded=[r for r in manifest['artifacts'] if r['path'].endswith('.npy') or r['path']=='FROZEN_GEOMETRY_SAMPLING_PLAN.json'],
    reason='Raw paired/oracle arrays remain remote; existing frozen geometry plan is not duplicated publicly; new transport coordinate plan is included.')
write_json(POST/'GATE1B_V2_PUBLIC_COPY_MANIFEST.json',copy)
archive=Path('/root/gate1b_v2_report_f2a3ed7.tar.gz')
with tarfile.open(archive,'x:gz') as tar:
    for f in included:tar.add(f,arcname=RUN.name+'/'+str(f.relative_to(RUN)),recursive=False)
    for f in sorted(POST.glob('*.json')):tar.add(f,arcname='postrun_f2a3ed7_attempt1/'+f.name,recursive=False)
print(json.dumps(dict(audit=audit,public_files=len(included),excluded_files=len(copy['excluded']),archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive)),indent=2),flush=True)
