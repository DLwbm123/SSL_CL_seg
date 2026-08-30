"""One authorized v2 attempt: all72 support units ->432 jobs -> admission."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from di_dmpa_gate1.bootstrap import registered_draws, matched_cosines
from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1.gate1a_reporting import artifact_manifest
from di_dmpa_gate1.recovery import request_stop, stop_requested, SHUTDOWN_TIMEOUT_SECONDS
from .binding import (PREREG, AUTH, CLOSURE, PLAN_SHA, ATTEMPT1, PANELS, NonfiniteFeature,
    ProtocolError, IncompletePanel, audit_inputs, check_hash, read_json, require, sha256, verify, write_json, write_text, reuse_sampling_plan)
from .features import extract_checkpoint, load_cache
from .census import validate_features, compile_census
from .geometry import fit, metrics, bootstrap_weights
from .reporting import report

ROOT=Path(__file__).resolve().parents[1]


def serial_fit(result):
    return {**result,'centers':result['centers'].tolist(),'active':result['active'].tolist()}


def case_weights(case_ids):
    count=Counter(case_ids)
    return np.asarray([1/len(count)/count[c] for c in case_ids],dtype=np.float64)


def geometry_job(task):
    output,old,metadata,entries,units,c,K=task
    first=entries['train_labeled'];arrays={};layouts={};expected={};nulls={}
    for role in ('train_labeled','val'):
        cache=next(x for x in entries[role]['class_caches'] if x['class_id']==c)
        arrays[role]=load_cache(output,cache);layouts[role]=sample_layout(units[role],c)
        expected[role]=len(layouts[role]['uids']);nulls[role]=cache['null_count']
    data=arrays['train_labeled'];layout=layouts['train_labeled']
    original=fit(data['directions'],data['active_mask'],layout['weights'],K,seed=first['seed'],stage=first['stage_index'],class_id=c,uid_rank=layout['uid_rank'])
    role_metrics={};strata={}
    for role in ('train_labeled','val'):
        a=arrays[role];l=layouts[role]
        role_metrics[role]=metrics(a['directions'],a['active_mask'],l['weights'],original['centers'],original['active'],l['uid_rank'])
        strata[role]={}
        for name,mask in [('boundary',l['boundary']),('interior',~l['boundary'])]:
            strata[role][name]=None if not mask.any() else metrics(a['directions'][mask],a['active_mask'][mask],
                case_weights(l['case_ids'][mask]),original['centers'],original['active'],l['uid_rank'][mask])
    boot=[]
    for draw in registered_draws(old,first['seed'],first['stage_index']):
        weights=bootstrap_weights(layout['case_ids'],draw['case_ids_with_replacement'])
        result=fit(data['directions'],data['active_mask'],weights,K,seed=first['seed'],stage=first['stage_index'],class_id=c,
            replicate=draw['replicate'],uid_rank=layout['uid_rank'])
        boot.append(dict(replicate=draw['replicate'],case_draw_sha256=draw['case_draw_sha256'],fit=serial_fit(result),
            matched_cosines=matched_cosines(original['centers'],original['active'],result['centers'],result['active']).tolist()))
    row=dict(panel_id=first['panel_id'],seed=first['seed'],stage_index=first['stage_index'],domain=first['domain'],class_id=c,K=K,
        metadata={**metadata,'panel_id':first['panel_id']},admission_radius_field='R95_null_worst_case',metrics=role_metrics,
        expected_registered_counts=expected,expected_null_counts=nulls,boundary_interior=strata,bootstrap=boot,fit=serial_fit(original))
    path=Path(output)/'geometry_units'/f'{first["panel_id"]}_seed{first["seed"]}_stage{first["stage_index"]}_class{c}_K{K}.json'
    write_json(path,row);print(f'geometry complete {path.name}',flush=True)
    return str(path)


def extract_shard(args):
    p,old,_=verify(ROOT,args.code_commit,remote=False)
    metadata=read_json(args.output/'GATE1A_V2_RUN_METADATA.json')
    check_hash(args.output/'SHARED_GEOMETRY_SAMPLING_PLAN.json',PLAN_SHA)
    plan=read_json(args.output/'SHARED_GEOMETRY_SAMPLING_PLAN.json')
    write_json(args.output/'shards'/f'extract_{args.shard}_metadata.json',dict(metadata,shard=args.shard,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    for i,checkpoint in enumerate(old['immutable_baseline']['checkpoint_inputs']):
        if i%args.shards!=args.shard:continue
        if stop_requested(args.output):
            write_json(args.output/'shards'/f'extract_{args.shard}_cancelled.json',dict(before_checkpoint=checkpoint['checkpoint_id'],current_checkpoint_guard_completed=True));return
        extract_checkpoint(ROOT,args.data_root,old,plan,checkpoint,args.output,metadata,device='cuda:0')


def feature_workers(args):
    devices=args.gpus.split(',');processes=[];logs=[];deadline=None
    try:
        for index,gpu in enumerate(devices):
            command=[sys.executable,'-m','di_dmpa_gate1_v2.runner','extract','--code-commit',args.code_commit,
                '--output',str(args.output),'--data-root',str(args.data_root),'--shard',str(index),'--shards',str(len(devices))]
            log=(args.output/f'extract_shard{index}.txt').open('x');logs.append(log)
            processes.append(subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (0,None) for p in processes):request_stop(args.output,'v2 feature shard failure')
            if stop_requested(args.output) and deadline is None:deadline=time.monotonic()+SHUTDOWN_TIMEOUT_SECONDS
            if deadline is not None and time.monotonic()>deadline:
                for p in processes:
                    if p.poll() is None:p.terminate()
                write_json(args.output/'FORCED_SHARD_TERMINATION.json',dict(timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS));break
            time.sleep(1)
        codes=[p.wait() for p in processes]
        if any(codes):
            errors=[read_json(p) for p in sorted((args.output/'failures').glob('*.json'))]
            error=RuntimeError(f'feature shard exits={codes}; failures={errors}')
            error.status=errors[0]['status'] if errors else 'BLOCKED_INCOMPLETE_PANEL'
            raise error
    finally:
        for log in logs:log.close()


def run(args):
    require(not args.output.exists(),'unique v2 attempt already exists; no overwrite/retry')
    args.output.mkdir(parents=True)
    metadata=dict(v2_preregistration_git_commit=PREREG,execution_authorization_git_commit=AUTH,v1_closure_git_commit=CLOSURE,
        diagnostic_code_git_commit=args.code_commit,model_optimizer_steps=0,transport_optimizer_steps=0,
        method_registered=False,di_dmpa_training_launched=False,hidden_gt_training_usage='none',test_gt_usage='none')
    error=None;census=None
    try:
        p,old,metadata=verify(ROOT,args.code_commit)
        require(args.output.parent.name==PREREG and args.output.name==f'gate1a_v2_{args.code_commit}_attempt1','wrong v2 namespace')
        tests=read_json(args.tests/'GATE1A_V2_UNIT_INTEGRATION_TEST_REPORT.json')
        require(tests['status']=='PASS' and tests['diagnostic_code_git_commit']==args.code_commit,'invalid exact-code tests')
        integration=read_json(args.tests/'GATE1A_V2_REAL_INTEGRATION.json')
        require(integration['status']=='PASS' and integration['known_null_row_retained'] and integration['model_unchanged'],'known-zero integration failed')
        for name in ('GATE1A_V2_UNIT_INTEGRATION_TEST_REPORT.json','pytest.xml','pytest_output.txt','GATE1A_V2_REAL_INTEGRATION.json'):
            shutil.copy2(args.tests/name,args.output/name)
        metadata.update(execution_scope='GATE1A_V2_ONLY',gpus=args.gpus.split(','),geometry_workers=args.workers,case_batch_size=8,
            v1_attempt1_code='8f4a71a5ea8d145183a3007ccd398ab79387478e',v1_attempt1_report='945b484072cb9f2757be98df34e5d72844596e84',
            v1_attempt2_code='a89716ddbd2eccbe76c574e97e520d424aa923ab',v1_attempt2_report='606a5c53a37d0e4c9605415e8b38a1f177d1604f',
            v1_status='CLOSED_ASSUMPTION_FALSIFIED',sampling_plan_reused=True,old_raw_caches_reused=False)
        write_json(args.output/'GATE1A_V2_RUN_METADATA.json',metadata)
        inputs=audit_inputs(ROOT,args.data_root,old)
        from di_dmpa_gate1.feature_extraction import audit_checkpoint_contents
        inputs['checkpoint_content_audit']=audit_checkpoint_contents(old,ROOT)
        write_json(args.output/'GATE1A_V2_INPUT_AUDIT.json',dict(inputs,metadata=metadata))
        plan,_=reuse_sampling_plan(ATTEMPT1/'SHARED_GEOMETRY_SAMPLING_PLAN.json',args.output)
        feature_workers(args)
        entries=[read_json(p) for p in sorted((args.output/'feature_units').glob('*.json'))]
        validate_features(args.output,entries,plan,metadata)
        audits=[read_json(p) for p in sorted((args.output/'immutability').glob('*.json'))]
        expected={c['checkpoint_id'] for c in old['immutable_baseline']['checkpoint_inputs']}
        require(len(audits)==18 and {a['checkpoint_id'] for a in audits}==expected and all(a['bitwise_unchanged'] for a in audits),'missing/mutated checkpoint audit')
        for c in old['immutable_baseline']['checkpoint_inputs']:check_hash(c['path'],c['sha256'])
        write_json(args.output/'GATE1A_V2_MODEL_IMMUTABILITY_AUDIT.json',dict(status='PASS',metadata=metadata,checkpoints=audits,all18_disk_hashes_unchanged=True))
        write_json(args.output/'GATE1A_V2_FEATURE_CACHE_MANIFEST.json',dict(status='PASS',metadata=metadata,feature_units=entries,raw_tensors_git_policy='remote_only'))
        census=compile_census(args.output,entries,metadata)
        write_json(args.output/'GEOMETRY_START_BARRIER.json',dict(status='PASS',census_sha256=sha256(args.output/'GATE1A_V2_FEATURE_SUPPORT_CENSUS.json'),
            feature_units=72,checkpoint_audits=18,clustering_jobs_started_before_this_barrier=0))
        tasks=[]
        for panel in PANELS:
            for seed in range(3):
                for stage in range(3):
                    es={r:next(e for e in entries if e['panel_id']==panel and e['seed']==seed and e['stage_index']==stage and e['role']==r) for r in ('train_labeled','val')}
                    us={r:next(u for u in plan['units'] if u['seed']==seed and u['stage_index']==stage and u['role']==r) for r in ('train_labeled','val')}
                    for c in range(3):
                        for K in (1,2,3,5):tasks.append((str(args.output),old,metadata,es,us,c,K))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:paths=list(pool.map(geometry_job,tasks,chunksize=1))
        rows=[read_json(p) for p in paths]
        report(args.output,metadata,rows,census)
    except Exception as exc:
        error=exc
        write_text(args.output/'failure_traceback.txt',traceback.format_exc())
        write_json(args.output/'GATE1A_V2_STATUS.json',dict(metadata,prototype_geometry_status=getattr(exc,'status','BLOCKED_PROTOCOL_OR_LEAKAGE'),
            feature_units_completed=len(list((args.output/'feature_units').glob('*.json'))),geometry_jobs_completed=len(list((args.output/'geometry_units').glob('*.json'))),
            selected_K=None,passing_K=None,A1_A6_computed=False,errors=[str(exc)],next_action='STOP_FOR_INDEPENDENT_REVIEW'))
        write_text(args.output/'GATE1A_V2_FINAL_REPORT.md',f'# Gate1A v2 blocked\n\n{type(exc).__name__}: {exc}\n\nNo admission PASS. Preserve this unique attempt; no retry or downstream work.\n')
    finally:
        write_text(args.output/'GATE1A_V2_EXACT_COMMANDS.md','# Exact command\n\n```text\n'+str(ROOT)+'\n'+sys.executable+' '+' '.join(sys.argv)+'\n```\n')
        write_text(args.output/'GATE1A_V2_FAILURES_AND_WARNINGS.md','# Failures and warnings\n\n'+('No fatal error. All non-converged fits and inactive slots retained.\n' if error is None else str(error)+'\n'))
        # The shared manifest routine is byte-only; publish a v2-named copy too.
        artifact_manifest(args.output)
        shutil.copy2(args.output/'GATE1A_ARTIFACT_MANIFEST.json',args.output/'GATE1A_V2_ARTIFACT_MANIFEST.json')
    if error is not None:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','extract'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--code-commit',required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/root/LCRSeg'));parser.add_argument('--tests',type=Path)
    parser.add_argument('--gpus',default='0,1');parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    if args.action=='run':run(args)
    else:
        try:extract_shard(args)
        except Exception as error:
            write_json(args.output/'failures'/f'extract_{args.shard}.json',dict(status=getattr(error,'status','BLOCKED_PROTOCOL_OR_LEAKAGE'),error=str(error),traceback=traceback.format_exc()))
            request_stop(args.output,'v2 feature failure after guard exit');raise


if __name__=='__main__':main()
