"""Gate 1A only. Plan -> all features -> all fits -> complete-panel adjudication."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .binding import (AUTHORIZATION, FILE_HASHES, H, PANELS, PREREG, NumericalError, ProtocolError,
                      check_hash, read_json, require, run_metadata, sha256, verify_registration, write_json, write_text)
from .bootstrap import matched_cosines, multiplicity_weights, registered_draws
from .geometry_metrics import geometry
from .sampling import sample_layout
from .spherical_kmeans import fit
from .recovery import (ATTEMPT1, SHUTDOWN_TIMEOUT_SECONDS, request_stop, stop_requested,
                       reuse_sampling_plan, verify_localization, norm_audit_reports)

ROOT=Path(__file__).resolve().parents[1]


def serial_fit(result):
    return {**result,'centers':result['centers'].tolist(),'active':result['active'].tolist()}


def _case_equal_weights(case_ids):
    counts=Counter(case_ids)
    return np.asarray([1/len(counts)/counts[c] for c in case_ids],dtype=np.float64)


def geometry_job(task):
    output,prereg,metadata,entries,units,c,K=task
    first=entries['train_labeled']
    panel,seed,stage=first['panel_id'],first['seed'],first['stage_index']
    arrays={};layouts={}
    for role in ('train_labeled','val'):
        cache=next(x for x in entries[role]['class_caches'] if x['class_id']==c)
        check_hash(Path(output)/cache['path'],cache['sha256'])
        arrays[role]=np.load(Path(output)/cache['path'],mmap_mode='r',allow_pickle=False)
        layouts[role]=sample_layout(units[role],c)
        require(arrays[role].shape==(len(layouts[role]['uids']),16),'cache/coordinate count mismatch')
    layout=layouts['train_labeled']; x=arrays['train_labeled']
    original=fit(x,layout['weights'],K,seed=seed,stage=stage,class_id=c,uid_rank=layout['uid_rank'])
    metrics={}; strata={}
    for role in ('train_labeled','val'):
        lay=layouts[role]
        metrics[role]=geometry(arrays[role],lay['weights'],original['centers'],original['active'],lay['uid_rank'])
        metrics[role]['dormant_assignment_count']=sum(v==0 for v in metrics[role]['occupancy'])
        strata[role]={}
        for name,mask in [('boundary',lay['boundary']),('interior',~lay['boundary'])]:
            strata[role][name]=None if not mask.any() else geometry(arrays[role][mask],_case_equal_weights(lay['case_ids'][mask]),
                    original['centers'],original['active'],lay['uid_rank'][mask])
    bootstrap=[]
    allowed_cases={case['case_id'] for case in units['train_labeled']['cases']}
    for draw in registered_draws(prereg,seed,stage):
        require(set(draw['case_ids_with_replacement']).issubset(allowed_cases),'unregistered bootstrap case')
        weights=multiplicity_weights(layout['case_ids'],draw['case_ids_with_replacement'])
        fitted=fit(x,weights,K,seed=seed,stage=stage,class_id=c,replicate=draw['replicate'],uid_rank=layout['uid_rank'])
        bootstrap.append(dict(replicate=draw['replicate'],case_draw_sha256=draw['case_draw_sha256'],
                    matched_cosines=matched_cosines(original['centers'],original['active'],fitted['centers'],fitted['active']).tolist(),
                    fit=serial_fit(fitted)))
    result=dict(panel_id=panel,seed=seed,stage_index=stage,domain=first['domain'],class_id=c,K=K,
                metadata={**metadata,'panel_id':panel},metrics=metrics,boundary_interior=strata,bootstrap=bootstrap,fit=serial_fit(original))
    path=Path(output)/'geometry_units'/f'{panel}_seed{seed}_stage{stage}_class{c}_K{K}.json'
    write_json(path,result)
    print(f'geometry complete {path.name}',flush=True)
    return str(path)


def validate_feature_manifest(output,manifest,plan,metadata):
    keys=[(x['panel_id'],x['seed'],x['stage_index'],x['role']) for x in manifest]
    expected={(p,s,t,r) for p in PANELS for s in range(3) for t in range(3) for r in ('train_labeled','val')}
    require(len(keys)==len(set(keys)) and set(keys)==expected,'feature panels incomplete or duplicate')
    for entry in manifest:
        require(entry['metadata']['sampling_plan_sha256']==metadata['sampling_plan_sha256'],'feature sampling plan differs')
        unit=next(u for u in plan['units'] if u['seed']==entry['seed'] and u['stage_index']==entry['stage_index'] and u['role']==entry['role'])
        require(entry['sampling_unit_sha256']==H(unit),'feature coordinates/multiplicity differ')
        summary=entry['diagnostics']['summary']
        require(entry['metadata']['recovery_diagnostic_code_git_commit']==metadata['recovery_diagnostic_code_git_commit'],
                'feature recovery code differs')
        require(summary['case_count']==len(unit['cases']),'incomplete per-case norm diagnostics')
        if summary['full_map_nonfinite_count'] or any(summary['registered_zero_count_by_class'].values()) or any(summary['registered_nonfinite_count_by_class'].values()):
            raise NumericalError('invalid registered features or nonfinite full map before geometry')
        for c in range(3):
            require(summary['registered_count_by_class'][str(c)]==sum(len(case['classes'][c]['coordinates']) for case in unit['cases']),
                    'registered norm diagnostic coverage differs')
        require({c['class_id'] for c in entry['class_caches']}=={0,1,2},'missing class feature cache')
        for cache in entry['class_caches']:
            check_hash(Path(output)/cache['path'],cache['sha256'])
            require(cache['shape'][1]==16 and cache['dtype']=='float64','wrong cache dtype/dimension')


def extraction_shard(args):
    from .feature_extraction import extract_checkpoint
    metadata=read_json(args.output/'GATE1A_RUN_METADATA.json')
    prereg,_=verify_registration(ROOT,args.code_commit,verify_remote=False)
    check_hash(args.output/'SHARED_GEOMETRY_SAMPLING_PLAN.json',metadata['sampling_plan_sha256'])
    plan=read_json(args.output/'SHARED_GEOMETRY_SAMPLING_PLAN.json')
    write_json(args.output/'shards'/f'extract_{args.shard}_metadata.json',{**metadata,'shard':args.shard,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'seed_excludes_shard_schedule':True})
    checkpoints=[c for index,c in enumerate(prereg['immutable_baseline']['checkpoint_inputs']) if index%args.shards==args.shard]
    checkpoint_sequence(checkpoints,args.output,args.shard,
        lambda checkpoint: extract_checkpoint(ROOT,args.data_root,prereg,plan,checkpoint,args.output,metadata,device='cuda:0'))


def checkpoint_sequence(checkpoints,output,shard,execute):
    for checkpoint in checkpoints:
        if stop_requested(output):
            write_json(Path(output)/'shards'/f'extract_{shard}_cancelled.json',
                dict(cancelled_before_checkpoint=checkpoint['checkpoint_id'],current_checkpoint_guard_completed=True))
            return
        execute(checkpoint)  # This returns/raises only AFTER ImmutabilityGuard.__exit__.


def _extraction_workers(args,metadata):
    devices=args.gpus.split(',')
    processes=[];logs=[]
    try:
        for shard,gpu in enumerate(devices):
            command=[sys.executable,'-m','di_dmpa_gate1.gate1a_runner','extract','--output',str(args.output),
                     '--data-root',str(args.data_root),'--code-commit',args.code_commit,'--shard',str(shard),'--shards',str(len(devices))]
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
            log=(args.output/f'extract_shard{shard}.txt').open('x')
            logs.append(log)
            processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        stop_deadline=None
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                request_stop(args.output,'feature shard failed')
            if stop_requested(args.output) and stop_deadline is None:
                stop_deadline=time.monotonic()+SHUTDOWN_TIMEOUT_SECONDS
            if stop_deadline is not None and time.monotonic()>=stop_deadline:
                forced=[]
                for i,p in enumerate(processes):
                    if p.poll() is None:
                        p.terminate(); forced.append(i)
                write_json(args.output/'FORCED_SHARD_TERMINATION.json',dict(shards=forced,
                    timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS,after_state_may_be_missing=True))
                break
            time.sleep(1)
        codes=[p.wait() for p in processes]
        if any(c!=0 for c in codes):
            errors=[read_json(p) for p in sorted((args.output/'failures').glob('*.json'))]
            if any(e['status']=='BLOCKED_PROTOCOL_OR_LEAKAGE' for e in errors):
                raise ProtocolError(f'feature shard failure: {errors}')
            raise NumericalError(f'feature shard failure: {errors}; exits={codes}')
    finally:
        for log in logs:
            log.close()


def _run(args):
    from .binding import audit_inputs
    from .gate1a_reporting import artifact_manifest, blocked_report, report
    require(not args.output.exists(),'attempt already exists; refusing overwrite')
    args.output.mkdir(parents=True)
    metadata=dict(preregistration_git_commit=PREREG,authorization_git_commit=AUTHORIZATION,diagnostic_code_git_commit=args.code_commit,
                  preregistration_md_sha256=FILE_HASHES['md'],preregistration_json_sha256=FILE_HASHES['json'],
                  model_optimizer_steps=0,transport_optimizer_steps=0,method_registered=False,di_dmpa_training_launched=False,
                  hidden_gt_training_usage='none',test_gt_usage='none',primary_feature_source='ema_teacher',feature_source_selection_performed=False)
    error=None
    try:
        prereg,receipt=verify_registration(ROOT,args.code_commit)
        require(args.localization is not None,'known-failure localization report required')
        localization=verify_localization(args.localization,args.code_commit)
        require(args.output.name==f'gate1a_formal_{args.code_commit}_attempt2','wrong recovery attempt namespace')
        require(args.tests.is_dir(),'test evidence directory missing')
        test_report=read_json(args.tests/'GATE1A_RECOVERY_UNIT_INTEGRATION_TEST_REPORT.json')
        require(test_report['status']=='PASS' and test_report['diagnostic_code_git_commit']==args.code_commit,'unit/integration test evidence invalid')
        for name in ('GATE1A_RECOVERY_UNIT_INTEGRATION_TEST_REPORT.json','GATE1A_RECOVERY_PYTEST.xml','GATE1A_RECOVERY_PYTEST_OUTPUT.txt'):
            shutil.copy2(args.tests/name,args.output/name)
        shutil.copy2(args.localization,args.output/'GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json')
        write_json(args.output/'GATE1A_SETUP_METADATA.json',{**receipt,'phase':'INPUT_AUDIT_AND_SHARED_PLAN_ONLY_NO_FEATURE_WORKERS',
                    'sampling_hash_lifecycle':'complete RUN/shard metadata is emitted only after plan exists and before any model/geometry worker',
                    'preregistration_json_sha256':FILE_HASHES['json'],'preregistration_md_sha256':FILE_HASHES['md']})
        from .feature_extraction import audit_checkpoint_contents
        input_audit=audit_inputs(ROOT,args.data_root,prereg)
        input_audit['checkpoint_content_audit']=audit_checkpoint_contents(prereg,ROOT)
        input_audit['checkpoint_tensor_schema_check']='PASS_ALL_18_BEFORE_SAMPLING'
        plan,sampling_sha=reuse_sampling_plan(ATTEMPT1/'SHARED_GEOMETRY_SAMPLING_PLAN.json',args.output)
        metadata=run_metadata(prereg,receipt,sampling_sha,panel_id='ALL_FOUR_SEPARATE')
        metadata.update(execution_scope='GATE1A_ONLY',gpus=args.gpus.split(','),geometry_workers=args.workers,
                        cpu_blas_threads=1,case_batch_size=8,platform=sys.platform,python=sys.version,
                        LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH'),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                        test_evidence_sha256=sha256(args.output/'GATE1A_RECOVERY_UNIT_INTEGRATION_TEST_REPORT.json'),
                        sampling_plan_reused_from_attempt1=True,localization_audit_sha256=sha256(args.localization),
                        localization_status=localization['localization_status'])
        write_json(args.output/'GATE1A_RUN_METADATA.json',metadata)
        write_json(args.output/'GATE1A_INPUT_AUDIT.json',{**input_audit,'metadata':metadata})
        _extraction_workers(args,metadata)
        entries=[read_json(p) for p in sorted((args.output/'feature_units').glob('*.json'))]
        validate_feature_manifest(args.output,entries,plan,metadata)
        norm_summary=norm_audit_reports(args.output,metadata,entries)
        metadata.update(feature_units_completed=len(entries),geometry_jobs_completed=0,clustering_jobs=0,
                        registered_norm_summary=norm_summary)
        write_json(args.output/'FEATURE_CACHE_MANIFEST.json',dict(metadata=metadata,status='PASS',entries=entries,raw_tensors_git_policy='excluded; remote only'))
        audits=[read_json(p) for p in sorted((args.output/'immutability').glob('*.json'))]
        require(len(audits)==18 and all(a['bitwise_unchanged'] for a in audits),'18 complete immutable checkpoint audits required')
        write_json(args.output/'GATE1A_MODEL_IMMUTABILITY_AUDIT.json',dict(metadata=metadata,status='PASS',checkpoints=audits))
        tasks=[]
        for panel in PANELS:
            for seed in range(3):
                for stage in range(3):
                    subset={r:next(e for e in entries if e['panel_id']==panel and e['seed']==seed and e['stage_index']==stage and e['role']==r) for r in ('train_labeled','val')}
                    units={r:next(u for u in plan['units'] if u['seed']==seed and u['stage_index']==stage and u['role']==r) for r in ('train_labeled','val')}
                    for c in range(3):
                        for K in (1,2,3,5):
                            tasks.append((str(args.output),prereg,metadata,subset,units,c,K))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            paths=list(pool.map(geometry_job,tasks,chunksize=1))
        results=[read_json(p) for p in paths]
        metadata.update(geometry_jobs_completed=len(results),clustering_jobs=len(results))
        report(args.output,metadata,results)
    except Exception as exception:
        error=exception
        write_text(args.output/'failure_traceback.txt',traceback.format_exc())
        entries=[read_json(p) for p in sorted((args.output/'feature_units').glob('*.json'))]
        if not (args.output/'GATE1A_REGISTERED_NORM_AUDIT.json').exists():
            summary=norm_audit_reports(args.output,metadata,entries)
        else:
            summary=read_json(args.output/'GATE1A_REGISTERED_NORM_AUDIT.json')
        metadata.update(feature_units_completed=len(entries),feature_units_expected=72,
                        geometry_jobs_completed=len(list((args.output/'geometry_units').glob('*.json'))),
                        clustering_jobs=len(list((args.output/'geometry_units').glob('*.json'))),
                        registered_norm_summary=summary)
        for name,folder in [('FEATURE_CACHE_MANIFEST.json','feature_units'),('GATE1A_MODEL_IMMUTABILITY_AUDIT.json','immutability')]:
            if not (args.output/name).exists():
                observed=[read_json(p) for p in sorted((args.output/folder).glob('*.json'))]
                write_json(args.output/name,dict(status='INCOMPLETE_BLOCKED',metadata=metadata,observed_partial_units=observed))
        blocked_report(args.output,metadata,exception)
    finally:
        write_text(args.output/'GATE1A_EXACT_COMMANDS.md','# Gate 1A exact execution\n\n```text\n'+str(ROOT)+'\n'+sys.executable+' '+' '.join(sys.argv)+'\n```\n\nGPU child shards call the same module with extract, the same output/data/code commit, shard indices 0..N-1 and --shards N. Each has CUDA_VISIBLE_DEVICES set to its listed physical GPU, BLAS/OpenMP threads=1, CUBLAS_WORKSPACE_CONFIG=:4096:8. CPU geometry is float64. Full shard metadata/logs are retained.\n')
        write_text(args.output/'GATE1A_FAILURES_AND_WARNINGS.md','# Gate 1A failures and warnings\n\n'+('No fatal error. Preserve complete pytest and shard transcripts; non-converged fits are reported, never excluded.\n' if error is None else f'{type(error).__name__}: {error}\n\nFull traceback and partial outputs retained. No thresholds/sampling/seed changes or downstream execution.\n'))
        artifact_manifest(args.output)
    if error is not None:
        raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','extract','localize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/root/LCRSeg'))
    parser.add_argument('--tests',type=Path)
    parser.add_argument('--localization',type=Path)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--planning-workers',type=int,default=8)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    if args.action=='run':
        _run(args)
    elif args.action=='localize':
        from .recovery import known_failure_localization
        audit=known_failure_localization(ROOT,args.data_root,args.output,args.code_commit)
        print(audit['localization_status'],flush=True)
        if not audit['attempt2_authorized']:
            raise SystemExit(2)
    else:
        try:
            extraction_shard(args)
        except Exception as error:
            write_json(args.output/'failures'/f'extract_{args.shard}.json',dict(status=getattr(error,'status','BLOCKED_NUMERICAL_FAILURE'),
                        error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                        provenance=getattr(error,'provenance',None)))
            request_stop(args.output,f'extract shard {args.shard} failed after checkpoint guard exit')
            raise


if __name__=='__main__':
    main()
