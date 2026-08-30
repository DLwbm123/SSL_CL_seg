"""One create-only attempt: complete census -> six fixed fits -> evaluator -> stop."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from di_dmpa_gate1.binding import audit_inputs
from di_dmpa_gate1.recovery import request_stop, stop_requested, SHUTDOWN_TIMEOUT_SECONDS
from .binding import (PREREG, DOMAINS, PLAN_SHA, ModelMutation, IncompleteEvidence, require,
    verify, sha256, check_hash, read_json, write_json, write_text, checkpoint)
from .plan import materialize
from .pairs import extract_transition, load_pair, census
from .transport import procrustes, fit_residual, feature_error, spectrum
from .evaluator import operational, evaluate_unit
from .reporting import complete, report, artifact_manifest

ROOT=Path(__file__).resolve().parents[1]


def disk_audit(p, frozen):
    rows={c['checkpoint_id']:check_hash(c['path'],c['sha256']) for c in p['immutable_baseline']['checkpoint_inputs'] if c['baseline']=='B0'}
    require(len(rows)==9, 'nine B0 checkpoints required')
    old=Path(frozen['formal_attempt_path'])
    for name,field in [('GATE1A_V2_ARTIFACT_MANIFEST.json','formal_artifact_manifest_sha256'),('GATE1A_V2_STATUS.json','status_sha256'),
        ('GATE1A_V2_FINAL_REPORT.md','raw_report_sha256')]:check_hash(old/name,frozen[field])
    check_hash(old/'SHARED_GEOMETRY_SAMPLING_PLAN.json',PLAN_SHA)
    for r in frozen['prototype_records']:check_hash(r['source_geometry_unit_remote_path'],r['source_file_sha256'])
    return rows


def model_audit(output, p, frozen, metadata, *, final=False):
    output=Path(output);expected={}
    for seed in range(3):
        for stage in (1,2):
            for side,index in [('source',stage-1),('target',stage)]:
                expected[f'paired_audits/seed{seed}_stage{stage}/{side}/immutability/B0_seed{seed}_stage{index}.json']=f'B0/seed{seed}/stage{index}'
        if final:
            for a,b in ((0,1),(1,2),(0,2)):
                expected[f'oracle_audits/seed{seed}_source{a}_target{b}/immutability/B0_seed{seed}_stage{b}.json']=f'B0/seed{seed}/stage{b}'
    observed={str(path.relative_to(output)) for folder in ('paired_audits','oracle_audits') for path in (output/folder).rglob('*.json')}
    complete(observed==set(expected), 'missing/extra model immutability guard')
    records=[]
    for rel,cpid in expected.items():
        r=read_json(output/rel)
        if not r['bitwise_unchanged'] or r['before']!=r['after'] or r['checkpoint_sha256_before']!=r['checkpoint_sha256_after']:
            raise ModelMutation(rel)
        require(r['checkpoint_id']==cpid and r['status']=='PASS' and r['extraction_completed'], 'incomplete checkpoint guard')
        require(all(r['metadata'][k]==v for k,v in metadata.items()), 'mixed immutability provenance')
        records.append(dict(path=rel,sha256=sha256(output/rel),audit=r))
    return dict(status='PASS',metadata=metadata,model_load_guards=len(records),guards=records,checkpoint_disk_sha256=disk_audit(p,frozen),
        all9_B0_disk_hashes_unchanged=True,all_model_states_unchanged=True,model_optimizer_steps=0,
        transport_optimizer_steps=6000 if final else 0)


def fit_job(task):
    output,seed,stage=task;output=Path(output)
    completed=0
    try:
        torch.set_num_threads(1)
        metadata=read_json(output/'GATE1B_V2_RUN_METADATA.json');barrier=read_json(output/'TRANSFORM_START_BARRIER.json')
        require(barrier['status']=='PASS' and barrier['transport_optimizer_steps']==0 and barrier['paired_units']==12, 'fit started before census barrier')
        for rel,digest in barrier['evidence_sha256'].items():check_hash(output/rel,digest)
        check_hash(output/'SHARED_TRANSPORT_COORDINATE_PLAN.json',metadata['transport_coordinate_plan_sha256'])
        plan=read_json(output/'SHARED_TRANSPORT_COORDINATE_PLAN.json');data={}
        for partition in ('fit','holdout'):
            entry=read_json(output/'paired_units'/f'seed{seed}_stage{stage}_{partition}.json')
            unit=next(u for u in plan['units'] if (u['seed'],u['stage_index'],u['partition'])==(seed,stage,partition))
            require(entry['metadata']==metadata, 'mixed fit provenance')
            data[partition]=load_pair(output,entry,unit)
        # Frozen sentinels check output validity only; no class/prototype target enters the objective.
        frozen=read_json(ROOT/'docs/di_dmpa_jascl/GATE1A_V2_FREEZE.json')
        prototypes=operational(frozen,seed,stage-1).reshape(-1,16)
        models={'T0':dict(kind='T0',optimizer_steps=0),'T1':procrustes(data['fit'])}
        trace_rel=f'transport_models/seed{seed}_stage{stage}_trace.csv'
        models['T2'],trace=fit_residual(data['fit'],prototypes,trace_path=output/trace_rel,stop_dir=output)
        completed=models['T2']['optimizer_steps']
        errors={partition:{method:feature_error(values,model) for method,model in models.items()} for partition,values in data.items()}
        row=dict(metadata=metadata,seed=seed,stage_index=stage,domain=DOMAINS[stage],role='train_unlabeled',models=models,
            feature_errors=errors,spectra={m:spectrum(model) for m,model in models.items()},trace_rows=len(trace),
            trace_path=trace_rel,trace_sha256=sha256(output/trace_rel),model_optimizer_steps=0,oracle_or_GT_access=False,
            category_agnostic_map=True,prototype_sentinels_role='validity_only_no_loss_no_target',all_finite=True)
        path=output/'transport_models'/f'seed{seed}_stage{stage}.json';write_json(path,row)
        print(f'transport complete seed{seed} stage{stage}:1000/1000; no result-based branching',flush=True)
        return str(path)
    except Exception as error:
        write_json(output/'failures'/f'fit_seed{seed}_stage{stage}.json',dict(status=getattr(error,'status','BLOCKED_INCOMPLETE_EVIDENCE'),
            error=str(error),traceback=traceback.format_exc(),seed=seed,stage_index=stage,
            transport_optimizer_steps=getattr(error,'transport_optimizer_steps_completed',completed),model_optimizer_steps=0))
        request_stop(output,'transport failure; no extra steps/retry');raise


def gpu_shard(args):
    p,_,frozen,_=verify(ROOT,args.code_commit,remote=False)
    meta=read_json(args.output/'GATE1B_V2_RUN_METADATA.json')
    require(meta['diagnostic_code_commit']==args.code_commit, 'mixed worker code')
    write_json(args.output/'shards'/f'{args.action}_{args.shard}_metadata.json',dict(meta,shard=args.shard,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    if args.action=='paired':
        check_hash(args.output/'SHARED_TRANSPORT_COORDINATE_PLAN.json',meta['transport_coordinate_plan_sha256'])
        plan=read_json(args.output/'SHARED_TRANSPORT_COORDINATE_PLAN.json')
        for seed,stage in ((s,t) for s in range(3) for t in (1,2)):
            # Alternate the unequal 63/41-case transitions across the two GPUs.
            if (seed+stage-1)%2!=args.shard:continue
            if stop_requested(args.output):return
            units=[u for u in plan['units'] if u['seed']==seed and u['stage_index']==stage]
            require([u['partition'] for u in units]==['fit','holdout'], 'wrong paired partitions')
            extract_transition(ROOT,args.data_root,p,units,args.output,meta)
    else:
        for index,(seed,a,b) in enumerate((s,a,b) for s in range(3) for a,b in ((0,1),(1,2),(0,2))):
            if index%2!=args.shard:continue
            if stop_requested(args.output):return
            evaluate_unit(ROOT,args.data_root,p,frozen,args.output,meta,seed,a,b)


def gpu_workers(args, action):
    processes=[];logs=[];deadline=None
    try:
        for shard,gpu in enumerate(('0','1')):
            command=[sys.executable,'-m','di_dmpa_gate1b_v2.runner',action,'--code-commit',args.code_commit,
                '--output',str(args.output),'--data-root',str(args.data_root),'--shard',str(shard)]
            log=(args.output/f'{action}_shard{shard}.txt').open('x');logs.append(log)
            processes.append(subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):request_stop(args.output,action+' shard failure')
            if stop_requested(args.output) and deadline is None:deadline=time.monotonic()+SHUTDOWN_TIMEOUT_SECONDS
            if deadline is not None and time.monotonic()>deadline:
                for p in processes:
                    if p.poll() is None:p.terminate()
                write_json(args.output/'FORCED_SHARD_TERMINATION.json',dict(action=action,timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS))
                break
            time.sleep(1)
        codes=[p.wait() for p in processes]
        if any(codes):
            failures=[read_json(p) for p in sorted((args.output/'failures').glob('*.json'))]
            error=IncompleteEvidence(f'{action} worker exits {codes}')
            error.status=failures[0]['status'] if failures else error.status
            raise error
    finally:
        for log in logs:log.close()


def actual_steps(output):
    steps={}
    for path in (Path(output)/'transport_models').glob('seed*_stage*_trace.csv'):
        stem=path.stem.split('_');key=(int(stem[0][4:]),int(stem[1][5:]))
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        steps[key]=max((int(r['step']) for r in rows if r.get('step','').isdigit()),default=0)
    for path in (Path(output)/'transport_models').glob('seed*_stage*.json'):
        r=read_json(path);steps[(r['seed'],r['stage_index'])]=r['models']['T2']['optimizer_steps']
    for path in (Path(output)/'failures').glob('fit_*.json'):
        r=read_json(path);steps[(r['seed'],r['stage_index'])]=max(steps.get((r['seed'],r['stage_index']),0),r['transport_optimizer_steps'])
    return sum(steps.values())


def run(args):
    require(not args.output.exists(), 'attempt already exists: no overwrite or automatic retry')
    args.output.mkdir(parents=True);metadata={};error=None;warnings=[]
    try:
        p,old,frozen,metadata=verify(ROOT,args.code_commit)
        require(args.output==Path('/root/LCRSeg/runs/di_dmpa_gate1b_v2')/PREREG/f'gate1b_v2_{args.code_commit}_attempt1', 'wrong attempt namespace')
        require(str(ROOT)==p['runtime']['root'] and str(args.data_root)==p['runtime']['data_root'], 'wrong execution/data root')
        for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):require(os.environ.get(variable)=='1','single-thread BLAS required')
        tests=read_json(args.tests/'GATE1B_V2_UNIT_INTEGRATION_TEST_REPORT.json')
        require(tests['status']=='PASS' and tests['diagnostic_code_commit']==args.code_commit and tests['failures']==tests['skipped']==0 and tests['passed']>=44, 'tests missing/incomplete/not exact code')
        for name,digest in tests['artifact_sha256'].items():check_hash(args.tests/name,digest)
        integration=read_json(args.tests/'GATE1B_V2_REAL_INTEGRATION.json')
        require(integration['status']=='PASS' and integration['model_optimizer_steps']==integration['transport_optimizer_steps']==0 and
            integration['metadata']['diagnostic_code_commit']==args.code_commit and integration['model_unchanged'] and integration['all_registered_rows_retained'], 'read-only integration failed')
        require([(u['seed'],u['stage_index'],u['case_id'],u['registered_count']) for u in integration['units']]==
            [(c['seed'],c['stage_index'],c['case_id'],2048) for c in p['integration']['selected_cases']], 'integration case coverage changed')
        for name in ('GATE1B_V2_UNIT_INTEGRATION_TEST_REPORT.json','pytest.xml','pytest_output.txt','GATE1B_V2_REAL_INTEGRATION.json'):
            shutil.copy2(args.tests/name,args.output/name)
        inputs=audit_inputs(ROOT,args.data_root,old)
        write_json(args.output/'GATE1B_V2_INPUT_AUDIT.json',dict(metadata=metadata,legacy_byte_only_audit=inputs,B0_disk_hashes=disk_audit(p,frozen),
            legacy_role_counters_scope='this byte-only audit constructs no role objects; separate Gate1B adapter constructs current train_unlabeled'))
        plan,digest=materialize(args.data_root,p,args.output,metadata)
        for test in integration['units']:
            u=next(u for u in plan['units'] if u['seed']==test['seed'] and u['stage_index']==test['stage_index'] and u['partition']=='fit')
            case=next(c for c in u['cases'] if c['case_id']==test['case_id'])
            require(case['coordinate_uid_sha256']==test['coordinate_uid_hash'], 'integration/full-plan coordinate mismatch')
        metadata=dict(metadata,transport_coordinate_plan_sha256=digest,execution_scope='GATE1B_V2_ONLY',
            extraction_gpus=[0,1],coordinate_workers=16,transport_workers=6,transport_device='cpu_float64',batch_size=8)
        write_json(args.output/'GATE1B_V2_RUN_METADATA.json',metadata)
        gpu_workers(args,'paired')
        entries=[read_json(path) for path in sorted((args.output/'paired_units').glob('*.json'))]
        paired_audit=model_audit(args.output,p,frozen,metadata)
        write_json(args.output/'PAIRED_MODEL_IMMUTABILITY_AUDIT.json',paired_audit)
        write_json(args.output/'PAIRED_FEATURE_CACHE_MANIFEST.json',dict(status='PASS',metadata=metadata,paired_units=entries,raw_tensors_git_policy='remote_only'))
        counts=census(args.output,entries,plan,metadata)
        barrier_files=['PAIRED_FEATURE_SUPPORT_CENSUS.json','PAIRED_FEATURE_CACHE_MANIFEST.json','PAIRED_MODEL_IMMUTABILITY_AUDIT.json',
            'TRANSPORT_SPLIT_AND_COORDINATE_AUDIT.json','GATE1B_V2_INPUT_AUDIT.json']
        write_json(args.output/'TRANSFORM_START_BARRIER.json',dict(status='PASS',metadata=metadata,paired_units=12,model_guards=12,
            all9_B0_disk_hashes_unchanged=True,transport_optimizer_steps=0,model_optimizer_steps=0,
            evidence_sha256={name:sha256(args.output/name) for name in barrier_files}))
        tasks=[(str(args.output),seed,stage) for seed in range(3) for stage in (1,2)]
        with ProcessPoolExecutor(max_workers=6) as pool:paths=list(pool.map(fit_job,tasks,chunksize=1))
        complete(len(paths)==6 and actual_steps(args.output)==6000, 'six fits/6000 updates required')
        write_json(args.output/'ORACLE_START_BARRIER.json',dict(status='PASS',metadata=metadata,six_transports_complete=True,
            transport_optimizer_steps=6000,model_optimizer_steps=0,model_sha256={Path(path).name:sha256(path) for path in paths}))
        source=Path(frozen['formal_attempt_path'])/'SHARED_GEOMETRY_SAMPLING_PLAN.json'
        check_hash(source,PLAN_SHA);shutil.copy2(source,args.output/'FROZEN_GEOMETRY_SAMPLING_PLAN.json')
        gpu_workers(args,'oracle')
        write_json(args.output/'GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json',model_audit(args.output,p,frozen,metadata,final=True))
        transports=[read_json(path) for path in paths];oracles=[read_json(path) for path in sorted((args.output/'oracle_units').glob('*.json'))]
        warnings=report(args.output,metadata,transports,oracles,counts)
    except Exception as exc:
        error=exc;status=getattr(exc,'status','BLOCKED_INCOMPLETE_EVIDENCE')
        write_text(args.output/'failure_traceback.txt',traceback.format_exc())
        if not (args.output/'GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json').exists():
            guards=[dict(path=str(path.relative_to(args.output)),audit=read_json(path),sha256=sha256(path))
                for folder in ('paired_audits','oracle_audits') for path in sorted((args.output/folder).rglob('*.json'))]
            write_json(args.output/'GATE1B_V2_MODEL_IMMUTABILITY_AUDIT.json',dict(metadata=metadata,guards=guards,
                scope='observed guards at stopped attempt; not a complete final audit',
                status='BLOCKED_MODEL_MUTATION' if any(not g['audit']['bitwise_unchanged'] for g in guards) else 'PARTIAL_EVIDENCE',
                all_observed_model_states_unchanged=all(g['audit']['bitwise_unchanged'] for g in guards),model_optimizer_steps=0))
        if not (args.output/'GATE1B_V2_STATUS.json').exists():
            write_json(args.output/'GATE1B_V2_STATUS.json',dict(metadata,transport_status=status,selected_K=2,
                selected_transport='T0_identity' if status=='FAIL_DIRECTIONAL_PAIR_SUPPORT_NOT_SUPPORTED' else None,
                transport_optimizer_steps=actual_steps(args.output),optimizer_step_accounting='persisted update counter / trace lower bound if a worker exited without failure receipt',model_optimizer_steps=0,
                paired_units_completed=len(list((args.output/'paired_units').glob('*.json'))),
                transports_completed=len(list((args.output/'transport_models').glob('seed*_stage*.json'))),
                oracle_units_completed=len(list((args.output/'oracle_units').glob('*.json'))),B1_B7_computed=False,
                hidden_gt_training_usage='none',test_gt_usage='none',method_registered=False,di_dmpa_training_launched=False,Gate1C=False,
                error=str(exc),next_action='STOP_FOR_INDEPENDENT_REVIEW'))
        if not (args.output/'GATE1B_V2_FINAL_REPORT.md').exists():
            write_text(args.output/'GATE1B_V2_FINAL_REPORT.md',f'# Gate 1B v2 stopped\n\n{status}: {exc}\n\nNo admission PASS; preserve all partial evidence, no retry or downstream execution.\n')
    finally:
        write_text(args.output/'GATE1B_V2_EXACT_COMMANDS.md','# Exact command\n\n```text\n'+str(ROOT)+'\n'+sys.executable+' '+' '.join(sys.argv)+'\n```\n\nEnvironment: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8; existing environment; no installs.\n')
        write_text(args.output/'GATE1B_V2_FAILURES_AND_WARNINGS.md','# Failures and warnings\n\n'+('No fatal execution error.\n' if error is None else str(error)+'\n')+
            f'Oracle warnings retained: {len(warnings)}; full records in diagnostic JSON. No rerun, restart extension, map substitution or downstream work.\n')
        artifact_manifest(args.output)
    if error is not None:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','paired','oracle']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-commit',required=True);parser.add_argument('--data-root',type=Path,default=Path('/root/LCRSeg'))
    parser.add_argument('--tests',type=Path);parser.add_argument('--shard',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.action=='run':run(args)
    else:
        try:gpu_shard(args)
        except Exception as error:
            write_json(args.output/'failures'/f'{args.action}_shard{args.shard}.json',dict(status=getattr(error,'status','BLOCKED_INCOMPLETE_EVIDENCE'),error=str(error),traceback=traceback.format_exc()))
            request_stop(args.output,'GPU failure after model guard exit');raise


if __name__=='__main__':main()
