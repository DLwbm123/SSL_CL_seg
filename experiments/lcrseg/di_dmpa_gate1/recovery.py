"""Published scope clarification, byte-exact plan reuse and localization gate."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

from .binding import H, check_hash, read_json, require, sha256, verify_ancestor, write_json, write_text

CLARIFICATION = 'e8336da9d7364f4b67912d03791195445318afc3'
ATTEMPT1_REPORT = '945b484072cb9f2757be98df34e5d72844596e84'
PLAN_SHA = '96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24'
ATTEMPT1 = Path('/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1')
SHUTDOWN_TIMEOUT_SECONDS = 600


def verify_clarification(root, code_commit):
    root=Path(root); gitroot=root.parents[1]
    hashes={}
    for ancestor in (ATTEMPT1_REPORT, CLARIFICATION):
        verify_ancestor(gitroot, ancestor, code_commit)
    for suffix in ('md','json'):
        path=root/'docs/di_dmpa_jascl'/f'GATE1A_NUMERICAL_SCOPE_CLARIFICATION_V1.{suffix}'
        blob=subprocess.check_output(['git','-C',str(gitroot),'show',f'{CLARIFICATION}:{path.relative_to(gitroot)}'])
        hashes[suffix]=check_hash(path,hashlib.sha256(blob).hexdigest())
    return dict(clarification_git_commit=CLARIFICATION,clarification_file_sha256=hashes,
        recovery_diagnostic_code_git_commit=code_commit,post_failure_scope_clarification=True,
        original_attempt_status='BLOCKED_NUMERICAL_FAILURE',original_attempt_report_commit=ATTEMPT1_REPORT,
        original_sampling_plan_sha256=PLAN_SHA,full_map_zero_policy='DIAGNOSTIC_ONLY_UNLESS_REGISTERED',
        registered_zero_policy='BLOCKED_NUMERICAL_FAILURE_NO_DROP_NO_EPS_NO_RESAMPLE',
        gate1b_executed=False,gate1c_executed=False,shutdown_timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS)


def reuse_sampling_plan(source, output):
    """No label reads, generation API or partial-cache reuse in recovery."""
    source=Path(source); output=Path(output)
    check_hash(source,PLAN_SHA)
    destination=output/'SHARED_GEOMETRY_SAMPLING_PLAN.json'
    require(not destination.exists(),'refusing to replace sampling plan')
    output.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src,dst)
    check_hash(destination,PLAN_SHA)
    destination.chmod(0o444)
    plan=read_json(destination)
    require(len(plan['units'])==18,'expected 18 frozen seed/domain/role plan units')
    write_text(output/'SHARED_GEOMETRY_SAMPLING_PLAN.sha256',PLAN_SHA+'  SHARED_GEOMETRY_SAMPLING_PLAN.json\n')
    write_json(output/'SAMPLING_PLAN_AUDIT.json',dict(status='PASS',sampling_plan_reused_from_attempt1=True,
        source=str(source),raw_sha256=PLAN_SHA,mode=oct(destination.stat().st_mode & 0o777),
        label_arrays_read_for_plan=0,coordinates_rematerialized=False,unit_count=18))
    return plan,PLAN_SHA


def request_stop(output, reason):
    path=Path(output)/'STOP_REQUESTED.json'
    try:
        write_json(path,dict(reason=reason,policy='FINISH_CURRENT_CHECKPOINT_GUARD_THEN_STOP',
            shutdown_timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS))
    except FileExistsError:
        pass


def stop_requested(output):
    return (Path(output)/'STOP_REQUESTED.json').exists()


def known_failure_localization(root, data_root, output, code_commit):
    """Full registered-coordinate read-only audit; no formal geometry/cache."""
    from unittest.mock import patch
    import torch
    from .binding import ProtocolError, audit_inputs, run_metadata, verify_registration
    from .feature_extraction import ImmutabilityGuard, extract_unit, load_models
    from .registered_features import RegisteredFeatureNumericalError, summarize_cases
    output=Path(output)
    require(not output.exists(),'localization output exists; refusing overwrite')
    output.mkdir(parents=True)
    prereg,receipt=verify_registration(root,code_commit)
    inputs=audit_inputs(root,data_root,prereg)
    plan,digest=reuse_sampling_plan(ATTEMPT1/'SHARED_GEOMETRY_SAMPLING_PLAN.json',output)
    checkpoint=next(c for c in prereg['immutable_baseline']['checkpoint_inputs'] if c['checkpoint_id']=='B0/seed1/stage0')
    unit=next(u for u in plan['units'] if u['seed']==1 and u['stage_index']==0 and u['role']=='val')
    require(checkpoint['sha256']=='1e3c99ab3fe39de9755401a31779b5670c624064a73e772938ae57cbb2c3a1b8','wrong known failure checkpoint')
    require(unit['domain']==checkpoint['domain']=='REFUGE','wrong localization domain')
    metadata=run_metadata(prereg,receipt,digest,panel_id='B0-EMA')
    metadata.update(execution_scope='KNOWN_FAILURE_LOCALIZATION_ONLY',sampling_plan_reused_from_attempt1=True,
                    clustering_jobs=0,case_batch_size=8)
    write_json(output/'GATE1A_RUN_METADATA.json',metadata)
    write_json(output/'GATE1A_INPUT_AUDIT.json',inputs)
    context=dict(panel_id='B0-EMA',baseline='B0',feature_source='ema_teacher',seed=1,stage_index=0,
        domain='REFUGE',role='val',checkpoint_id=checkpoint['checkpoint_id'],checkpoint_sha256=checkpoint['sha256'],
        sampling_plan_sha256=digest)
    failure=None
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=ProtocolError('optimizer construction forbidden')):
        models,payload=load_models(root,checkpoint,device='cuda:0')
        require(payload['config_hash']==prereg['immutable_baseline']['configs']['B0']['resolved_config_sha256'],'config mismatch')
        try:
            with ImmutabilityGuard(models,checkpoint,output,metadata):
                arrays,details=extract_unit(models['ema_teacher'],unit,data_root,device='cuda:0',batch_size=8,
                                            context=context,collect_all_invalid=True)
                cases=details['cases']
                del arrays
        except RegisteredFeatureNumericalError as error:
            failure=error
            cases=error.case_audits
    summary=summarize_cases(cases)
    require([c['case_id'] for c in cases]==[c['case_id'] for c in unit['cases']],'incomplete localization cases')
    for c in range(3):
        expected=sum(len(case['classes'][c]['coordinates']) for case in unit['cases'])
        require(summary['registered_count_by_class'][str(c)]==expected,'incomplete registered coordinates')
    nonfinite=summary['full_map_nonfinite_count'] or sum(summary['registered_nonfinite_count_by_class'].values())
    zeros=sum(summary['registered_zero_count_by_class'].values())
    status=('BLOCKED_NUMERICAL_FAILURE' if nonfinite else 'BLOCKED_REGISTERED_ZERO_FEATURE' if zeros
            else 'PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED')
    require((failure is None)==(status=='PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED'),'numerical classification inconsistent')
    state=read_json(output/'immutability/B0_seed1_stage0.json')
    require(state['status']=='PASS' and state['bitwise_unchanged'],'state changed during localization')
    audit=dict(metadata=metadata,localization_status=status,**summary,cases=cases,
        complete_registered_coordinate_coverage=True,sampling_unit_sha256=H(unit),
        checkpoint_sha256_before=checkpoint['sha256'],checkpoint_sha256_after=sha256(checkpoint['path']),
        model_state_before=state['before'],model_state_after=state['after'],bitwise_unchanged=state['bitwise_unchanged'],
        model_optimizer_steps=0,transport_optimizer_steps=0,clustering_jobs=0,hidden_gt_training_usage='none',test_gt_usage='none',
        errors=[] if failure is None else getattr(failure,'all_failures',[failure.provenance]),
        attempt2_authorized=status=='PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED',
        next_action='FORMAL_ATTEMPT2_ALLOWED' if status.startswith('PASS_') else 'STOP_AWAIT_NEW_GATE1_V2_PREREGISTRATION')
    write_json(output/'GATE1A_KNOWN_FAILURE_LOCALIZATION_AUDIT.json',audit)
    return audit


def verify_localization(path, code_commit):
    audit=read_json(path)
    require(audit['localization_status']=='PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED','localization did not PASS; attempt2 forbidden')
    require(audit['metadata']['recovery_diagnostic_code_git_commit']==code_commit,'localization code mismatch')
    require(audit['metadata']['sampling_plan_sha256']==PLAN_SHA,'localization sampling mismatch')
    require(audit['complete_registered_coordinate_coverage'] and audit['bitwise_unchanged'],'localization incomplete or state changed')
    require(audit['full_map_nonfinite_count']==0 and not any(audit['registered_zero_count_by_class'].values())
            and not any(audit['registered_nonfinite_count_by_class'].values()),'localization invalid vectors')
    return audit


def norm_audit_reports(output, metadata, entries):
    from .registered_features import summarize_cases
    cases=[c for e in entries for c in e['diagnostics']['cases']]
    failures=[read_json(p) for p in sorted((Path(output)/'numerical_failures').glob('*.json'))]
    failed_cases=[c for failure in failures for c in failure['cases']]
    summary=summarize_cases(cases)
    value=dict(metadata=metadata,feature_units_complete=len(entries),feature_units_expected=72,
               status='COMPLETE' if len(entries)==72 else 'INCOMPLETE_BLOCKED',
               failed_unit_diagnostics=failures,failed_case_summary=summarize_cases(failed_cases),
               **summary,units=[dict(panel_id=e['panel_id'],seed=e['seed'],stage_index=e['stage_index'],role=e['role'],
               summary=e['diagnostics']['summary']) for e in entries])
    write_json(Path(output)/'GATE1A_REGISTERED_NORM_AUDIT.json',value)
    write_json(Path(output)/'GATE1A_FULL_MAP_ZERO_DIAGNOSTIC.json',value)
    return summary
