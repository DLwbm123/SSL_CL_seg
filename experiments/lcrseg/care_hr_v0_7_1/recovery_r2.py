"""E1 parent-content reuse and E2 metadata-only admission. Never reads GT bytes."""
from pathlib import Path
import hashlib
import time
import numpy as np

from .execute_r1 import ROOT, DATA, FORMAL, source_state, require, original_inputs
from .io_r1 import digest, read_json, write_json, verify_files, project_training, validate_population
from .io_r2 import asset_preflight, counters

DOC=ROOT/'experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r2'
R1DOC=DOC.parent/'continuation_r1'
PARENT=Path('/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/care_hr_v0_7_1_r1_20260907_01/run_01')
ACTION_SOURCE='9bbbacd25f3c3abf885205009eb14d698b36f32b'
PARENT_HASH='9b186f8c4e80236987700fcb74cf90a5b41071b63df128a5b1e28142df7141a4'
OBSERVED=dict(proposals=949,strict_actions=15796,no_area_actions=26374,free_actions=159232,
              max_strict=597,max_free=8191,strict_noop_only=67,free_noop_only=43,zero_foreground=0)


def separate_output(output):
    parent=PARENT.resolve();output=Path(output).resolve()
    require(output!=parent and not output.is_relative_to(parent) and not parent.is_relative_to(output),'R2 writes must be disjoint from R1 root')


def kernels():
    p=read_json(DOC/'R2_RECOVERY_PREREGISTRATION.json')
    verify_files(ROOT,p['scientific_kernel_hashes'])
    require(digest(R1DOC/'SCORING_CONTRACT_R1.json')==p['scoring_contract_sha256'],'scoring contract changed')
    require(digest(R1DOC/'R1_PREREGISTRATION.json')==p['original_preregistration_sha256'],'R1 protocol changed')
    return p


def verify_parent():
    require(digest(PARENT/'ACTION_SPACE_SEAL.json')==PARENT_HASH,'BLOCKED_PARENT_SEAL_OR_INPUT_MISMATCH')
    seal=read_json(PARENT/'ACTION_SPACE_SEAL.json')
    require(seal['source_commit']==ACTION_SOURCE,'parent action-generation source mismatch')
    verify_files(PARENT,seal['files']);verify_files(FORMAL,seal['original_files'])
    return seal


def parent_state_hashes():
    names=('EVALUATOR_ACCESS_RESERVATION.json','public/CAPACITY_STATUS.json','public/RUNTIME_COUNTERS.json')
    files={name:digest(PARENT/name) for name in names}
    require(files['EVALUATOR_ACCESS_RESERVATION.json']==digest(R1DOC/'EVALUATOR_ACCESS_RESERVATION.json'),'R1 reservation changed')
    require(files['public/CAPACITY_STATUS.json']==digest(R1DOC/'CAPACITY_STATUS.json'),'R1 terminal changed')
    require(files['public/RUNTIME_COUNTERS.json']==digest(R1DOC/'RUNTIME_COUNTERS.json'),'R1 counters changed')
    receipt=PARENT.parent/'EVALUATOR_PARENT.json'
    expected=read_json(R1DOC/'PARENT_EXIT_RECEIPTS.json')['EVALUATOR_PARENT.json']['private_receipt_sha256']
    require(digest(receipt)==expected and read_json(receipt)['actual_child_exit_code']==1,'R1 parent receipt mismatch')
    return files


def qualify(output, evidence_path):
    separate_output(output);source=source_state();p=kernels()
    evidence=read_json(evidence_path)
    require(evidence['source_commit']==source and evidence['A1_complete'] and evidence['deployment_layout_passed'],'R2 exact-source engineering qualification missing')
    output.mkdir(parents=True,exist_ok=False);(output/'public').mkdir()
    seal=verify_parent();state_hashes=parent_state_hashes();expected,_=original_inputs()
    rows=read_json(PARENT/'blind_rows.json');meta=read_json(PARENT/'input_metadata.json')
    projected={}
    for s in range(3):
        path=DATA/'manifests/training'/f'lcrseg_v1_seed{s}.csv'
        require(digest(path)==meta['manifest_hashes'][str(s)],'manifest identity changed')
        projected[s]=project_training(path)
    population=validate_population(rows,projected)
    route=np.load(PARENT/'routes.npy',allow_pickle=False)
    require(route.dtype==np.int64 and route.shape==(198,) and hashlib.sha256(route.tobytes()).hexdigest()==expected['C6_route']['route_sha256'],'C6 route identity mismatch')
    require(int(np.sum(route<2))==155 and int(np.sum(route==2))==43,'C6 route count mismatch')
    probability_names=[name for name in seal['original_files'] if name.startswith('expert_probability_cache/')]
    require(len(probability_names)==9,'nine parent probability files required')
    for name in probability_names:
        require(seal['original_files'][name]==expected['probability_cache_sha256'][Path(name).stem],'parent probability lineage mismatch')
    diagnostics=read_json(PARENT/'blind_diagnostics.json')
    observed={k:sum(r[k] for r in diagnostics) for k in ('proposals','strict_actions','no_area_actions','free_actions')}
    observed.update(max_strict=max(r['strict_actions'] for r in diagnostics),max_free=max(r['free_actions'] for r in diagnostics),
      strict_noop_only=sum(r['strict_actions']==1 for r in diagnostics),free_noop_only=sum(r['free_actions']==1 for r in diagnostics),zero_foreground=sum(r['zero_current_foreground'] for r in diagnostics))
    require(observed==OBSERVED and len(diagnostics)==198,'parent enumeration observations mismatch')
    for i,d in enumerate(diagnostics):
        prepared=read_json(PARENT/'cases'/f'{i:03d}.json');a=prepared['actions']
        require(d['row_index']==i and prepared['route']==int(route[i]),'prepared row route mismatch')
        require(len(prepared['proposals'])==d['proposals'] and len(a)==d['free_actions'] and sum(r['O_CAP'] for r in a)==d['strict_actions'] and sum(r['O_NO_AREA'] for r in a)==d['no_area_actions'],'sealed action count mismatch')
    write_json(output/'QUALIFICATION.json',evidence)
    wrapper=dict(schema_version='care_hr_v0_7_1_R2_dual_source_v1',action_generation_source_commit=ACTION_SOURCE,
      evaluator_source_commit=source,parent_action_space_seal_sha256=PARENT_HASH,parent_action_content_manifest=seal['files'],
      original_files=seal['original_files'],parent_terminal_files=state_hashes,new_protocol_sha256=digest(DOC/'R2_RECOVERY_PREREGISTRATION.json'),
      scoring_contract_sha256=p['scoring_contract_sha256'],scientific_kernel_hashes=p['scientific_kernel_hashes'],
      new_A1_and_layout_qualification_sha256=digest(output/'QUALIFICATION.json'),prior_exposure=p['prior_exposure'],
      sealed_input_root=str(PARENT),run_output_root=str(output),new_model_forwards=0,new_real_fits=0)
    write_json(output/'ACTION_SPACE_SEAL.json',wrapper)
    write_json(output/'public'/'ACTION_SPACE_REUSE_QUALIFICATION.json',dict(status='PASS_PARENT_CONTENT_REUSED_UNCHANGED',
      population=population,observations=observed,probability_cache_files=9,C6_historical=155,C6_current=43,
      parent_seal_sha256=PARENT_HASH,R2_seal_sha256=digest(output/'ACTION_SPACE_SEAL.json'),
      action_generation_source_commit=ACTION_SOURCE,evaluator_source_commit=source,
      action_regeneration=False,parent_files_written=0,GT_payload_reads=0,new_domain_records_materialized=0))


def verify_recovery(output):
    separate_output(output);source=source_state();p=kernels();s=read_json(output/'ACTION_SPACE_SEAL.json')
    require(s['schema_version']=='care_hr_v0_7_1_R2_dual_source_v1' and s['evaluator_source_commit']==source and s['action_generation_source_commit']==ACTION_SOURCE,'dual-source identity mismatch')
    require(s['parent_action_space_seal_sha256']==PARENT_HASH and s['sealed_input_root']==str(PARENT) and s['run_output_root']==str(output),'dual-source root/parent mismatch')
    require(s['new_protocol_sha256']==digest(DOC/'R2_RECOVERY_PREREGISTRATION.json') and s['scientific_kernel_hashes']==p['scientific_kernel_hashes'] and s['scoring_contract_sha256']==p['scoring_contract_sha256'],'dual-source protocol mismatch')
    require(s['new_A1_and_layout_qualification_sha256']==digest(output/'QUALIFICATION.json'),'R2 qualification changed')
    parent=verify_parent()
    require(s['parent_action_content_manifest']==parent['files'] and s['original_files']==parent['original_files'],'parent content map changed')
    verify_files(PARENT,s['parent_terminal_files'])
    meta=read_json(PARENT/'input_metadata.json')
    for seed,h in meta['manifest_hashes'].items():require(digest(DATA/'manifests/training'/f'lcrseg_v1_seed{seed}.csv')==h,'seed manifest changed')
    return s


def preflight(output):
    seal=verify_recovery(output)
    release=read_json(output/'SEAL_PUBLICATION.json')
    require(release['seal_sha256']==digest(output/'ACTION_SPACE_SEAL.json') and release['remote_verified'] and release['anonymous_verified'],'R2 seal must be published before loader metadata')
    count=counters();start=time.monotonic()
    rows=read_json(PARENT/'blind_rows.json')
    allowlist,errors=asset_preflight(DATA,rows,count)
    write_json(output/'RESOLVED_ASSET_ALLOWLIST.json',allowlist)
    if errors:write_json(output/'PRIVATE_PREFLIGHT_ERRORS.json',errors)
    status='PASS_ALL_198_STAT_ONLY' if not errors and len(allowlist)==198 else 'BLOCKED_ASSET_BINDING_PREFLIGHT'
    report=dict(status=status,rows_checked=count['asset_metadata_rows_checked'],rows_admitted=len(allowlist),
      unique_physical_assets=count['asset_unique_files_stat_checked'],duplicate_row_uses=len(allowlist)-count['asset_unique_files_stat_checked'],
      patients=len({r['patient_id'] for r in allowlist}),errors=len(errors),GT_payload_open_attempts=0,GT_payload_reads=0,
      true_domain_records_materialized=0,allowlist_sha256=digest(output/'RESOLVED_ASSET_ALLOWLIST.json'),elapsed_seconds=time.monotonic()-start)
    write_json(output/'public'/'ASSET_BINDING_PREFLIGHT.json',report)
    require(status=='PASS_ALL_198_STAT_ONLY','BLOCKED_ASSET_BINDING_PREFLIGHT')
    write_json(output/'ASSET_BINDING_ADMISSION.json',dict(status=status,R2_seal_sha256=digest(output/'ACTION_SPACE_SEAL.json'),
      allowlist_sha256=report['allowlist_sha256'],preflight_sha256=digest(output/'public'/'ASSET_BINDING_PREFLIGHT.json'),source_commit=seal['evaluator_source_commit']))
