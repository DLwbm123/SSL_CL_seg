"""Canonical-layout production E1/E2/E3 rehearsals; all 198 rows are synthetic."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import pytest
from care_hr_v0_7_1 import execute_r1 as old, execute_r2 as e, recovery_r2 as r
from care_hr_v0_7_1.blind_r1 import prepare_case
from care_hr_v0_7_1.io_r1 import digest, write_json, verify_files
from care_hr_v0_7_1.scoring_r1 import case_metrics
from care_hr_v0_7_1.summary_r1 import DOMAINS, legacy_aggregate
from test_layout_r2 import put


def configure(base, settings):
    # Test-only roots and external identities, never replaced loader/seal/science functions.
    for mod in (r,e,old):mod.DATA=base/'data';mod.FORMAL=base/'formal'
    r.PARENT=base/'parent';r.PARENT_HASH=settings['parent_hash'];r.OBSERVED=settings['observed']
    r.DOC=base/'docs2';r.R1DOC=base/'docs1';e.DOC=base/'docs1'
    r.source_state=e.source_state=lambda:'synthetic_evaluator'
    r.original_inputs=e.original_inputs=lambda:(settings['expected'],settings['entries'])
    e.REFERENCE=settings['references']


@pytest.fixture
def deployment(tmp_path,monkeypatch):
    # Save external constants; production functions remain unchanged throughout the test.
    for mod,names in ((r,('DATA','FORMAL','PARENT','PARENT_HASH','OBSERVED','DOC','R1DOC','source_state','original_inputs')),
                      (e,('DATA','FORMAL','DOC','source_state','original_inputs','REFERENCE')),(old,('DATA','FORMAL'))):
        for name in names:monkeypatch.setattr(mod,name,getattr(mod,name))
    data=tmp_path/'data';formal=tmp_path/'formal';parent=tmp_path/'parent';d1=tmp_path/'docs1';d2=tmp_path/'docs2'
    for path in (data,formal,parent,d1,d2):path.mkdir()
    (data/'manifests/training').mkdir(parents=True);(parent/'cases').mkdir();(parent/'public').mkdir()
    for name in ('R1_PREREGISTRATION.json','SCORING_CONTRACT_R1.json'):(d1/name).write_bytes((r.R1DOC/name).read_bytes())
    (d2/'R2_RECOVERY_PREREGISTRATION.json').write_bytes((r.DOC/'R2_RECOVERY_PREREGISTRATION.json').read_bytes())
    rows=[];originals=[];manifests={s:[] for s in range(3)};diagnostics=[];predictions={};routes=np.where(np.arange(198)<155,0,2).astype(np.int64)
    p=np.zeros((3,384,384),dtype=np.float32);p[1]=1;h=p.copy();h[1,0,:8]=0;h[0,0,:8]=1
    for i in range(198):
        seed=i//66;case=str(i%177);patient='p'+case;fold=int(case)%5;d=(int(case)%66)//22
        blind=dict(seed=seed,case_id=case,patient_id=patient,fold=fold,row_index=i,alpha=[.1,.2,.7],image_h5_relpath='images/'+case,image_sha256='h'+case)
        rows.append(blind);rel='labels/'+case+'.h5';path=data/'h5/v1'/rel
        if not path.exists():put(path,255 if case=='0' else 1)
        manifests[seed].append(dict(case_id=case,patient_id=patient,primary_20pct_split='train_labeled',dataset='fundus',site_or_vendor=DOMAINS[d],
            label_h5_relpath=rel,label_sha256=digest(path),image_h5_relpath=blind['image_h5_relpath'],image_sha256=blind['image_sha256']))
        prepared=prepare_case(p,h if routes[i]==0 else p,int(routes[i]))
        arrays={key:prepared.pop(key) for key in ('current_hard','blended_hard','masks','probability_changed')}
        np.savez_compressed(parent/'cases'/f'{i:03d}.npz',**arrays);write_json(parent/'cases'/f'{i:03d}.json',prepared)
        acts=prepared['actions'];diagnostics.append(dict(row_index=i,proposals=len(prepared['proposals']),strict_actions=sum(a['O_CAP'] for a in acts),no_area_actions=sum(a['O_NO_AREA'] for a in acts),free_actions=len(acts),quantity_rejected=0,foreground_area_rejected=0,image_area_rejected=0,zero_current_foreground=False))
        truth=np.full((384,384),255 if case=='0' else 1,dtype=np.uint8)
        for policy in (*e.BASE_POLICIES,'C7'):
            pred=(h if policy=='C6' and routes[i]==0 else p).argmax(axis=0).astype(np.uint8)
            predictions[i,policy]=pred
            originals.append(dict(seed=seed,case_id=case,patient_id=patient,domain=DOMAINS[d],domain_index=d,policy=policy,**case_metrics(pred,truth)))
    hashes={}
    for seed,records in manifests.items():
        records.append(dict(records[-1],case_id='forbidden',primary_20pct_split='val',site_or_vendor='FORBIDDEN',label_h5_relpath='/must_not_open'))
        path=data/'manifests/training'/f'lcrseg_v1_seed{seed}.csv';old.write_csv(path,records);hashes[str(seed)]=digest(path)
    (formal/'case_metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in originals))
    folders={};controls={}
    for fold in range(5):
        ids=[i for i,row in enumerate(rows) if row['fold']==fold];folders[str(fold)]=ids
        for policy in e.BASE_POLICIES:
            name=f'fold{fold}_{policy}.npy';np.save(formal/name,np.stack([predictions[i,policy] for i in ids]));controls[f'{fold}:{policy}']=name
    (formal/'expert_probability_cache').mkdir();probs={}
    for seed in range(3):
        for expert in range(3):
            stem=f'seed{seed}_expert{expert}';path=formal/'expert_probability_cache'/f'{stem}.npy';np.save(path,np.asarray([seed,expert]));probs[stem]=digest(path)
    entries={str(path.relative_to(formal)):dict(bytes=path.stat().st_size,sha256=digest(path)) for path in formal.rglob('*') if path.is_file()}
    write_json(parent/'blind_rows.json',rows);np.save(parent/'routes.npy',routes);write_json(parent/'blind_diagnostics.json',diagnostics)
    write_json(parent/'input_metadata.json',dict(manifest_hashes=hashes,control_paths=controls,fold_indices=folders))
    files={str(path.relative_to(parent)):digest(path) for path in parent.rglob('*') if path.is_file()}
    originals_map={name:v['sha256'] for name,v in entries.items() if name!='case_metrics.jsonl'}
    write_json(parent/'ACTION_SPACE_SEAL.json',dict(source_commit=r.ACTION_SOURCE,files=files,original_files=originals_map))
    for name,value in [('EVALUATOR_ACCESS_RESERVATION.json',{'synthetic_R1':True}),('public/CAPACITY_STATUS.json',{'status':'INCOMPLETE_EVALUATION'}),('public/RUNTIME_COUNTERS.json',{'GT_reads':1,'domain_reads':1})]:
        write_json(parent/name,value);(d1/Path(name).name).write_bytes((parent/name).read_bytes())
    write_json(tmp_path/'EVALUATOR_PARENT.json',{'actual_child_exit_code':1})
    write_json(d1/'PARENT_EXIT_RECEIPTS.json',{'EVALUATOR_PARENT.json':{'private_receipt_sha256':digest(tmp_path/'EVALUATOR_PARENT.json')}})
    observed={k:sum(d[k] for d in diagnostics) for k in ('proposals','strict_actions','no_area_actions','free_actions')}
    observed.update(max_strict=3,max_free=3,strict_noop_only=43,free_noop_only=43,zero_foreground=0)
    refs={e.BASE_POLICIES[a['policy']]:a['foreground_dice'] for a in legacy_aggregate(originals) if a['level']=='overall' and a['policy'] in e.BASE_POLICIES}
    settings=dict(parent_hash=digest(parent/'ACTION_SPACE_SEAL.json'),observed=observed,expected={'C6_route':{'route_sha256':hashlib.sha256(routes.tobytes()).hexdigest()},'probability_cache_sha256':probs},entries=entries,references=refs)
    write_json(tmp_path/'settings.json',settings);configure(tmp_path,settings)
    qualification=tmp_path/'A1.json';write_json(qualification,dict(source_commit='synthetic_evaluator',A1_complete=True,deployment_layout_passed=True))
    return tmp_path,qualification,{str(path.relative_to(parent)):digest(path) for path in parent.rglob('*') if path.is_file()}


def admitted(deployment):
    base,qualification,unchanged=deployment;out=base/'r2_run'
    r.qualify(out,qualification)
    write_json(out/'SEAL_PUBLICATION.json',dict(seal_sha256=digest(out/'ACTION_SPACE_SEAL.json'),remote_verified=True,anonymous_verified=True))
    r.preflight(out)
    pre=json.load(open(out/'public/ASSET_BINDING_PREFLIGHT.json'))
    assert pre['rows_checked']==pre['rows_admitted']==198 and pre['unique_physical_assets']==177
    assert pre['GT_payload_open_attempts']==pre['true_domain_records_materialized']==0
    return base,out,unchanged


def test_production_recovery_e2e_success_preserves_parent_and_counts(deployment):
    base,out,unchanged=admitted(deployment);e.evaluate(out)
    status=json.load(open(out/'public/CAPACITY_STATUS.json'));count=json.load(open(out/'public/RUNTIME_COUNTERS.json'))
    assert status['status']=='FAIL_FROZEN_ACTION_SPACE_CAPACITY'
    for name in ('GT_payload_open_attempts','GT_payload_open_successes','GT_payload_reads_completed','GT_file_hash_checks_passed','GT_label_decode_attempts','GT_label_decodes_completed','GT_rows_bound_to_validated_labels','true_domain_records_materialized','true_domain_rows_processed','oracle_case_rows_completed'):assert count[name]==198
    assert count['GT_unique_files_decoded']==177
    assert count['baseline_case_policy_comparisons_completed']==792 and count['baseline_scalar_comparisons_completed']==3168
    assert all(v==0 for v in count['real_fits_by_kind'].values()) and count['new_sample_expert_forwards']==0
    assert (out/'public/CAPACITY_METRICS.csv').is_file() and (out/'private_evaluation/pareto.jsonl').is_file()
    verify_files(base/'parent',unchanged)
    with pytest.raises(FileExistsError):e.evaluate(out)
    with pytest.raises(ValueError,match='disjoint'):r.separate_output(base/'parent/new')


def test_production_fault_parent_exit_and_no_baseline_oracle(deployment):
    base,out,unchanged=admitted(deployment)
    # First actual file changes after metadata-only admission: loader must fail its hash gate.
    (base/'data/h5/v1/labels/0.h5').write_bytes(b'corrupt after preflight')
    script=base/'child.py'
    testdir=str(Path(__file__).resolve().parent)
    script.write_text(f'import sys,json\nfrom pathlib import Path\nsys.path.insert(0,{testdir!r})\nfrom test_recovery_r2 import configure,e\nbase=Path({str(base)!r})\nconfigure(base,json.load(open(base/"settings.json")))\ne.evaluate(base/"r2_run")\n')
    result=subprocess.run([sys.executable,'-m','care_hr_v0_7_1.parent_r1','--receipt',str(base/'R2_PARENT.json'),'--log',str(base/'R2_PRIVATE.log'),'--',sys.executable,str(script)],capture_output=True,text=True)
    assert result.returncode==1
    assert json.load(open(base/'R2_PARENT.json'))['actual_child_exit_code']==1
    assert json.load(open(out/'public/CAPACITY_STATUS.json'))['status']=='INCOMPLETE_EVALUATION'
    count=json.load(open(out/'public/RUNTIME_COUNTERS.json'))
    assert count['GT_payload_open_attempts']==count['GT_payload_open_successes']==count['GT_payload_reads_completed']==1
    assert count['GT_file_hash_checks_passed']==count['GT_label_decodes_completed']==count['baseline_case_policy_comparisons_completed']==count['oracle_case_rows_completed']==0
    assert count['true_domain_records_materialized']==66 and count['true_domain_rows_processed']==1
    assert not (out/'public/BASELINE_PARITY.json').exists()
    verify_files(base/'parent',unchanged)


def test_preflight_aggregates_all_missing_and_never_opens_payload(deployment,monkeypatch):
    base,qualification,unchanged=deployment;out=base/'r2_run';r.qualify(out,qualification)
    write_json(out/'SEAL_PUBLICATION.json',dict(seal_sha256=digest(out/'ACTION_SPACE_SEAL.json'),remote_verified=True,anonymous_verified=True))
    (base/'data/h5/v1/labels/0.h5').unlink();(base/'data/h5/v1/labels/1.h5').unlink()
    original=Path.open
    def guarded(path,*args,**kwargs):
        if 'h5/v1' in str(path):raise AssertionError('stat-only preflight opened GT')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(ValueError,match='PREFLIGHT'):r.preflight(out)
    result=json.load(open(out/'public/ASSET_BINDING_PREFLIGHT.json'))
    assert result['rows_checked']==198 and result['errors']==4 and result['GT_payload_reads']==0
    assert not (out/'ASSET_BINDING_ADMISSION.json').exists()
