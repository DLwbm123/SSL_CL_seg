"""Full 198-row executor rehearsal using generated HDF5 only, never real data."""
import hashlib
import json
import numpy as np
import pytest
from care_hr_v0_7_1 import execute_r1 as e
from care_hr_v0_7_1.blind_r1 import prepare_case
from care_hr_v0_7_1.io_r1 import digest, write_json
from care_hr_v0_7_1.scoring_r1 import case_metrics
from care_hr_v0_7_1.summary_r1 import DOMAINS


@pytest.mark.parametrize('baseline_mismatch',(False,True))
def test_full_evaluator_atomic_gate_reports_and_baseline_stop(tmp_path,monkeypatch,baseline_mismatch):
    import h5py
    out=tmp_path/'run';out.mkdir(); formal=tmp_path/'formal';formal.mkdir();data=tmp_path/'data';data.mkdir()
    original_prereg=json.load(open(e.DOC/'R1_PREREGISTRATION.json'))
    docs=tmp_path/'docs';docs.mkdir();(data/'manifests/training').mkdir(parents=True)
    monkeypatch.setattr(e,'FORMAL',formal);monkeypatch.setattr(e,'DATA',data);monkeypatch.setattr(e,'DOC',docs)
    monkeypatch.setattr(e,'source_state',lambda:'synthetic_source')
    monkeypatch.setattr(e,'REFERENCE',{p:1.0 for p in e.REFERENCE})
    write_json(docs/'R1_PREREGISTRATION.json',original_prereg)
    write_json(docs/'SCORING_CONTRACT_R1.json',{'synthetic':True})
    write_json(out/'A1_RELEASE.json',{'synthetic':True})
    rows=[]; originals=[]; manifests={s:[] for s in range(3)}
    hard=np.zeros((384,384),dtype=np.uint8);truth=hard.copy()
    p=np.zeros((3,384,384),dtype=np.float32);p[0]=1
    prepared=prepare_case(p,p,2)
    monkeypatch.setattr(e,'load_prepared',lambda output,index:prepared)
    for i in range(198):
        seed=i//66;d=(i%66)//22;case=str(i);patient=str(i%177)
        rows.append(dict(row_index=i,case_id=case,patient_id=patient,seed=seed,fold=i%5))
        t=np.full_like(truth,255) if i==0 else truth
        label=data/f'{i}.h5'
        with h5py.File(label,'w') as f:f.create_dataset('label',data=t,compression='gzip')
        manifests[seed].append(dict(case_id=case,patient_id=patient,primary_20pct_split='train_labeled',dataset='fundus',
             site_or_vendor=DOMAINS[d],label_h5_relpath=label.name,label_sha256=digest(label),image_h5_relpath='unused',image_sha256='unused'))
        for policy in (*e.BASE_POLICIES,'C7'):
            m=case_metrics(hard,t)
            if baseline_mismatch and i==0 and policy=='C0':m['rim_dice']=.9
            originals.append(dict(seed=seed,case_id=case,domain=DOMAINS[d],domain_index=d,policy=policy,**m))
    hashes={}
    for seed,records in manifests.items():
        # Unauthorized rows are excluded by the evaluator's selected-ID CSV projection.
        records.append(dict(records[-1],case_id='forbidden',primary_20pct_split='val',label_h5_relpath='DO_NOT_OPEN',site_or_vendor='FORBIDDEN'))
        path=data/'manifests/training'/f'lcrseg_v1_seed{seed}.csv';e.write_csv(path,records);hashes[str(seed)]=digest(path)
    (formal/'case_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in originals))
    names={};folds={}
    for fold in range(5):
        indices=[i for i in range(198) if i%5==fold];folds[str(fold)]=indices
        for policy in e.BASE_POLICIES:
            name=f'{fold}_{policy}.npy';np.save(formal/name,np.zeros((len(indices),384,384),dtype=np.uint8));names[f'{fold}:{policy}']=name
    entries={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in formal.iterdir()}
    monkeypatch.setattr(e,'original_inputs',lambda:({},entries))
    write_json(out/'input_metadata.json',dict(manifest_hashes=hashes,fold_indices=folds,control_paths=names))
    write_json(out/'blind_rows.json',rows)
    write_json(out/'blind_diagnostics.json',[dict(proposals=0,strict_actions=1,no_area_actions=1,free_actions=1,quantity_rejected=0,foreground_area_rejected=0,image_area_rejected=0,zero_current_foreground=True) for _ in rows])
    write_json(out/'ACTION_SPACE_SEAL.json',dict(status='PASS_ACTION_SPACE_SEALED_BEFORE_GT',source_commit='synthetic_source',
        protocol_sha256=digest(docs/'R1_PREREGISTRATION.json'),scoring_contract_sha256=digest(docs/'SCORING_CONTRACT_R1.json'),
        A1_release_sha256=digest(out/'A1_RELEASE.json'),files={p.name:digest(p) for p in out.iterdir()},original_files={p:ent['sha256'] for p,ent in entries.items()}))
    if baseline_mismatch:
        with pytest.raises(ValueError,match='BASELINE_MISMATCH'):e.evaluate(out)
        status=json.load(open(out/'public/CAPACITY_STATUS.json'))
        assert status['status']=='BLOCKED_BASELINE_MISMATCH'
        assert not (out/'public/CAPACITY_METRICS.csv').exists()
    else:
        e.evaluate(out)
        status=json.load(open(out/'public/CAPACITY_STATUS.json'))
        assert status['status']=='FAIL_FROZEN_ACTION_SPACE_CAPACITY' and status['patients']==177
        runtime=json.load(open(out/'public/RUNTIME_COUNTERS.json'))
        assert runtime['GT_reads']==runtime['domain_reads']==runtime['scientific_rows_completed']==198
        assert runtime['new_sample_expert_forwards']==runtime['real_router_fits']==0
        assert (out/'public/EVALUABLE_ONLY_SENSITIVITY.csv').is_file()
        with pytest.raises(FileExistsError):e.evaluate(out)
