"""Synthetic full production target -> fit -> deploy -> seal -> report witness.
Only external fixture roots, expected baseline values and cache storage provider
are substituted. Actual label loader, scoring, nesting, serialization and seals run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
import pytest
from care_hr_v0_7_1.io_r1 import write_json,digest
from care_hr_v0_7_1.scoring_r1 import case_metrics
from care_hr_v0_7_1.summary_r1 import legacy_aggregate
from shor_uv_v0_8 import pipeline as p
from shor_uv_v0_8 import report as rep


def test_full_production_pipeline(tmp_path,monkeypatch):
    output=tmp_path/'run';output.mkdir();(output/'public').mkdir()
    data=tmp_path/'DATA';(data/'h5/v1').mkdir(parents=True);(data/'manifests/training').mkdir(parents=True)
    formal=tmp_path/'formal';formal.mkdir();parent=tmp_path/'parent';parent.mkdir();r2=tmp_path/'r2';r2.mkdir()
    doc=tmp_path/'docs';doc.mkdir()
    write_json(doc/'HISTORY_PROTECTION.json',{})
    for module in (p,rep):
        monkeypatch.setattr(module,'FORMAL',formal)
        monkeypatch.setattr(module,'DOC',doc)
        monkeypatch.setattr(module,'PARENT',parent)
        monkeypatch.setattr(module,'R2',r2)
        monkeypatch.setattr(module,'source_state',lambda:'synthetic-exact-functions')
    monkeypatch.setattr(p,'DATA',data)
    # Canonical fixtures do not call the production resolver.
    hard=np.zeros((384,384),np.uint8);hard[80:300,80:300]=1;hard[140:230,140:230]=2
    prob=np.eye(3,dtype=np.float32)[hard].transpose(2,0,1)
    cache={(s,h):np.broadcast_to(prob,(66,3,384,384)) for s in range(3) for h in range(3)}
    monkeypatch.setattr(p,'probabilities',lambda meta:cache)
    monkeypatch.setattr(rep,'probabilities',lambda meta:cache)
    rows=[];assets=[];folds={};domain_rows={s:[] for s in range(3)}
    for i in range(198):
        patient=i%177;s=i//66;domain=patient%3
        rel=f'label{patient}.h5';path=data/'h5/v1'/rel
        if not path.exists():
            with h5py.File(path,'w') as f:f.create_dataset('label',data=hard,compression='gzip')
        r=dict(row_index=i,case_id=f'case{patient}',patient_id=f'patient{patient}',seed=s,fold=patient%5,
               alpha=[.1,.2,.7],image_h5_relpath=f'not_opened{patient}',image_sha256='a'*64)
        rows.append(r)
        a=dict(case_id=r['case_id'],patient_id=r['patient_id'],primary_20pct_split='train_labeled',
               label_h5_relpath=rel,label_sha256=digest(path),seed=s,row_index=i)
        assets.append(a);domain_rows[s].append({**a,'site_or_vendor':p.DOMAINS[domain]})
    for s,rr in domain_rows.items():
        import csv
        with (data/'manifests/training'/f'lcrseg_v1_seed{s}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    indices={str(f):[i for i,r in enumerate(rows) if r['fold']==f] for f in range(5)}
    controls={};original=[]
    m=case_metrics(hard,hard)
    for f in range(5):
        for k in p.BASE_POLICIES:
            name=f'fold{f}_{k}.npy';np.save(formal/name,np.broadcast_to(hard,(len(indices[str(f)]),384,384)))
            controls[f'{f}:{k}']=name
        for i in indices[str(f)]:
            r=rows[i]
            for k in p.BASE_POLICIES:
                original.append({**r,'policy':k,'route':2,'domain':p.DOMAINS[(i%177)%3],
                                 'domain_index':(i%177)%3,**m})
    with (formal/'case_metrics.jsonl').open('w') as f:
        for r in original:f.write(json.dumps(r)+'\n')
    monkeypatch.setattr(p,'original_inputs',lambda:({},{}))
    monkeypatch.setattr(p,'check_original',lambda entries,name:formal/name)
    monkeypatch.setattr(p,'REFERENCE',{v:1. for v in p.BASE_POLICIES.values()})
    meta=dict(fold_indices=indices,control_paths=controls,probability_positions={str(i):i%66 for i in range(198)})
    np.save(parent/'routes.npy',np.full(198,2))
    np.save(output/'SHOR_routes.npy',np.full(198,2))
    rng=np.random.default_rng(14);np.save(output/'features.npy',rng.normal(size=(198,2,18)))
    write_json(output/'rows.json',rows);write_json(output/'meta.json',meta);write_json(output/'assets.json',assets)
    for f in range(5):
        folds[str(f)]=dict(outer_train_patients=sorted({r['patient_id'] for r in rows if r['fold']!=f}),
                          outer_test_patients=sorted({r['patient_id'] for r in rows if r['fold']==f}))
    write_json(output/'folds.json',folds)
    write_json(output/'history_private.json',dict(parent_files={},R2_files={}))
    p.seal(output,'INPUT_SEAL.json',[output/n for n in ('rows.json','meta.json','assets.json','features.npy','SHOR_routes.npy','folds.json')])
    p.targets(output)
    original_read=p.read_json
    for f in range(5):
        def scoped_read(path):
            assert Path(path).name not in ('evaluation_targets.json','baseline_records.json')
            if Path(path).name.startswith('train_targets_fold'):
                assert Path(path).name==f'train_targets_fold{f}.json'
            return original_read(path)
        monkeypatch.setattr(p,'read_json',scoped_read)
        p.train_fold(output,f)
        def blind_read(path):
            assert not Path(path).name.startswith('train_targets_fold')
            assert Path(path).name not in ('evaluation_targets.json','baseline_records.json','INNER_SELECTION.json')
            return original_read(path)
        monkeypatch.setattr(p,'read_json',blind_read)
        p.deploy_fold(output,f)
    monkeypatch.setattr(p,'read_json',original_read)
    p.seal_all(output)
    rep.evaluate(output)
    status=json.loads((output/'public/STATUS.json').read_text())
    assert status['all_outer_rows']==198 and status['outer_new_policy_rows']==792
    assert status['status']=='DEVELOPMENT_UTILITY_SIGNAL_NOT_ESTABLISHED'
    count=json.loads((output/'TARGET_COUNTERS.json').read_text())
    assert count['GT_payload_reads_completed']==198 and count['GT_unique_files_decoded']==177
    fits=json.loads((output/'public/FIT_ACCOUNTING.json').read_text())
    assert fits['actual_shared_design_solves']==85
    # Routed-only has no training pairs and must not claim a successful fit.
    assert all(v['actual_design_solves']==0 for v in fits['fits'] if v['variant']=='routed_only')
    assert (output/'public/FINAL_REPORT.md').is_file()
