"""Small synthetic files exercise cache ->72 census -> geometry -> report."""
import copy
from pathlib import Path
import numpy as np

from di_dmpa_gate1.sampling import sample_layout
from di_dmpa_gate1_v2.binding import H, PANELS, sha256
from di_dmpa_gate1_v2.features import split_support, weight_hash
from di_dmpa_gate1_v2.census import validate_features, compile_census
from di_dmpa_gate1_v2.runner import geometry_job
from di_dmpa_gate1_v2.reporting import report
from di_dmpa_gate1.binding import read_json


def test_synthetic_cache_census_geometry_report_roundtrip(tmp_path):
    cases=[dict(case_id=name,classes=[dict(coordinates=[[c,0],[c,1]],sampled_pixels=2,boundary=[True,False]) for c in range(3)]) for name in ('a','b')]
    units=[dict(seed=s,stage_index=t,role=r,cases=copy.deepcopy(cases)) for s in range(3) for t in range(3) for r in ('train_labeled','val')]
    plan={'units':units};meta={'sampling_plan_sha256':'a'*64,'diagnostic_code_git_commit':'synthetic'}
    raw=np.zeros((4,16));raw[[0,2],0]=1;raw[3,1]=1
    arrays=split_support(raw);caches=[]
    for c in range(3):
        layout=sample_layout(units[0],c)
        cache=dict(class_id=c,registered_count=4,active_count=3,null_count=1,uid_order_sha256=H(layout['uids']),original_weight_order_sha256=weight_hash(layout['weights']),arrays={})
        for name,array in arrays.items():
            path=tmp_path/f'class{c}_{name}.npy';np.save(path,array,allow_pickle=False)
            cache['arrays'][name]=dict(path=path.name,shape=list(array.shape),dtype=str(array.dtype),sha256=sha256(path))
        caches.append(cache)
    entries=[]
    for p in PANELS:
        for u in units:
            rows=[]
            for i,case in enumerate(cases):
                for c in range(3):
                    rows.append(dict(panel_id=p,seed=u['seed'],stage_index=u['stage_index'],domain=str(u['stage_index']),role=u['role'],case_id=case['case_id'],class_id=c,
                        registered_count=2,active_count=1 if i==0 else 2,null_count=1 if i==0 else 0,null_fraction=.5 if i==0 else 0.,
                        weighted_null_mass=.25 if i==0 else 0.,registered_nonfinite_count=0,full_map_nonfinite_count=0))
            entries.append(dict(panel_id=p,seed=u['seed'],stage_index=u['stage_index'],domain=str(u['stage_index']),role=u['role'],sampling_plan_sha256='a'*64,
                metadata=meta,all_finite=True,null_rows_preserved=True,old_raw_cache_reused=False,sampling_unit_sha256=H(u),class_caches=caches,case_support=rows))
    validate_features(tmp_path,entries,plan,meta)
    census=compile_census(tmp_path,entries,meta)
    assert census['null_count']==72*3 and census['active_count']==72*9
    old={'shared_sampling':{'plans':[dict(seed=0,stage_index=0,bootstrap=[dict(replicate=i,case_ids_with_replacement=['a','b'],case_draw_sha256=H(['a','b'])) for i in range(5)])]}}
    es={r:next(e for e in entries if e['panel_id']=='B0-EMA' and e['seed']==e['stage_index']==0 and e['role']==r) for r in ('train_labeled','val')}
    us={r:next(u for u in units if u['seed']==u['stage_index']==0 and u['role']==r) for r in ('train_labeled','val')}
    prototypes={K:read_json(geometry_job((str(tmp_path),old,meta,es,us,1,K))) for K in (1,2,3,5)}
    rows=[]
    for p in PANELS:
        for s in range(3):
            for t in range(3):
                for c in range(3):
                    for K in (1,2,3,5):
                        row=copy.deepcopy(prototypes[K]);row.update(panel_id=p,seed=s,stage_index=t,class_id=c)
                        rows.append(row)
    status=report(tmp_path,meta,rows,census)
    assert status['geometry_jobs_completed']==432 and status['selected_K']==1
    assert status['prototype_geometry_status']=='FAIL_MULTI_MODALITY_NOT_SUPPORTED'
    assert len(read_json(tmp_path/'PROTOTYPE_GEOMETRY_DIAGNOSTIC_V2.json')['foreground_macro_units'])==288
