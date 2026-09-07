import copy
import json
from pathlib import Path
import numpy as np
import pytest
from care_hr_v0_7.proposals import generate_proposals, _sort_key
from care_hr_v0_7_1.actions import enumerate_actions
from care_hr_v0_7_1.io_r1 import BLIND_COLUMNS, digest, project_training, validate_population, verify_files, write_json
from care_hr_v0_7_1.summary_r1 import balanced, legacy_aggregate, summarize, terminal, DOMAINS
from test_actions import regions


def test_r1_preserves_every_predecessor_document_and_all_82_old_protections():
    root=Path(__file__).resolve().parents[4]
    manifest=json.loads((root/'experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r1/HISTORY_PROTECTION_R1.json').read_text())
    assert manifest['original_protected_count']==82 and manifest['protected_count']==99
    verify_files(root,{r['path']:r['sha256'] for r in manifest['entries']})


def test_zero_foreground_all_three_spaces_and_twelve_four_three_limits():
    for space in ("O_CAP", "O_NO_AREA", "O_FREE_SUBSET"):
        assert list(enumerate_actions(regions(12), np.zeros((80,80),dtype=int), space)) == [((),None)]


def test_original_sort_overlap_minimum_four_connected_and_cap():
    c=np.zeros((60,60),dtype=int); h=c.copy()
    for i in range(14): h[2*i,:8]=1
    h[40,:7]=1
    p=generate_proposals(c,h)
    assert len(p)==12 and [x.centroid_row for x in p]==list(range(0,24,2))
    c[:]=0; h[:]=0; c[:2,:8]=2; h[:2,:8]=1
    p=generate_proposals(c,h)
    assert p[0].target_class==1 and p[0].direction=='add' and p[0].area==16
    assert np.max(np.stack([x.mask for x in p]).sum(axis=0))==1
    base=dict(area=8,centroid_row=0,centroid_col=1,target_class=1,direction='add')
    assert _sort_key(base)<_sort_key(dict(base,direction='remove'))<_sort_key(dict(base,target_class=2))


def synthetic_population():
    rows=[]; manifests={s:[] for s in range(3)}
    for s in range(3):
        for i in range(2):
            r=dict(seed=s,case_id=str(i),patient_id='p'+str(i),row_index=len(rows),fold=i,
                   image_h5_relpath='images/'+str(i),image_sha256='hash'+str(i),alpha=[.2,.3,.5])
            rows.append(r); manifests[s].append({**{k:r[k] for k in BLIND_COLUMNS if k in r},'primary_20pct_split':'train_labeled'})
    return rows,manifests


def test_patient_cross_seed_mapping_and_own_seed_role_before_gt():
    rows,manifests=synthetic_population()
    assert validate_population(rows,manifests,6,2,2)['unique_cases']==2
    manifests[1][0]['primary_20pct_split']='val'
    with pytest.raises(ValueError,match='own-seed'): validate_population(rows,manifests,6,2,2)
    rows,manifests=synthetic_population(); rows[2]['patient_id']='wrong'
    with pytest.raises(ValueError): validate_population(rows,manifests,6,2,2)
    rows,manifests=synthetic_population(); rows[0]['GT']=object()
    with pytest.raises(ValueError,match='schema'): validate_population(rows,manifests,6,2,2)


def test_projected_csv_never_materializes_gt_or_domain(tmp_path):
    p=tmp_path/'mixed.csv'
    p.write_text(','.join(BLIND_COLUMNS)+',site_or_vendor,label_h5_relpath\na,p,train_labeled,img,sha,FORBIDDEN_DOMAIN,FORBIDDEN_GT\n')
    rows=project_training(p)
    assert set(rows[0])==set(BLIND_COLUMNS)
    assert 'FORBIDDEN' not in json.dumps(rows)


def test_atomic_seal_tamper_and_path_escape(tmp_path):
    p=tmp_path/'seal.json'; write_json(p,{'frozen':True})
    with pytest.raises(FileExistsError): write_json(p,{})
    files={'seal.json':digest(p)}; verify_files(tmp_path,files)
    p.write_text('{}')
    with pytest.raises(ValueError,match='hash'): verify_files(tmp_path,files)
    with pytest.raises(ValueError,match='escapes'): verify_files(tmp_path,{'../x':'bad'})


def metric_rows():
    rows=[]
    for s in range(3):
        for d in range(3):
            for policy in ('current','frozen_PPC_C6','O_CAP_envelope','O_SAFE0_envelope'):
                g=0 if policy=='current' else .3
                rows.append(dict(seed=s,domain=DOMAINS[d],domain_index=d,policy=policy,patient_id=f'p{s}{d}',
                                 foreground_dice=.5+g,rim_dice=.5+g,cup_dice=.5+g,mean_iou=.4+g,
                                 gain_macro_fg=g,gain_rim=g,gain_cup=g,harm=0,has_evaluable_gt=True))
    return rows


def test_full_legacy_aggregation_and_empty_group_sensitivity():
    rows=metric_rows(); modern=summarize(rows); old=legacy_aggregate(rows)
    lookup={(r['level'],r['key'],r['policy']):r for r in modern}
    for r in old:
        for f in ('foreground_dice','rim_dice','cup_dice','mean_iou'):
            assert r[f]==lookup[r['level'],r['key'],r['policy']][f]
    assert terminal(rows)=='PASS_ACTION_SPACE_SCREEN_ONLY'
    for r in rows:
        if r['seed']==0 and r['domain_index']==0:r['has_evaluable_gt']=False
    assert terminal(rows)=='BLOCKED_NO_EVALUABLE_SUPPORT'
    assert balanced([r for r in rows if r['policy']=='current'],'foreground_dice',True) is None
    assert balanced([r for r in rows if r['policy']=='current'],'foreground_dice')==.5


def test_scientific_gates_are_unrounded_and_relaxed_oracle_cannot_rescue():
    rows=metric_rows()
    for r in rows:
        if r['policy']=='O_CAP_envelope':r['gain_macro_fg']=np.nextafter(.17,0)
    assert terminal(rows)=='FAIL_FROZEN_ACTION_SPACE_CAPACITY'
    rows=metric_rows()
    for r in rows:
        if r['policy']=='O_SAFE0_envelope':r['gain_macro_fg']=.26
    assert terminal(rows)=='CAPACITY_PRESENT_SAFETY_UNRESOLVED'


def test_patient_equal_and_evaluable_preserve_original_group_weights():
    rows=metric_rows()[:1]
    rows[0]['foreground_dice']=0
    rows += [dict(rows[0],patient_id='q',foreground_dice=1),dict(rows[0],patient_id='q',foreground_dice=1)]
    assert balanced(rows,'foreground_dice')==2/3
    assert balanced(rows,'foreground_dice',patient_equal=True)==.5
    rows[0]['has_evaluable_gt']=False
    assert balanced(rows,'foreground_dice',evaluable_only=True)==1
