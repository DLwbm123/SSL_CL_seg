import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.lcrseg.five_frameworks_v1.gate import require_approval,ReviewRequired,BINDINGS,CAPS
from experiments.lcrseg.five_frameworks_v1.planner import expand,expand_baselines,dag,validate_dag,effective_aliases,cost_plan,write
from experiments.lcrseg.five_frameworks_v1.analyze import sequence_metrics,select_candidate,paired_patient_bootstrap
from experiments.lcrseg.five_frameworks_v1.evaluate import binary_metrics
from experiments.lcrseg.five_frameworks_v1.model import Model
from experiments.lcrseg.five_frameworks_v1.parent_bridge import SyntheticParentBridge


PLAN=Path('experiments/lcrseg/docs/five_frameworks_v1/delivery/configs/study_plan.json')


def test_complete_DAG_and_budget(tmp_path):
    p=json.loads(PLAN.read_text());cs=expand(p);bs=expand_baselines(p)
    assert len(cs)==40 and len(bs)==32 and len(p['parent_configs'])==4
    nodes=dag(p);targets=[n for n in nodes if n['kind']=='target_stage']
    assert len(targets)==424 and len([n for n in nodes if n['kind']=='source'])==4
    assert len([n for n in targets if n['phase']=='C'])==304
    assert len([n for n in targets if n['phase']=='D'])==120
    assert all(n['resolved_candidate'] is None for n in nodes if n['kind']=='selection' and 'resolved_candidate' in n)
    assert all(n['updates'] is None for n in targets)
    c=cost_plan(p,[3200,2100]);assert c['formal_updates']==1123600
    assert c['F2_probe_VJPs']==1056 and c['F4_inner_VJPs']==279840 and c['F5_extra_clean_U_forwards']==93280
    assert c['F4_readout_only_upper_bound']==466400
    assert cost_plan(p)['formal_updates'] is None
    write(p,tmp_path);assert (tmp_path/'TASK_MATRIX_PREVIEW.csv').is_file()
    targets[-1]['parent_checkpoint']=targets[0]['id']
    with pytest.raises(ValueError):validate_dag(nodes)


def test_alias_rank_clamping_and_degenerate_spectrum():
    p=json.loads(PLAN.read_text());r=effective_aliases(expand(p),2)
    assert r['aliases'] and effective_aliases(expand(p))['status']=='PENDING_PARENT_FEATURE_WIDTH'
    parent=SyntheticParentBridge()
    with torch.no_grad():parent.readout.weight.zero_()
    m=Model(parent,'F1');assert m.spectral['proxy_degenerate'] and torch.equal(m.sidecar.q,torch.eye(4)[:,:2])
    assert Model(SyntheticParentBridge(),'F1',ratio=.0001).sidecar.q.shape[1]==1
    assert Model(SyntheticParentBridge(),'F1',ratio=1.).sidecar.q.shape[1]==3
    with pytest.raises(ValueError):Model(SyntheticParentBridge(d=1,rank=1),'F1')


def receipt_fixture():
    # Test fixture is never written as an operator approval or used by a launcher.
    actual={k:'a'*(40 if k=='reviewed_code_commit' else 64) for k in BINDINGS}
    caps={k:0 for k in CAPS}
    receipt={**actual,**caps,'is_template':False,'decision':'APPROVED_FOR_EXPERIMENTS',
             'user_launch_confirmation':True,'review_evidence':'SYNTHETIC_FIXTURE_ONLY','reviewer':'unit-test',
             'approved_phases':['B','C','D']}
    return receipt,actual,caps


@pytest.mark.parametrize('key',BINDINGS+CAPS)
def test_every_approval_binding_and_cap_required(key):
    r,a,c=receipt_fixture();r.pop(key)
    with pytest.raises(ReviewRequired):require_approval(r,a,['B'],c,{'status':'BOUND_VERIFIED'})


@pytest.mark.parametrize('change',[{'is_template':True},{'review_evidence':None},{'user_launch_confirmation':False},
                                   {'reviewer':None},{'decision':'NOT_REVIEWED'},{'max_real_L_smoke_updates':25}])
def test_approval_fields_fail_closed(change):
    r,a,c=receipt_fixture();r.update(change)
    with pytest.raises(ReviewRequired):require_approval(r,a,['B'],c,{'status':'BOUND_VERIFIED'})


def test_phase_parent_gate_and_CLI_before_data_access(tmp_path):
    r,a,c=receipt_fixture()
    for phases,parent in [(['E'],{'status':'BOUND_VERIFIED'}),(['C'],{'status':'PARENT_BINDING_REQUIRED'})]:
        with pytest.raises(ReviewRequired):require_approval(r,a,phases,c,parent)
    require_approval(r,a,['B'],c,{'status':'BOUND_VERIFIED'})
    nonexistent=tmp_path/'must_not_be_opened.json'
    code='import sys,runpy;sys.argv='+repr(['cli','run','--approval',str(nonexistent),'--phases','B'])+";runpy.run_module('experiments.lcrseg.five_frameworks_v1.cli',run_name='__main__')"
    result=subprocess.run([sys.executable,'-'],input=code,capture_output=True,text=True)
    assert result.returncode!=0 and 'PARENT_BINDING_REQUIRED' in result.stderr and 'FileNotFound' not in result.stderr


def test_metrics_negative_forgetting_and_empty_distances():
    r=np.array([[.7,np.nan,np.nan],[.8,.6,np.nan],[.9,.8,.7]])
    m=sequence_metrics(r)
    assert m['Forget']<0 and m['BWT']>0 and abs(m['Final']-(2*m['Old']+m['Incoming'])/3)<1e-12
    z=np.zeros((5,5),bool);one=z.copy();one[2,2]=1
    assert binary_metrics(z,z)['Dice']==1 and binary_metrics(z,z)['HD95']==0
    assert np.isinf(binary_metrics(z,one)['HD95']) and binary_metrics(z,one)['distance_unit']=='pixels'


def test_selection_negative_results_allowed_and_paired_bootstrap():
    rows=[{'candidate_id':'a','Final':-.2,'Old':-.1,'sealed_all_orders':True},
          {'candidate_id':'b','Final':-.20005,'Old':0.,'sealed_all_orders':True}]
    assert select_candidate(rows)['candidate_id']=='b'
    values=np.arange(5,dtype=float)[None,None,None,:]*np.ones((3,2,2,1))
    got=paired_patient_bootstrap({'REFUGE':values,'RIM':values},repetitions=2000)
    assert got['replicates'].shape==(2000,3,2,2)
    assert np.array_equal(got['replicates'][:,0,0],got['replicates'][:,2,1])


def test_locked_source_setup_reuses_external_files(dependency_root):
    from experiments.lcrseg.five_frameworks_v1.fetch_dependencies import fetch
    result=fetch(dependency_root)
    assert result=={'source_files':4,'images':0,'checkpoints':0}
    with pytest.raises(ValueError):fetch(Path.cwd())
