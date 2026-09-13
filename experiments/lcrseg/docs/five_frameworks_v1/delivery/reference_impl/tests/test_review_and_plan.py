import importlib.util,json
from pathlib import Path
import pytest
from sslcl5.review_gate import require_execution_approval,ReviewRequired
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('make_plan',ROOT/'tools/make_plan.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_plan_all_five_have_eight_candidates():
    p=json.loads((ROOT/'configs/study_plan.json').read_text());rows=mod.expand(p)
    assert len(rows)==40
    assert {f:sum(r['family']==f for r in rows) for f in p['priority_order']}==dict.fromkeys(p['priority_order'],8)

def test_budget_counts_sources_separately():
    p=json.loads((ROOT/'configs/study_plan.json').read_text());b=mod.budget(p,[3200,2100])
    assert (b['phase_C_sequences'],b['phase_D_sequences'])==(152,60)
    assert b['incremental_optimizer_updates']==1123600
    assert b['source_updates_if_no_legal_reuse'] is None
    assert b['optional_phase_E_updates_not_authorized']==254400

def test_template_refuses_training():
    r=json.loads((ROOT/'configs/review_approval.template.json').read_text())
    with pytest.raises(ReviewRequired):require_execution_approval(r,{},['C'],1)

def test_genuine_fixture_hash_and_scope_checks():
    actual=dict.fromkeys(('reviewed_code_commit','reviewed_code_tree_sha256','parent_binding_sha256',
                'expanded_plan_sha256','external_dependency_lock_sha256'),'test-fixture-only')
    r={**actual,'is_template':False,'decision':'APPROVED_FOR_EXPERIMENTS','user_launch_confirmation':True,
       'review_evidence':'SYNTHETIC_FIXTURE_NOT_AN_APPROVAL','approved_phases':['B','C','D'],'max_formal_optimizer_updates':10}
    require_execution_approval(r,actual,['C'],10)
    with pytest.raises(ReviewRequired):require_execution_approval(r,actual,['E'],10)
    with pytest.raises(ReviewRequired):require_execution_approval(r,actual,['C'],11)
    with pytest.raises(ReviewRequired):require_execution_approval(r,{**actual,'reviewed_code_commit':'changed'},['C'],1)
