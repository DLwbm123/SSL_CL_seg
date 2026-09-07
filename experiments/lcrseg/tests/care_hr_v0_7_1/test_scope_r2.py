import json
from pathlib import Path
import pytest
from care_hr_v0_7_1.io_r1 import verify_files
from care_hr_v0_7_1 import recovery_r2 as r
from test_recovery_r2 import deployment, admitted

ROOT=Path(__file__).resolve().parents[4]


def test_all_150_protected_files_and_scientific_kernels_unchanged():
    m=json.loads((ROOT/'experiments/lcrseg/docs/care_hr_v0_7_1/continuation_r2/HISTORY_PROTECTION_R2.json').read_text())
    assert m['original_protected_count']==99 and m['protected_count']==150
    verify_files(ROOT,{e['path']:e['sha256'] for e in m['entries']})
    r.kernels()


def test_scientific_orchestration_copy_diff_is_only_io_and_counting():
    folder=ROOT/'experiments/lcrseg/care_hr_v0_7_1'
    old=(folder/'execute_r1.py').read_text();new=(folder/'execute_r2.py').read_text()
    start='        # This mixed historical score file';end='    except Exception as exc:'
    before=old[old.index(start):old.index(end,old.index(start))]
    after=new[new.index(start):new.index(end,new.index(start))]
    after=after.replace('case_policy_comparisons=len(baseline),scalar_comparisons=len(baseline)*4','per_case_comparisons=len(baseline)*4')
    after=after.replace("\n                    count['baseline_case_policy_comparisons_completed']+=1\n                    count['baseline_scalar_comparisons_completed']+=4",'')
    after=after.replace('prepared=load_prepared(recovery.PARENT,i)','prepared=load_prepared(output,i)')
    after=after.replace("diagnostics=read_json(recovery.PARENT/'blind_diagnostics.json')","diagnostics=read_json(output/'blind_diagnostics.json')")
    after=after.replace("\n                    count['oracle_case_rows_completed']+=1",'')
    assert after==before


def test_dual_source_seal_and_parent_tamper_rejected(deployment):
    base,out,unchanged=admitted(deployment)
    seal=out/'ACTION_SPACE_SEAL.json';original=seal.read_bytes();s=json.loads(original)
    assert s['evaluator_source_commit']!=s['action_generation_source_commit']
    s['action_generation_source_commit']=s['evaluator_source_commit'];seal.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='dual-source'):r.verify_recovery(out)
    seal.write_bytes(original)
    path=base/'parent/cases/000.json';path.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):r.verify_recovery(out)
