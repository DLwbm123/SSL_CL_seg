import ast
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest
import torch

from di_dmpa_gate1c_v2 import binding as b
from di_dmpa_gate1c_v3 import durable as d
from di_dmpa_gate1c_v3.archive import promote
from pmgc_v0_1 import protocol as p, core as c


def test_base_report_status_hash_and_dual_resolution_binding():
    reg, _=p.authority()
    assert reg["base_commit"]=="3126e59a63b205f2a075f28efa9f5d83b3911792"
    status=d.read(p.ROOT/"docs/mmpr_gs_v0_1/MMPR_GS_STATUS.json")
    assert status["status"]=="FAIL_MATCHED_MASS_RANKING_NOT_SUPPORTED"
    tests=d.read(p.ROOT/"docs/mmpr_gs_v0_1/MMPR_GS_TEST_REPORT.json")
    assert tests["tests"]==326 and tests["failures"]==tests["errors"]==tests["skips"]==0
    archive=d.read(p.ROOT/"docs/mmpr_gs_v0_1/MMPR_GS_PRIVATE_ARCHIVE_AUDIT.json")
    assert archive==reg["inputs"]["mmpr_private_archive"]
    assert archive["total_real_forwards"]==375


def test_old_line_closure_cannot_reopen_and_Q1_Q2_evidence_retained():
    reg,_=p.authority();closure=d.read(p.REPO/reg["closure"]["files"]["json"]["path"])
    assert closure["prototype_selection_status"]==closure["prototype_weighting_status"]=="FAIL"
    assert not closure["additional_selection_attempts_authorized"] and closure["relation_method_status"]=="FROZEN_FAILED"
    text=(p.REPO/reg["closure"]["files"]["md"]["path"]).read_text()
    assert "Q1" in text and "Q2" in text


def test_base_mutation_is_blocked():
    original=d.read
    def changed(path):
        value=original(path)
        if Path(path).name=="PMGC_V0_1_FEASIBILITY_PREREGISTRATION.json":value["base_commit"]="0"*40
        return value
    with patch.object(d,"read",changed),pytest.raises(c.Blocked):p.authority()


def test_separate_published_authorization():
    reg,_=p.authority();auth=d.read(p.DOCS/"PMGC_V0_1_EXECUTION_AUTHORIZATION.json")
    assert p.REG_COMMIT!=p.AUTH_COMMIT and auth["preregistration_remote_verified_commit"]==p.REG_COMMIT
    assert reg["publication_barrier"]["real_model_forwards"]==reg["publication_barrier"]["autograd_calls"]==0


def test_integration_six_unit_coverage_and_48_pair_identity():
    reg, inherited=p.authority();units=reg["fixed_units"]
    assert len(units)==6 and len({u["integration_pair_id"] for u in units})==6
    assert [x for u in units for x in u["formal_pairs"]]==[x for x in inherited["gradient_diagnostic"]["batch_pairs"] if x["stage_index"] in (1,2)]
    assert all(len(u["formal_pairs"])==8 for u in units)
    assert all(len(v["batches"])==8 and len(set(v["case_ids"]))==16 for u in units for v in u["validation"].values())


def test_guard_and_all_classifier_seeds_itemized():
    reg,_=p.authority()
    assert sum(len(u["guard_batches"]) for u in reg["fixed_units"])==39
    for unit in reg["fixed_units"]:
        assert unit["pixel_sampling_seed"]>=0 and len(unit["clustering_seeds"])==30
        for batch in unit["guard_batches"]:
            assert batch["student_classifier_seed"]>=0 and batch["old_classifier_seed"]>=0


@pytest.mark.parametrize("action", ["optimizer","backward"])
def test_runtime_guards_prevent_optimizer_and_backward(action):
    with b.no_updates(),pytest.raises(b.ProtocolError):
        if action=="optimizer":torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.1)
        else:torch.ones(1,requires_grad=True).sum().backward()


def test_production_has_no_grad_writes_or_backward_or_optimizer():
    for path in (p.ROOT/"pmgc_v0_1").glob("*.py"):
        if path.name=="testing.py":continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                attr=node.func.attr if isinstance(node.func,ast.Attribute) else ""
                assert attr not in ("backward","step","zero_grad","update_teacher","update_gas")
            if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                assert not any(isinstance(t,ast.Attribute) and t.attr=="grad" for t in targets)


def test_create_only_state_and_manifest_exact_coverage(tmp_path):
    payload=tmp_path/"a.json";d.write_new(payload,dict(status="sealed"))
    with pytest.raises(FileExistsError):d.write_new(payload,dict(status="overwrite"))
    manifest=d.seal(tmp_path);assert d.verify(tmp_path)["content_sha256"]==manifest["content_sha256"]
    (tmp_path/"extra").write_text("unregistered")
    with pytest.raises(RuntimeError):d.verify(tmp_path)


def test_private_archive_full_hash_and_preserved_source(tmp_path):
    source=tmp_path/"source";source.mkdir();(source/"evidence").write_bytes(b"unmodified")
    d.seal(source);digest=d.sha256(source/"PRIVATE_BUNDLE_MANIFEST.json")
    import shutil
    incoming=tmp_path/"incoming";shutil.copytree(source,incoming)
    receipt=promote(incoming,tmp_path/"archives",digest)
    assert receipt["status"]=="PASS_PRIVATE_ARCHIVE" and source.exists()
    assert d.verify(receipt["archive"])["content_sha256"]==receipt["content_sha256"]


def test_durable_actual_child_exit_receipt(tmp_path):
    output=tmp_path/"phase"
    d.launch(output,"test",[sys.executable,"-c","print('synthetic child')"],cwd=tmp_path)
    deadline=time.monotonic()+15
    while not (output/"PHASE_test_MANIFEST.json").exists() and time.monotonic()<deadline:time.sleep(.02)
    assert d.read(output/"PROCESS_EXIT.json")["actual_child_exit_code"]==0
    assert d.read(output/"EXECUTION_COMPLETION.json")["status"]=="COMMAND_COMPLETED"
    d.verify(output,"PHASE_test_MANIFEST.json")


def test_missing_durable_completion_cannot_admit_phase(tmp_path):
    with pytest.raises(FileNotFoundError):p.completed(tmp_path,"formal","result.json")


def test_NAS_only_and_no_new_line_after_failure():
    reg,_=p.authority()
    assert reg["destination"]["root"].startswith("/data_nas/") and not reg["destination"]["home_fallback"]
    assert reg["destination"]["allowed_physical_gpus"]==[4,5,6,7]
    assert not reg["hard_stop"]["additional_variants_authorized"] and not reg["hard_stop"]["training_authorized"]
