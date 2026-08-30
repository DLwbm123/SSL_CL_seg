import json
import os
from pathlib import Path

import pytest
import torch

from di_dmpa_jascl.config import sha256_file
from di_dmpa_jascl.checkpoint import save_checkpoint
from di_dmpa_jascl.gradient_audit import GradientAuditPolicy
from di_dmpa_jascl.metrics import write_json
from di_dmpa_jascl.runner import Gate0RepairedRunner
from scripts.verify_resume_equivalence import GROUPS, compare
from tests.gate0.test_resume_v2 import synthetic_bundle

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_batch_selection_and_empty_default():
    point = dict(domain="REFUGE", epoch=25, batch_index=1, global_step=53)
    assert GradientAuditPolicy.from_mapping().should_audit(**point)
    assert not GradientAuditPolicy.from_mapping(dict(mode="fixed_batches",interval=0,fixed_batch_ids=[])).should_audit(**point)
    policy = GradientAuditPolicy.from_mapping(dict(mode="fixed_batches",interval=0,
        fixed_batch_ids=["REFUGE/epoch25/unlabeled_batch1"]))
    assert policy.should_audit(**point)


@pytest.mark.parametrize("payload", [{"mode":"bogus","interval":0,"fixed_batch_ids":[]},
    {"mode":"disabled","interval":1,"fixed_batch_ids":[]},
    {"mode":"fixed_batches","interval":0,"fixed_batch_ids":["x","x"]}])
def test_audit_policy_rejects_invalid_config(payload):
    with pytest.raises(ValueError):
        GradientAuditPolicy.from_mapping(payload)


def test_audit_on_off_trajectory_exact_parity(tmp_path):
    """Fresh synthetic model copies only; never loads/updates a frozen baseline."""
    config, protocol = synthetic_bundle(tmp_path / "data")
    device = os.environ.get("GATE1_PARITY_DEVICE", "cpu")
    runners = {}
    initial_path = tmp_path / "shared_synthetic_initial.pt"
    for mode in ("every_batch", "disabled"):
        runner = Gate0RepairedRunner(repo_root=ROOT, config=config, protocol=protocol, seed=0,
            output_dir=tmp_path/mode, device=device,
            gradient_audit=dict(mode=mode,interval=0,fixed_batch_ids=[]))
        if mode == "every_batch":
            # The official module import resets global RNG. Both paths must
            # start from the same explicit model/optimizer/RNG snapshot,
            # independent of cold-vs-cached import order.
            save_checkpoint(initial_path, runner._checkpoint_payload())
        runner.run(resume_path=initial_path, stop_after_global_step=55)
        runners[mode] = runner
    paths = [tmp_path/mode/"last.pt" for mode in ("every_batch", "disabled")]
    left,right = [torch.load(p,map_location="cpu",weights_only=False) for p in paths]
    assert left["prototypes"] is not None and left["sampler_state"]["phase"] == "supervised"
    groups = {}
    for name in GROUPS:
        ok, maximum = compare(left[name],right[name],atol=0,rtol=0)
        assert ok and maximum == 0, name
        groups[name] = dict(within_tolerance=ok,max_abs_difference=maximum)
    x = torch.arange(768,device=device).float().reshape(1,3,16,16)/768
    with torch.no_grad():
        a = runners["every_batch"].wrapper.student(x,stochastic_classifier=False)[0]
        b = runners["disabled"].wrapper.student(x,stochastic_classifier=False)[0]
    assert torch.equal(a,b)
    groups["deterministic_logits"] = dict(within_tolerance=True,max_abs_difference=0.0)
    counts = {}
    logs = {}
    for mode in runners:
        log = [json.loads(line) for line in (tmp_path/mode/"train.jsonl").read_text().splitlines()]
        pas = [row for row in log if row["phase"] == "unlabeled"]
        counts[mode] = sum(row["gradient_audit_executed"] for row in pas)
        logs[mode] = log
        assert all(row["pas_joint_valid_pixels"] > 0 and row["loss_consistency"] > 0 for row in pas)
        assert all((row["student_unsupervised_gradient_norm"] is None) == (mode == "disabled") for row in pas)
    assert counts["every_batch"] > 0 and counts["disabled"] == 0
    assert [row["loss_total"] for row in logs["every_batch"]] == [row["loss_total"] for row in logs["disabled"]]
    report = dict(status="PASS",atol=0,rtol=0,groups=groups,device=device,
        model="actual_UNet2D_JASCL_3x3",data_kind="synthetic_hashed_hdf5",audit_counts=counts,
        synthetic_fixture_optimizer_steps_per_path=55,model_optimizer_steps_on_frozen_baselines=0,
        frozen_baseline_checkpoint_loaded=False,
        shared_initial_checkpoint_sha256=sha256_file(initial_path),
        nonzero_pas_coverage_and_loss=True, full_loss_trajectory_exact=True,
        checkpoint_sha256={str(p):sha256_file(p) for p in paths},
        scope="synthetic integration test only; not Gate 1 model training")
    if os.environ.get("GATE1_PARITY_REPORT"):
        write_json(Path(os.environ["GATE1_PARITY_REPORT"]),report)
