from __future__ import annotations

from pathlib import Path

import pytest

from di_dmpa_jascl.config import load_yaml
from di_dmpa_jascl.data import LCRSegH5Dataset, collate
from di_dmpa_jascl.manifest import LCRSegManifestAdapter


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/root/LCRSeg")


@pytest.fixture(scope="module")
def adapter():
    if not DATA_ROOT.is_dir():
        pytest.skip("frozen LCRSeg data are available only on the experiment node")
    protocol = load_yaml(ROOT / "docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml")
    return LCRSegManifestAdapter(DATA_ROOT, protocol, seed=0, benchmark="fundus")


def test_current_domain_only_and_hidden_gt_isolation(adapter) -> None:
    labeled = adapter.records(domain="REFUGE", role="train_labeled", purpose="train")
    unlabeled = adapter.records(domain="REFUGE", role="train_unlabeled", purpose="train")
    adapter.assert_current_domain_only([*labeled, *unlabeled], "REFUGE")
    assert all(record.label_h5_relpath is None for record in unlabeled)
    assert adapter.leakage_audit()["hidden_gt_training_usage"] == "none"


def test_training_cannot_request_val_or_test(adapter) -> None:
    for role in ("val", "test"):
        with pytest.raises(RuntimeError):
            adapter.records(domain="REFUGE", role=role, purpose="train")


def test_unlabeled_batch_has_no_label_key(adapter) -> None:
    records = adapter.records(domain="REFUGE", role="train_unlabeled", purpose="train")
    dataset = LCRSegH5Dataset(DATA_ROOT, records[:1], require_label=False, output_hw=(384, 384))
    batch = collate(dataset, [0], require_label=False)
    assert "label" not in batch
    assert batch["domain"] == ["REFUGE"]
